package adkanyllm

import (
	"context"
	"strings"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type streamToolCallState struct {
	args     *strings.Builder
	id       string
	toolType string
	name     string
}

func (m *Model) generateStream(
	ctx context.Context,
	params anyllm.CompletionParams,
	yield func(*model.LLMResponse, error) bool,
) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	params.StreamOptions = &anyllm.StreamOptions{IncludeUsage: true}

	chunks, errs := m.provider.CompletionStream(ctx, params)

	var (
		textBuilder      strings.Builder
		reasoningBuilder strings.Builder
		toolCallStates   []streamToolCallState
		finishReason     string
		usage            *anyllm.Usage
		modelVersion     string
	)

	for chunk := range chunks {
		if chunk.Model != "" {
			modelVersion = chunk.Model
		}
		if chunk.Usage != nil {
			usage = chunk.Usage
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		if choice.FinishReason != "" {
			finishReason = choice.FinishReason
		}

		if choice.Delta.Content != "" {
			textBuilder.WriteString(choice.Delta.Content)
			partial := &model.LLMResponse{
				Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						genai.NewPartFromText(choice.Delta.Content),
					},
				},
				Partial: true,
			}
			if !yield(partial, nil) {
				finishStream(chunks, errs, cancel)
				return
			}
		}

		if choice.Delta.Reasoning != nil && choice.Delta.Reasoning.Content != "" {
			reasoningBuilder.WriteString(choice.Delta.Reasoning.Content)
		}

		for _, toolCall := range choice.Delta.ToolCalls {
			if err := aggregateStreamToolCall(&toolCallStates, toolCall); err != nil {
				finishStream(chunks, errs, cancel)
				yield(nil, err)
				return
			}
		}
	}

	if err := <-errs; err != nil {
		yield(nil, wrapProviderError(err))
		return
	}

	content, err := contentFromStreamAggregation(textBuilder.String(), reasoningBuilder.String(), toolCallStates)
	if err != nil {
		yield(nil, err)
		return
	}

	usageMetadata, err := usageMetadataFromUsage(usage)
	if err != nil {
		yield(nil, err)
		return
	}

	final := &model.LLMResponse{
		Content:       content,
		UsageMetadata: usageMetadata,
		FinishReason:  mapFinishReason(finishReason),
		ModelVersion:  modelVersion,
		Partial:       false,
		TurnComplete:  true,
	}

	yield(final, nil)
}

func finishStream(
	chunks <-chan anyllm.ChatCompletionChunk,
	errs <-chan error,
	cancel context.CancelFunc,
) {
	cancel()
	// Drain remaining chunks to prevent provider goroutine leak.
	// Done synchronously — the consumer has already stopped yielding,
	// so there is nothing else to do. A background goroutine would risk
	// leaking if the provider closed errs before its chunk sender unblocked.
	for range chunks {
	}
	<-errs
}

func aggregateStreamToolCall(states *[]streamToolCallState, delta anyllm.ToolCall) error {
	if delta.ID != "" || delta.Function.Name != "" {
		*states = append(*states, streamToolCallState{
			id:       delta.ID,
			toolType: delta.Type,
			name:     delta.Function.Name,
			args:     new(strings.Builder),
		})
	}

	if len(*states) == 0 {
		return newError("stream tool call arguments without preceding tool call header")
	}

	// Dereference once instead of repeating &(*states)[n] everywhere.
	s := *states
	s[len(s)-1].args.WriteString(delta.Function.Arguments)
	return nil
}

func contentFromStreamAggregation(
	text string,
	reasoning string,
	toolCallStates []streamToolCallState,
) (*genai.Content, error) {
	parts := make([]*genai.Part, 0, 1+len(toolCallStates))

	if text != "" {
		parts = append(parts, genai.NewPartFromText(text))
	}
	if reasoning != "" {
		parts = append(parts, &genai.Part{
			Text:    reasoning,
			Thought: true,
		})
	}

	for _, state := range toolCallStates {
		part, err := functionCallPart(state.id, state.toolType, state.name, state.args.String())
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}

	if len(parts) == 0 {
		return nil, nil
	}

	return &genai.Content{
		Role:  genai.RoleModel,
		Parts: parts,
	}, nil
}
