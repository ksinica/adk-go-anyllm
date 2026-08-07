package adkanyllm

import (
	"context"
	"slices"
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
	includeThoughts bool,
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
		streamErr        error
	)

	// Read chunks and errs concurrently rather than draining chunks to
	// completion before checking errs: a provider using an unbuffered errs
	// channel that sends its terminal error before closing chunks would
	// otherwise deadlock (its send blocks forever because nothing reads
	// errs until the chunks range finishes, which never happens because
	// the provider goroutine is itself stuck on that blocked send).
	for chunks != nil || errs != nil {
		select {
		case chunk, ok := <-chunks:
			if !ok {
				chunks = nil
				continue
			}

			if chunk.Model != "" {
				modelVersion = chunk.Model
			}
			if chunk.Usage != nil {
				usage = chunk.Usage
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			// The adapter supports a single candidate; a provider streaming
			// more than one choice per chunk would have the rest silently
			// discarded by picking Choices[0], so reject it loudly instead.
			if len(chunk.Choices) > 1 {
				finishStream(chunks, errs, cancel)
				yield(nil, unsupportedFeatureError("multiple choices in streamed completion chunk"))
				return
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

		case err, ok := <-errs:
			if !ok {
				errs = nil
				continue
			}
			streamErr = err
		}
	}

	if streamErr != nil {
		yield(nil, wrapProviderError(streamErr))
		return
	}

	content, err := contentFromStreamAggregation(textBuilder.String(), reasoningBuilder.String(), toolCallStates, includeThoughts)
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

// finishStream cancels the in-flight request and drains any remaining
// chunks/errs so the provider's sender goroutine does not leak. It reads
// both channels through a select loop instead of assuming close order, and
// tolerates either channel already being nil (already fully drained by the
// caller) without blocking on it.
func finishStream(
	chunks <-chan anyllm.ChatCompletionChunk,
	errs <-chan error,
	cancel context.CancelFunc,
) {
	cancel()

	for chunks != nil || errs != nil {
		select {
		case _, ok := <-chunks:
			if !ok {
				chunks = nil
			}
		case _, ok := <-errs:
			if !ok {
				errs = nil
			}
		}
	}
}

// aggregateStreamToolCall folds one streamed tool-call delta into states.
// AnyLLM's ToolCall has no index field (the OpenAI-compatible provider drops
// the wire "index" when building deltas), so fragments cannot be correlated
// positionally. Instead:
//
//   - A fragment carrying a non-empty ID is correlated by ID: if a state
//     with that ID already exists (providers that repeat ID+Name on every
//     fragment), its arguments are appended to that state rather than
//     starting a duplicate; otherwise a new state is started. A later
//     same-ID fragment that supplies a name/type the stored state is still
//     missing fills it in; one that disagrees with an already-known
//     name/type is rejected rather than silently overwritten.
//   - A pure continuation fragment (empty ID and empty name, the standard
//     OpenAI streaming shape for the single-tool-call case) can only be
//     attributed when exactly one state is in flight. With zero states it is
//     an orphan fragment; with more than one it is genuinely ambiguous —
//     both are reported as errors rather than silently appended to the
//     wrong (or a nonexistent) call.
//
// Limitation: streaming PARALLEL tool calls from a provider that omits a
// per-fragment id on continuation fragments is unsupported. any-llm-go's
// ToolCall drops the wire "index" the OpenAI protocol uses to route
// continuation fragments back to their call, so once two or more calls are
// open at once a bare continuation fragment cannot be attributed and this
// function returns an error instead of guessing. It is fine for the common
// case of one tool call at a time, or for providers that repeat the id on
// every fragment.
func aggregateStreamToolCall(states *[]streamToolCallState, delta anyllm.ToolCall) error {
	isHeader := delta.ID != "" || delta.Function.Name != ""

	if isHeader {
		if delta.ID != "" {
			if idx := slices.IndexFunc(*states, func(s streamToolCallState) bool {
				return s.id == delta.ID
			}); idx >= 0 {
				state := &(*states)[idx]
				if err := mergeStreamToolCallMetadata(state, delta); err != nil {
					return err
				}
				state.args.WriteString(delta.Function.Arguments)
				return nil
			}
		}

		*states = append(*states, streamToolCallState{
			id:       delta.ID,
			toolType: delta.Type,
			name:     delta.Function.Name,
			args:     new(strings.Builder),
		})
		(*states)[len(*states)-1].args.WriteString(delta.Function.Arguments)
		return nil
	}

	switch len(*states) {
	case 0:
		return newError("stream tool call arguments without preceding tool call header")
	case 1:
		(*states)[0].args.WriteString(delta.Function.Arguments)
		return nil
	default:
		return newError("stream tool call continuation fragment is ambiguous among multiple in-flight tool calls")
	}
}

// mergeStreamToolCallMetadata folds a same-ID fragment's name/type into an
// already-started state, filling in whatever the header fragment omitted.
// A fragment that disagrees with an already-known name or type is rejected
// rather than silently overwriting it, since that would hide a genuine
// provider inconsistency behind whichever fragment happened to arrive last.
func mergeStreamToolCallMetadata(state *streamToolCallState, delta anyllm.ToolCall) error {
	if delta.Function.Name != "" {
		switch {
		case state.name == "":
			state.name = delta.Function.Name
		case state.name != delta.Function.Name:
			return newErrorf("stream tool call %q name changed from %q to %q across fragments", state.id, state.name, delta.Function.Name)
		}
	}

	if delta.Type != "" {
		switch {
		case state.toolType == "":
			state.toolType = delta.Type
		case state.toolType != delta.Type:
			return newErrorf("stream tool call %q type changed from %q to %q across fragments", state.id, state.toolType, delta.Type)
		}
	}

	return nil
}

func contentFromStreamAggregation(
	text string,
	reasoning string,
	toolCallStates []streamToolCallState,
	includeThoughts bool,
) (*genai.Content, error) {
	parts := make([]*genai.Part, 0, 2+len(toolCallStates))

	if text != "" {
		parts = append(parts, genai.NewPartFromText(text))
	}
	// genai only returns thought summaries when IncludeThoughts was
	// explicitly requested; a provider's reasoning is otherwise suppressed
	// even when reasoning effort itself is enabled.
	if includeThoughts && reasoning != "" {
		parts = append(parts, &genai.Part{
			Text:    reasoning,
			Thought: true,
		})
	}

	// Pre-registering every provider-supplied id before synthesizing any
	// missing one guarantees a synthetic id cannot collide with an explicit
	// id used elsewhere in the same stream aggregation, regardless of which
	// tool call carries it. Two states with the same explicit id make
	// correlation with a later FunctionResponse ambiguous, so that must fail
	// loudly rather than being silently preserved.
	usedIDs := make(map[string]struct{}, len(toolCallStates))
	for _, state := range toolCallStates {
		if state.id == "" {
			continue
		}
		if _, exists := usedIDs[state.id]; exists {
			return nil, newErrorf("duplicate tool call id %q in stream aggregation", state.id)
		}
		usedIDs[state.id] = struct{}{}
	}

	for idx, state := range toolCallStates {
		id := state.id
		if id == "" {
			id = uniqueSyntheticToolCallID(usedIDs, state.name, idx+1)
		}

		part, err := functionCallPart(id, state.toolType, state.name, state.args.String())
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
