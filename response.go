package adkanyllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func (m *Model) generateOnce(
	ctx context.Context,
	params anyllm.CompletionParams,
	includeThoughts bool,
	yield func(*model.LLMResponse, error) bool,
) {
	completion, err := m.provider.Completion(ctx, params)
	if err != nil {
		yield(nil, wrapProviderError(err))
		return
	}

	resp, err := responseFromCompletion(completion, includeThoughts)
	if err != nil {
		yield(nil, err)
		return
	}

	yield(resp, nil)
}

func responseFromCompletion(completion *anyllm.ChatCompletion, includeThoughts bool) (*model.LLMResponse, error) {
	if completion == nil {
		return nil, newError("nil completion response")
	}

	usageMetadata, err := usageMetadataFromUsage(completion.Usage)
	if err != nil {
		return nil, err
	}

	resp := &model.LLMResponse{
		UsageMetadata: usageMetadata,
		ModelVersion:  completion.Model,
	}

	if len(completion.Choices) == 0 {
		resp.TurnComplete = true
		return resp, nil
	}

	// The adapter supports a single candidate; a provider returning more
	// than one choice would have the rest silently discarded by picking
	// Choices[0], so reject it loudly instead.
	if len(completion.Choices) > 1 {
		return nil, unsupportedFeatureError("multiple choices in completion response")
	}

	choice := completion.Choices[0]
	resp.FinishReason = mapFinishReason(choice.FinishReason)

	content, err := contentFromMessage(choice.Message, includeThoughts)
	if err != nil {
		return nil, err
	}
	resp.Content = content
	resp.TurnComplete = true

	return resp, nil
}

func contentFromMessage(message anyllm.Message, includeThoughts bool) (*genai.Content, error) {
	parts := make([]*genai.Part, 0, 2+len(message.ToolCalls))

	text, err := textFromMessageContent(message)
	if err != nil {
		return nil, err
	}
	if text != "" {
		parts = append(parts, genai.NewPartFromText(text))
	}

	// genai only returns thought summaries when IncludeThoughts was
	// explicitly requested; a provider's reasoning is otherwise suppressed
	// even when reasoning effort itself is enabled.
	if includeThoughts && message.Reasoning != nil && message.Reasoning.Content != "" {
		parts = append(parts, &genai.Part{
			Text:    message.Reasoning.Content,
			Thought: true,
		})
	}

	// Pre-registering every provider-supplied id before synthesizing any
	// missing one guarantees a synthetic id cannot collide with an explicit
	// id used elsewhere in the same response, regardless of which tool call
	// carries it. A provider returning two tool calls with the same explicit
	// id makes correlation with a later FunctionResponse ambiguous, so that
	// must fail loudly rather than being silently preserved.
	usedIDs := make(map[string]struct{}, len(message.ToolCalls))
	for _, tc := range message.ToolCalls {
		if tc.ID == "" {
			continue
		}
		if _, exists := usedIDs[tc.ID]; exists {
			return nil, newErrorf("duplicate tool call id %q in response", tc.ID)
		}
		usedIDs[tc.ID] = struct{}{}
	}

	for idx, tc := range message.ToolCalls {
		id := tc.ID
		if id == "" {
			id = uniqueSyntheticToolCallID(usedIDs, tc.Function.Name, idx+1)
		}

		part, err := functionCallPart(id, tc.Type, tc.Function.Name, tc.Function.Arguments)
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

// textFromMessageContent extracts text from an assistant message's Content
// field. Providers normally return a plain string, but the OpenAI-compatible
// wire format also allows a content-parts array (e.g. a provider echoing
// back multimodal content). Text parts are concatenated; any other part
// type, or any other Content type entirely, is rejected loudly rather than
// silently dropped.
func textFromMessageContent(message anyllm.Message) (string, error) {
	if message.Content == nil {
		return "", nil
	}
	if text, ok := message.Content.(string); ok {
		return text, nil
	}

	contentParts := message.ContentParts()
	if contentParts == nil {
		return "", unsupportedFeatureErrorf("assistant message content type %T", message.Content)
	}

	var textBuilder strings.Builder
	for _, part := range contentParts {
		if part.Type != contentTypeText {
			return "", unsupportedFeatureErrorf("assistant message content part type %q", part.Type)
		}
		if textBuilder.Len() > 0 {
			textBuilder.WriteByte('\n')
		}
		textBuilder.WriteString(part.Text)
	}

	return textBuilder.String(), nil
}

// syntheticToolCallID synthesizes a deterministic id for a tool call whose
// provider returned an empty one. ADK correlates a FunctionResponse back to
// its FunctionCall by id, so a genai.FunctionCall with an empty ID cannot be
// round-tripped through an agent loop; ordinal is the call's 1-based
// position among the tool calls in the same response, which keeps ids
// unique within a response even when multiple calls target the same
// function name.
func syntheticToolCallID(name string, ordinal int) string {
	return fmt.Sprintf("call_%s_%d", name, ordinal)
}

// uniqueSyntheticToolCallID returns a synthesized id for an ID-less tool
// call that does not collide with any id already used in the same response
// or stream aggregation. used must already contain every non-empty
// provider-supplied id in the response before this is called for any tool
// call in it, so that an explicit id on a later tool call (e.g.
// "call_lookup_2" on the second of two ID-less "lookup" calls) cannot
// collide with the synthetic id generated for an earlier one. startOrdinal
// is the call's 1-based position among the tool calls in the response or
// stream, used as the starting point for the search.
func uniqueSyntheticToolCallID(used map[string]struct{}, name string, startOrdinal int) string {
	for ordinal := startOrdinal; ; ordinal++ {
		candidate := syntheticToolCallID(name, ordinal)
		if _, exists := used[candidate]; exists {
			continue
		}

		used[candidate] = struct{}{}
		return candidate
	}
}

func functionCallPart(id, toolType, name, arguments string) (*genai.Part, error) {
	if toolType != "" && toolType != toolTypeFunction {
		return nil, newErrorf("unsupported tool call type %q", toolType)
	}
	if name == "" {
		return nil, newError("tool call name is required")
	}

	args, err := parseArguments(arguments)
	if err != nil {
		return nil, err
	}

	return &genai.Part{
		FunctionCall: &genai.FunctionCall{
			ID:   id,
			Name: name,
			Args: args,
		},
	}, nil
}

func parseArguments(data string) (map[string]any, error) {
	if data == "" {
		return map[string]any{}, nil
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(data), &args); err != nil {
		return nil, wrapError("invalid function arguments json", err)
	}

	if args == nil {
		// json.Unmarshal leaves args nil for JSON "null" without an error;
		// normalize it to a non-nil empty map so a downstream tool writing
		// to the map does not panic. Any other non-object JSON (arrays,
		// numbers, strings, booleans) already fails the Unmarshal above.
		args = map[string]any{}
	}

	return args, nil
}

func usageMetadataFromUsage(usage *anyllm.Usage) (*genai.GenerateContentResponseUsageMetadata, error) {
	if usage == nil {
		return nil, nil
	}

	// Extract the 3-operand condition into a named boolean (code-style rule).
	allZero := usage.TotalTokens == 0 && usage.PromptTokens == 0 && usage.CompletionTokens == 0
	if allZero {
		return nil, nil
	}

	promptTokens, err := intToInt32(usage.PromptTokens)
	if err != nil {
		return nil, err
	}

	completionTokens, err := intToInt32(usage.CompletionTokens)
	if err != nil {
		return nil, err
	}

	totalTokens, err := intToInt32(usage.TotalTokens)
	if err != nil {
		return nil, err
	}

	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     promptTokens,
		CandidatesTokenCount: completionTokens,
		TotalTokenCount:      totalTokens,
	}, nil
}

func intToInt32(value int) (int32, error) {
	if value < 0 || value > math.MaxInt32 {
		return 0, newErrorf("token count %d overflows int32", value)
	}

	return int32(value), nil
}

func mapFinishReason(reason string) genai.FinishReason {
	switch reason {
	case anyllm.FinishReasonStop:
		return genai.FinishReasonStop
	case anyllm.FinishReasonLength:
		return genai.FinishReasonMaxTokens
	case anyllm.FinishReasonContentFilter:
		return genai.FinishReasonSafety
	case anyllm.FinishReasonToolCalls:
		return genai.FinishReasonOther
	default:
		return genai.FinishReasonUnspecified
	}
}

func wrapProviderError(err error) error {
	if err == nil {
		return nil
	}

	var adapterErr *AdapterError
	if errors.As(err, &adapterErr) {
		return err
	}

	return wrapError("provider completion", err)
}
