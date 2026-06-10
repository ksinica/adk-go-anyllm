package adkanyllm

import (
	"context"
	"encoding/json"
	"math"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func (m *Model) generateOnce(
	ctx context.Context,
	params anyllm.CompletionParams,
	yield func(*model.LLMResponse, error) bool,
) {
	completion, err := m.provider.Completion(ctx, params)
	if err != nil {
		yield(nil, wrapProviderError(err))
		return
	}

	resp, err := responseFromCompletion(completion)
	if err != nil {
		yield(nil, err)
		return
	}

	yield(resp, nil)
}

func responseFromCompletion(completion *anyllm.ChatCompletion) (*model.LLMResponse, error) {
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
		return resp, nil
	}

	choice := completion.Choices[0]
	resp.FinishReason = mapFinishReason(choice.FinishReason)

	content, err := contentFromMessage(choice.Message)
	if err != nil {
		return nil, err
	}
	resp.Content = content
	resp.TurnComplete = true

	return resp, nil
}

func contentFromMessage(message anyllm.Message) (*genai.Content, error) {
	parts := make([]*genai.Part, 0, 1+len(message.ToolCalls))

	if text := message.ContentString(); text != "" {
		parts = append(parts, genai.NewPartFromText(text))
	}

	if message.Reasoning != nil && message.Reasoning.Content != "" {
		parts = append(parts, &genai.Part{
			Text:    message.Reasoning.Content,
			Thought: true,
		})
	}

	for _, tc := range message.ToolCalls {
		part, err := functionCallPart(tc.ID, tc.Type, tc.Function.Name, tc.Function.Arguments)
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
	if canUnwrap(err) {
		return err
	}

	return wrapError("provider completion", err)
}
