package adkanyllm

import (
	"context"
	"errors"
	"math"
	"testing"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestParseArguments(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		raw         string
		expectedErr bool
	}{
		{name: "object json", raw: `{"ok":true}`},
		{name: "empty json string", raw: ""},
		{name: "invalid json", raw: `{"ok":`, expectedErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			args, err := parseArguments(tt.raw)
			if tt.expectedErr {
				if err == nil {
					t.Fatal("expected error but got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if args == nil {
				t.Fatal("expected args map, got nil")
			}
		})
	}
}

func TestMapFinishReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		raw      string
		expected genai.FinishReason
	}{
		{name: "stop", raw: anyllm.FinishReasonStop, expected: genai.FinishReasonStop},
		{name: "length", raw: anyllm.FinishReasonLength, expected: genai.FinishReasonMaxTokens},
		{name: "content filter", raw: anyllm.FinishReasonContentFilter, expected: genai.FinishReasonSafety},
		{name: "tool calls", raw: anyllm.FinishReasonToolCalls, expected: genai.FinishReasonOther},
		{name: "unknown", raw: "other", expected: genai.FinishReasonUnspecified},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := mapFinishReason(tt.raw)
			if got != tt.expected {
				t.Fatalf("mapFinishReason(%q)=%v expected=%v", tt.raw, got, tt.expected)
			}
		})
	}
}

func TestUsageMetadataFromUsageRejectsOverflow(t *testing.T) {
	t.Parallel()

	_, err := usageMetadataFromUsage(&anyllm.Usage{
		PromptTokens: int(math.MaxInt32) + 1,
	})
	if err == nil {
		t.Fatal("expected token overflow error")
	}
}

func TestResponseFromCompletionRejectsUsageOverflow(t *testing.T) {
	t.Parallel()

	_, err := responseFromCompletion(&anyllm.ChatCompletion{
		Usage: &anyllm.Usage{
			TotalTokens: int(math.MaxInt32) + 1,
		},
	})
	if err == nil {
		t.Fatal("expected token overflow error")
	}
}

func TestResponseFromCompletionToolCallTurnIsComplete(t *testing.T) {
	t.Parallel()

	completion := &anyllm.ChatCompletion{
		Model: "gpt-test",
		Choices: []anyllm.Choice{
			{
				FinishReason: anyllm.FinishReasonToolCalls,
				Message: anyllm.Message{
					ToolCalls: []anyllm.ToolCall{
						{
							ID:   "call_1",
							Type: toolTypeFunction,
							Function: anyllm.FunctionCall{
								Name:      "get_weather",
								Arguments: `{"city":"Paris"}`,
							},
						},
					},
				},
			},
		},
	}

	resp, err := responseFromCompletion(completion)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !resp.TurnComplete {
		t.Fatal("expected TurnComplete=true for final completion event")
	}
	if resp.Content == nil || len(resp.Content.Parts) == 0 {
		t.Fatal("expected content parts")
	}
	if resp.Content.Parts[0].FunctionCall == nil {
		t.Fatal("expected function call part")
	}
}

func TestResponseFromCompletionNilCompletion(t *testing.T) {
	t.Parallel()

	_, err := responseFromCompletion(nil)
	if err == nil {
		t.Fatal("expected error for nil completion")
	}
}

func TestResponseFromCompletionEmptyChoices(t *testing.T) {
	t.Parallel()

	resp, err := responseFromCompletion(&anyllm.ChatCompletion{Model: "gpt-test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ModelVersion != "gpt-test" {
		t.Fatalf("ModelVersion=%q expected gpt-test", resp.ModelVersion)
	}
	if resp.Content != nil {
		t.Fatal("expected nil content for empty choices")
	}
}

func TestContentFromMessageRejectsUnsupportedToolType(t *testing.T) {
	t.Parallel()

	_, err := contentFromMessage(anyllm.Message{
		ToolCalls: []anyllm.ToolCall{{
			Type: "custom",
			Function: anyllm.FunctionCall{
				Name:      "fn",
				Arguments: `{}`,
			},
		}},
	})
	if err == nil {
		t.Fatal("expected unsupported tool type error")
	}
}

func TestContentFromMessageRejectsMissingToolName(t *testing.T) {
	t.Parallel()

	_, err := contentFromMessage(anyllm.Message{
		ToolCalls: []anyllm.ToolCall{{
			Type:     toolTypeFunction,
			Function: anyllm.FunctionCall{Arguments: `{}`},
		}},
	})
	if err == nil {
		t.Fatal("expected missing tool name error")
	}
}

func TestWrapProviderErrorPreservesWrappedErrors(t *testing.T) {
	t.Parallel()

	cause := wrapError("upstream", errors.New("root"))
	got := wrapProviderError(cause)
	if got != cause {
		t.Fatalf("expected original wrapped error, got %v", got)
	}
}

func TestWrapProviderErrorWrapsPlainErrors(t *testing.T) {
	t.Parallel()

	cause := errors.New("plain")
	got := wrapProviderError(cause)
	if !errors.Is(got, cause) {
		t.Fatalf("expected wrapped cause, got %v", got)
	}
	var adapterErr *AdapterError
	if !errors.As(got, &adapterErr) {
		t.Fatal("expected *AdapterError")
	}
}

func TestResponseFromCompletionMapsReasoning(t *testing.T) {
	t.Parallel()

	completion := &anyllm.ChatCompletion{
		Choices: []anyllm.Choice{
			{
				Message: anyllm.Message{
					Content: "answer",
					Reasoning: &anyllm.Reasoning{
						Content: "thinking",
					},
				},
			},
		},
	}

	resp, err := responseFromCompletion(completion)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(resp.Content.Parts))
	}
	if !resp.Content.Parts[1].Thought || resp.Content.Parts[1].Text != "thinking" {
		t.Fatalf("unexpected reasoning part: %#v", resp.Content.Parts[1])
	}
}

func TestGenerateOnceRejectsNilCompletion(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	m := &Model{provider: &fakeProvider{
		completionFn: func(_ context.Context, _ anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
			return nil, nil
		},
	}}

	var gotErr error
	yield := func(_ *model.LLMResponse, err error) bool {
		gotErr = err
		return true
	}
	m.generateOnce(ctx, anyllm.CompletionParams{}, yield)
	if gotErr == nil {
		t.Fatal("expected nil completion error")
	}
}

func TestUsageMetadataFromUsageZeroIsNil(t *testing.T) {
	t.Parallel()

	got, err := usageMetadataFromUsage(&anyllm.Usage{})
	if err != nil || got != nil {
		t.Fatalf("expected nil, got %#v err=%v", got, err)
	}
}
