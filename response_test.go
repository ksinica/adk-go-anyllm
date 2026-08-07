package adkanyllm

import (
	"context"
	"errors"
	"fmt"
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
		{name: "json null", raw: `null`},
		{name: "json array", raw: `[1,2]`, expectedErr: true},
		{name: "json string", raw: `"hi"`, expectedErr: true},
		{name: "json number", raw: `5`, expectedErr: true},
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
	}, false)
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

	resp, err := responseFromCompletion(completion, false)
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

	_, err := responseFromCompletion(nil, false)
	if err == nil {
		t.Fatal("expected error for nil completion")
	}
}

func TestResponseFromCompletionEmptyChoices(t *testing.T) {
	t.Parallel()

	resp, err := responseFromCompletion(&anyllm.ChatCompletion{Model: "gpt-test"}, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ModelVersion != "gpt-test" {
		t.Fatalf("ModelVersion=%q expected gpt-test", resp.ModelVersion)
	}
	if resp.Content != nil {
		t.Fatal("expected nil content for empty choices")
	}
	if !resp.TurnComplete {
		t.Fatal("expected TurnComplete=true even with zero choices")
	}
}

// TestResponseFromCompletionRejectsMultipleChoices verifies that a provider
// returning more than one choice fails loudly instead of silently
// discarding every choice past Choices[0].
func TestResponseFromCompletionRejectsMultipleChoices(t *testing.T) {
	t.Parallel()

	_, err := responseFromCompletion(&anyllm.ChatCompletion{
		Choices: []anyllm.Choice{
			{Message: anyllm.Message{Content: "first"}},
			{Message: anyllm.Message{Content: "second"}},
		},
	}, false)
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
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
	}, false)
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
	}, false)
	if err == nil {
		t.Fatal("expected missing tool name error")
	}
}

func TestWrapProviderErrorPreservesAdapterErrors(t *testing.T) {
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

// TestWrapProviderErrorWrapsUnwrappableNonAdapterErrors verifies that an
// error implementing Unwrap (e.g. via fmt.Errorf's %w) but that is NOT an
// *AdapterError still gets adapter context added. Using "does it implement
// Unwrap" as a proxy for "already has adapter context" let plain provider
// errors escape wrapping while errors.New errors got wrapped -
// inconsistent, and callers could not reliably errors.As to *AdapterError.
func TestWrapProviderErrorWrapsUnwrappableNonAdapterErrors(t *testing.T) {
	t.Parallel()

	root := errors.New("root cause")
	cause := fmt.Errorf("db timeout: %w", root)

	got := wrapProviderError(cause)

	var adapterErr *AdapterError
	if !errors.As(got, &adapterErr) {
		t.Fatalf("expected *AdapterError, got %T (%v)", got, got)
	}
	if !errors.Is(got, root) {
		t.Fatal("expected wrapped cause to remain reachable via errors.Is")
	}
}

// TestResponseFromCompletionMapsReasoning verifies that reasoning is
// returned as a Thought part when the request set IncludeThoughts=true; that
// is the correct precondition genai defines for returning thought summaries.
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

	resp, err := responseFromCompletion(completion, true)
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

// TestResponseFromCompletionSuppressesReasoningWhenNotIncluded verifies the
// item-2 fix: with IncludeThoughts=false (the default), a provider's
// reasoning must not surface as a Thought part, even though reasoning effort
// itself is enabled server-side and the provider returned reasoning content.
func TestResponseFromCompletionSuppressesReasoningWhenNotIncluded(t *testing.T) {
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

	resp, err := responseFromCompletion(completion, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Content.Parts) != 1 {
		t.Fatalf("expected reasoning to be suppressed, got %d parts: %#v", len(resp.Content.Parts), resp.Content.Parts)
	}
	for _, part := range resp.Content.Parts {
		if part.Thought {
			t.Fatalf("unexpected Thought part when IncludeThoughts=false: %#v", part)
		}
	}
}

func TestContentFromMessageConvertsTextContentParts(t *testing.T) {
	t.Parallel()

	resp, err := contentFromMessage(anyllm.Message{
		Content: []anyllm.ContentPart{
			{Type: contentTypeText, Text: "hello"},
			{Type: contentTypeText, Text: "world"},
		},
	}, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Parts) != 1 || resp.Parts[0].Text != "hello\nworld" {
		t.Fatalf("unexpected parts: %#v", resp.Parts)
	}
}

func TestContentFromMessageRejectsUnsupportedContentPartType(t *testing.T) {
	t.Parallel()

	_, err := contentFromMessage(anyllm.Message{
		Content: []anyllm.ContentPart{
			{Type: contentTypeImageURL, ImageURL: &anyllm.ImageURL{URL: "https://example.com/x.png"}},
		},
	}, false)
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestContentFromMessageRejectsUnsupportedContentType(t *testing.T) {
	t.Parallel()

	_, err := contentFromMessage(anyllm.Message{Content: 5}, false)
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
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
	m.generateOnce(ctx, anyllm.CompletionParams{}, false, yield)
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

func TestContentFromMessageSynthesizesEmptyToolCallID(t *testing.T) {
	t.Parallel()

	content, err := contentFromMessage(anyllm.Message{
		ToolCalls: []anyllm.ToolCall{{
			Type:     toolTypeFunction,
			Function: anyllm.FunctionCall{Name: "get_weather", Arguments: `{"city":"Paris"}`},
		}},
	}, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fc := content.Parts[0].FunctionCall
	if fc == nil || fc.ID == "" {
		t.Fatalf("expected synthesized non-empty id, got %#v", fc)
	}
}

func TestContentFromMessageSynthesizesUniqueIDsAcrossMultipleToolCalls(t *testing.T) {
	t.Parallel()

	content, err := contentFromMessage(anyllm.Message{
		ToolCalls: []anyllm.ToolCall{
			{Type: toolTypeFunction, Function: anyllm.FunctionCall{Name: "get_weather", Arguments: `{}`}},
			{Type: toolTypeFunction, Function: anyllm.FunctionCall{Name: "get_weather", Arguments: `{}`}},
		},
	}, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	id1 := content.Parts[0].FunctionCall.ID
	id2 := content.Parts[1].FunctionCall.ID
	if id1 == "" || id2 == "" || id1 == id2 {
		t.Fatalf("expected distinct synthesized ids, got %q and %q", id1, id2)
	}
}

// TestContentFromMessageAvoidsCollisionWithExplicitProviderID verifies the
// item-2 fix: a synthesized id for an ID-less tool call is checked against
// the explicit ids of other tool calls in the same response, and advanced
// past a collision rather than reusing one.
func TestContentFromMessageAvoidsCollisionWithExplicitProviderID(t *testing.T) {
	t.Parallel()

	content, err := contentFromMessage(anyllm.Message{
		ToolCalls: []anyllm.ToolCall{
			{ID: "call_lookup_2", Type: toolTypeFunction, Function: anyllm.FunctionCall{Name: "lookup", Arguments: `{}`}},
			{Type: toolTypeFunction, Function: anyllm.FunctionCall{Name: "lookup", Arguments: `{}`}},
		},
	}, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	id1 := content.Parts[0].FunctionCall.ID
	id2 := content.Parts[1].FunctionCall.ID
	if id1 != "call_lookup_2" {
		t.Fatalf("id1=%q expected explicit call_lookup_2 preserved", id1)
	}
	if id2 == "" || id2 == id1 {
		t.Fatalf("expected synthesized id2 distinct from explicit id1, got %q", id2)
	}
	if id2 != "call_lookup_3" {
		t.Fatalf("id2=%q expected advanced past collision to call_lookup_3", id2)
	}
}

// TestContentFromMessageRejectsDuplicateExplicitToolCallIDs verifies the
// item-3 fix: two provider tool calls sharing the same explicit,
// non-empty id are rejected during pre-registration rather than silently
// preserved, since that would make later FunctionResponse correlation
// ambiguous.
func TestContentFromMessageRejectsDuplicateExplicitToolCallIDs(t *testing.T) {
	t.Parallel()

	_, err := contentFromMessage(anyllm.Message{
		ToolCalls: []anyllm.ToolCall{
			{ID: "call_1", Type: toolTypeFunction, Function: anyllm.FunctionCall{Name: "get_weather", Arguments: `{}`}},
			{ID: "call_1", Type: toolTypeFunction, Function: anyllm.FunctionCall{Name: "get_time", Arguments: `{}`}},
		},
	}, false)
	var adapterErr *AdapterError
	if !errors.As(err, &adapterErr) {
		t.Fatalf("expected *AdapterError for duplicate explicit tool call id, got %v", err)
	}
}

func TestContentFromMessagePreservesNonEmptyToolCallID(t *testing.T) {
	t.Parallel()

	content, err := contentFromMessage(anyllm.Message{
		ToolCalls: []anyllm.ToolCall{{
			ID:       "call_1",
			Type:     toolTypeFunction,
			Function: anyllm.FunctionCall{Name: "get_weather", Arguments: `{}`},
		}},
	}, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := content.Parts[0].FunctionCall.ID; got != "call_1" {
		t.Fatalf("ID=%q expected call_1", got)
	}
}
