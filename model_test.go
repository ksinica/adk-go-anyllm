package adkanyllm

import (
	"context"
	"errors"
	"testing"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

func TestModelGenerateContentOnce(t *testing.T) {
	t.Parallel()

	m, err := New(
		&fakeProvider{},
		WithModel("gpt-test"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var resp *model.LLMResponse
	seq := m.GenerateContent(t.Context(), &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("hello", genai.RoleUser),
		},
	}, false)
	seq(func(r *model.LLMResponse, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		resp = r
		return true
	})

	if resp == nil || resp.Content == nil {
		t.Fatal("expected response content")
	}
	if resp.Content.Parts[0].Text != "ok" {
		t.Fatalf("content=%q expected ok", resp.Content.Parts[0].Text)
	}
}

func TestGenerateContentNilRequest(t *testing.T) {
	t.Parallel()

	m, err := New(&fakeProvider{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var gotErr error
	seq := m.GenerateContent(t.Context(), nil, false)
	seq(func(_ *model.LLMResponse, err error) bool {
		gotErr = err
		return true
	})
	if gotErr == nil {
		t.Fatal("expected nil request error")
	}
	var adapterErr *AdapterError
	if !errors.As(gotErr, &adapterErr) {
		t.Fatalf("expected *AdapterError, got %T", gotErr)
	}
}

func TestGenerateContentUnconfiguredModel(t *testing.T) {
	t.Parallel()

	var gotErr error
	seq := (&Model{}).GenerateContent(t.Context(), &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
	}, false)
	seq(func(_ *model.LLMResponse, err error) bool {
		gotErr = err
		return true
	})
	if gotErr == nil {
		t.Fatal("expected unconfigured model error")
	}
}

// TestNewRejectsTypedNilProvider verifies that a typed-nil provider (e.g. a
// nil *fakeProvider boxed into the anyllm.Provider interface) is rejected
// just like a literal nil interface. A plain "provider == nil" check would
// miss this: the interface value itself is non-nil, it just wraps a nil
// pointer.
func TestNewRejectsTypedNilProvider(t *testing.T) {
	t.Parallel()

	var nilProvider *fakeProvider

	_, err := New(nilProvider)
	if err == nil {
		t.Fatal("expected error for typed-nil provider")
	}
	var adapterErr *AdapterError
	if !errors.As(err, &adapterErr) {
		t.Fatalf("expected *AdapterError, got %T", err)
	}
}

// TestGenerateContentTypedNilProviderReturnsError mirrors
// TestGenerateContentNilProviderReturnsError for a Model constructed with a
// typed-nil provider rather than a literally nil interface.
func TestGenerateContentTypedNilProviderReturnsError(t *testing.T) {
	t.Parallel()

	var nilProvider *fakeProvider

	var gotErr error
	seq := (&Model{provider: nilProvider, defaultModel: "gpt-test"}).GenerateContent(t.Context(), &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
	}, false)
	seq(func(_ *model.LLMResponse, err error) bool {
		gotErr = err
		return true
	})
	if gotErr == nil {
		t.Fatal("expected typed-nil provider error")
	}
	var adapterErr *AdapterError
	if !errors.As(gotErr, &adapterErr) {
		t.Fatalf("expected *AdapterError, got %T", gotErr)
	}
}

func TestIsNilValue(t *testing.T) {
	t.Parallel()

	var (
		nilProvider *fakeProvider
		nilMap      map[string]any
		nilSlice    []string
	)

	tests := []struct {
		name string
		v    any
		want bool
	}{
		{name: "untyped nil", v: nil, want: true},
		{name: "typed nil pointer", v: nilProvider, want: true},
		{name: "non-nil pointer", v: &fakeProvider{}, want: false},
		{name: "nil map", v: nilMap, want: true},
		{name: "non-nil map", v: map[string]any{}, want: false},
		{name: "nil slice", v: nilSlice, want: true},
		{name: "non-nil slice", v: []string{}, want: false},
		{name: "plain int", v: 5, want: false},
		{name: "plain string", v: "s", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := isNilValue(tt.v); got != tt.want {
				t.Fatalf("isNilValue(%#v)=%v expected %v", tt.v, got, tt.want)
			}
		})
	}
}

func TestModelNameNilReceiver(t *testing.T) {
	t.Parallel()

	var m *Model
	if got := m.Name(); got != "" {
		t.Fatalf("Name()=%q expected empty string for nil receiver", got)
	}
}

// TestGenerateContentNilProviderReturnsError verifies that a zero-valued
// &Model{} (nil provider interface, e.g. constructed without New) returns a
// clean *AdapterError instead of panicking inside m.provider.Completion.
// defaultModel is set so buildCompletionParams succeeds and execution
// actually reaches the provider call.
func TestGenerateContentNilProviderReturnsError(t *testing.T) {
	t.Parallel()

	var gotErr error
	seq := (&Model{defaultModel: "gpt-test"}).GenerateContent(t.Context(), &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
	}, false)
	seq(func(_ *model.LLMResponse, err error) bool {
		gotErr = err
		return true
	})
	if gotErr == nil {
		t.Fatal("expected nil provider error")
	}
	var adapterErr *AdapterError
	if !errors.As(gotErr, &adapterErr) {
		t.Fatalf("expected *AdapterError, got %T", gotErr)
	}
}

func TestGenerateContentUnsupportedFeature(t *testing.T) {
	t.Parallel()

	m, err := New(
		&fakeProvider{},
		WithModel("gpt-test"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var gotErr error
	seq := m.GenerateContent(t.Context(), &model.LLMRequest{
		Model: "gpt-test",
		Config: &genai.GenerateContentConfig{
			TopK: ptrFloat32(4),
		},
		Contents: []*genai.Content{
			genai.NewContentFromText("hello", genai.RoleUser),
		},
	}, false)
	seq(func(_ *model.LLMResponse, err error) bool {
		gotErr = err
		return true
	})
	if !errors.Is(gotErr, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", gotErr)
	}
}

func TestGenerateContentUsesDefaultModel(t *testing.T) {
	t.Parallel()

	var capturedModel string
	m, err := New(
		&fakeProvider{
			completionFn: func(_ context.Context, params anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
				capturedModel = params.Model
				return &anyllm.ChatCompletion{
					Choices: []anyllm.Choice{{
						Message:      anyllm.Message{Content: "ok"},
						FinishReason: anyllm.FinishReasonStop,
					}},
				}, nil
			},
		},
		WithModel("default-model"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	seq := m.GenerateContent(t.Context(), &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)},
	}, false)
	seq(func(_ *model.LLMResponse, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return true
	})
	if capturedModel != "default-model" {
		t.Fatalf("model=%q expected default-model", capturedModel)
	}
}

func TestGenerateContentPropagatesProviderError(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("provider down")
	m, err := New(&fakeProvider{
		completionFn: func(_ context.Context, _ anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
			return nil, wantErr
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var gotErr error
	seq := m.GenerateContent(t.Context(), &model.LLMRequest{
		Model:    "gpt-test",
		Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)},
	}, false)
	seq(func(_ *model.LLMResponse, err error) bool {
		gotErr = err
		return true
	})
	if !errors.Is(gotErr, wantErr) {
		t.Fatalf("expected provider error, got %v", gotErr)
	}
}

// TestGenerateContentSuppressesReasoningWhenNotIncluded is an end-to-end
// check that GenerateContent (non-streaming) suppresses a provider's
// returned reasoning when the request's ThinkingConfig.IncludeThoughts is
// false, even though reasoning effort is otherwise enabled.
func TestGenerateContentSuppressesReasoningWhenNotIncluded(t *testing.T) {
	t.Parallel()

	m, err := New(&fakeProvider{
		completionFn: func(_ context.Context, _ anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
			return &anyllm.ChatCompletion{
				Choices: []anyllm.Choice{{
					Message: anyllm.Message{
						Content:   "answer",
						Reasoning: &anyllm.Reasoning{Content: "thinking"},
					},
					FinishReason: anyllm.FinishReasonStop,
				}},
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var resp *model.LLMResponse
	seq := m.GenerateContent(t.Context(), &model.LLMRequest{
		Model:    "gpt-test",
		Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: false},
		},
	}, false)
	seq(func(r *model.LLMResponse, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		resp = r
		return true
	})
	if resp == nil || resp.Content == nil {
		t.Fatal("expected response content")
	}
	for _, part := range resp.Content.Parts {
		if part.Thought {
			t.Fatalf("unexpected Thought part when IncludeThoughts=false: %#v", part)
		}
	}
}

// TestGenerateContentIncludesReasoningWhenRequested pins the counterpart of
// TestGenerateContentSuppressesReasoningWhenNotIncluded: IncludeThoughts=true
// is the correct precondition for a Thought part to be returned.
func TestGenerateContentIncludesReasoningWhenRequested(t *testing.T) {
	t.Parallel()

	m, err := New(&fakeProvider{
		completionFn: func(_ context.Context, _ anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
			return &anyllm.ChatCompletion{
				Choices: []anyllm.Choice{{
					Message: anyllm.Message{
						Content:   "answer",
						Reasoning: &anyllm.Reasoning{Content: "thinking"},
					},
					FinishReason: anyllm.FinishReasonStop,
				}},
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var resp *model.LLMResponse
	seq := m.GenerateContent(t.Context(), &model.LLMRequest{
		Model:    "gpt-test",
		Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: true},
		},
	}, false)
	seq(func(r *model.LLMResponse, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		resp = r
		return true
	})
	if resp == nil || resp.Content == nil {
		t.Fatal("expected response content")
	}
	found := false
	for _, part := range resp.Content.Parts {
		if part.Thought && part.Text == "thinking" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected a Thought part, got %#v", resp.Content.Parts)
	}
}

func TestGenerateContentPropagatesExtra(t *testing.T) {
	t.Parallel()

	var capturedExtra map[string]any
	m, err := New(
		&fakeProvider{
			completionFn: func(_ context.Context, params anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
				capturedExtra = params.Extra
				return &anyllm.ChatCompletion{
					Choices: []anyllm.Choice{{
						Message:      anyllm.Message{Content: "ok"},
						FinishReason: anyllm.FinishReasonStop,
					}},
				}, nil
			},
		},
		WithModel("gpt-test"),
		WithExtra(map[string]any{"foo": "bar"}),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	seq := m.GenerateContent(t.Context(), &model.LLMRequest{
		Model:    "gpt-test",
		Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)},
	}, false)
	seq(func(_ *model.LLMResponse, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return true
	})
	if capturedExtra["foo"] != "bar" {
		t.Fatalf("extra=%#v expected foo=bar", capturedExtra)
	}
}
