package adkanyllm

import (
	"context"
	"errors"
	"testing"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/adk/model"
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
