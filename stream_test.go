package adkanyllm

import (
	"context"
	"errors"
	"testing"
	"time"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestGenerateStreamEmitsPartialAndFinalResponses(t *testing.T) {
	t.Parallel()

	m := &Model{provider: &fakeProvider{}}
	req := &model.LLMRequest{
		Model: "gpt-test",
		Contents: []*genai.Content{
			genai.NewContentFromText("hello", genai.RoleUser),
		},
	}

	var yielded []*model.LLMResponse
	seq := m.GenerateContent(t.Context(), req, true)
	seq(func(resp *model.LLMResponse, err error) bool {
		if err != nil {
			t.Fatalf("unexpected stream error: %v", err)
		}
		yielded = append(yielded, resp)
		return true
	})

	if len(yielded) != 3 {
		t.Fatalf("expected 3 yielded responses (2 partial + 1 final), got %d", len(yielded))
	}
	if !yielded[0].Partial || yielded[0].Content.Parts[0].Text != "Hel" {
		t.Fatalf("unexpected first partial response: %#v", yielded[0])
	}
	if !yielded[1].Partial || yielded[1].Content.Parts[0].Text != "lo" {
		t.Fatalf("unexpected second partial response: %#v", yielded[1])
	}
	final := yielded[2]
	if final.Partial || !final.TurnComplete {
		t.Fatalf("expected final non-partial turn-complete response")
	}
	if final.Content.Parts[0].Text != "Hello" {
		t.Fatalf("expected final accumulated content Hello, got %#v", final.Content)
	}
}

func TestAggregateStreamToolCallFragments(t *testing.T) {
	t.Parallel()

	var states []streamToolCallState
	if err := aggregateStreamToolCall(&states, anyllm.ToolCall{
		ID:   "call_1",
		Type: toolTypeFunction,
		Function: anyllm.FunctionCall{
			Name:      "get_weather",
			Arguments: `{"city":`,
		},
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := aggregateStreamToolCall(&states, anyllm.ToolCall{
		Function: anyllm.FunctionCall{Arguments: `"Paris"}`},
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(states) != 1 {
		t.Fatalf("expected 1 aggregated tool call, got %d", len(states))
	}
	if states[0].args.String() != `{"city":"Paris"}` {
		t.Fatalf("unexpected args: %q", states[0].args.String())
	}
}

func TestGenerateStreamPropagatesTerminalError(t *testing.T) {
	t.Parallel()

	expectedErr := errors.New("boom")
	m := &Model{
		provider: &fakeProvider{
			streamFn: func(_ context.Context, _ anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
				chunks, errs := fakeStreamChannels()
				close(chunks)
				errs <- expectedErr
				close(errs)
				return chunks, errs
			},
		},
	}

	var gotErr error
	seq := m.GenerateContent(t.Context(), &model.LLMRequest{
		Model: "gpt-test",
		Contents: []*genai.Content{
			genai.NewContentFromText("hello", genai.RoleUser),
		},
	}, true)
	seq(func(_ *model.LLMResponse, err error) bool {
		if err != nil {
			gotErr = err
		}
		return true
	})

	if gotErr == nil {
		t.Fatal("expected terminal stream error")
	}
}

func TestAggregateStreamToolCallRejectsOrphanArguments(t *testing.T) {
	t.Parallel()

	var states []streamToolCallState
	err := aggregateStreamToolCall(&states, anyllm.ToolCall{
		Function: anyllm.FunctionCall{Arguments: `{"city":"Paris"}`},
	})
	if err == nil {
		t.Fatal("expected orphan arguments error")
	}
}

func TestGenerateStreamFinalIncludesToolCalls(t *testing.T) {
	t.Parallel()

	m := &Model{
		provider: &fakeProvider{
			streamFn: func(_ context.Context, _ anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
				chunks, errs := fakeStreamChannels()
				go func() {
					defer close(chunks)
					defer close(errs)
					chunks <- anyllm.ChatCompletionChunk{
						Choices: []anyllm.ChunkChoice{{
							Delta: anyllm.ChunkDelta{
								ToolCalls: []anyllm.ToolCall{{
									ID:   "call_1",
									Type: toolTypeFunction,
									Function: anyllm.FunctionCall{
										Name:      "get_weather",
										Arguments: `{"city":"Paris"}`,
									},
								}},
							},
							FinishReason: anyllm.FinishReasonToolCalls,
						}},
					}
				}()
				return chunks, errs
			},
		},
	}

	var final *model.LLMResponse
	seq := m.GenerateContent(t.Context(), &model.LLMRequest{
		Model:    "gpt-test",
		Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)},
	}, true)
	seq(func(resp *model.LLMResponse, err error) bool {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp != nil && !resp.Partial {
			final = resp
		}
		return true
	})
	if final == nil || final.Content == nil || final.Content.Parts[0].FunctionCall == nil {
		t.Fatalf("expected final tool call content, got %#v", final)
	}
}

func TestGenerateStreamRejectsInvalidToolArguments(t *testing.T) {
	t.Parallel()

	m := &Model{
		provider: &fakeProvider{
			streamFn: func(_ context.Context, _ anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
				chunks, errs := fakeStreamChannels()
				go func() {
					defer close(chunks)
					defer close(errs)
					chunks <- anyllm.ChatCompletionChunk{
						Choices: []anyllm.ChunkChoice{{
							Delta: anyllm.ChunkDelta{
								ToolCalls: []anyllm.ToolCall{{
									ID:   "call_1",
									Type: toolTypeFunction,
									Function: anyllm.FunctionCall{
										Name:      "get_weather",
										Arguments: `{"city":`,
									},
								}},
							},
							FinishReason: anyllm.FinishReasonToolCalls,
						}},
					}
				}()
				return chunks, errs
			},
		},
	}

	var gotErr error
	seq := m.GenerateContent(t.Context(), &model.LLMRequest{
		Model:    "gpt-test",
		Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)},
	}, true)
	seq(func(_ *model.LLMResponse, err error) bool {
		if err != nil {
			gotErr = err
		}
		return true
	})
	if gotErr == nil {
		t.Fatal("expected invalid tool arguments error")
	}
}

func TestGenerateStreamCancelsWhenConsumerStops(t *testing.T) {
	t.Parallel()

	cancelled := make(chan struct{}, 1)
	provider := &fakeProvider{
		streamFn: func(ctx context.Context, _ anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
			chunks, errs := fakeStreamChannels()

			go func() {
				defer close(chunks)
				defer close(errs)

				select {
				case chunks <- anyllm.ChatCompletionChunk{
					Choices: []anyllm.ChunkChoice{
						{Delta: anyllm.ChunkDelta{Content: "Hel"}},
					},
				}:
				case <-ctx.Done():
					errs <- ctx.Err()
					return
				}

				<-ctx.Done()
				cancelled <- struct{}{}
				errs <- ctx.Err()
			}()

			return chunks, errs
		},
	}

	m := &Model{provider: provider}
	seq := m.GenerateContent(t.Context(), &model.LLMRequest{
		Model: "gpt-test",
		Contents: []*genai.Content{
			genai.NewContentFromText("hello", genai.RoleUser),
		},
	}, true)
	seq(func(_ *model.LLMResponse, _ error) bool {
		return false
	})

	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("expected provider stream to observe cancellation")
	}
}
