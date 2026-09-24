package adkanyllm

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/adk/v2/model"
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

// TestAggregateStreamToolCallFragments covers the common, supported wire
// shape: a single tool call streamed as a header fragment (carrying the id
// and function name) followed by a bare continuation fragment with an empty
// id — exactly what real OpenAI-compatible providers send when only one
// tool call is in flight.
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

func TestAggregateStreamToolCallRepeatedIDPerFragment(t *testing.T) {
	t.Parallel()

	var states []streamToolCallState

	// Some providers repeat ID+Name on every fragment instead of sending a
	// header followed by bare continuations. Each repeat must correlate
	// back to the same state, not create a new bogus one.
	fragments := []anyllm.ToolCall{
		{ID: "call_1", Type: toolTypeFunction, Function: anyllm.FunctionCall{Name: "get_weather", Arguments: `{"city":`}},
		{ID: "call_1", Type: toolTypeFunction, Function: anyllm.FunctionCall{Name: "get_weather", Arguments: `"Paris",`}},
		{ID: "call_1", Type: toolTypeFunction, Function: anyllm.FunctionCall{Name: "get_weather", Arguments: `"unit":"C"}`}},
	}
	for _, frag := range fragments {
		if err := aggregateStreamToolCall(&states, frag); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	if len(states) != 1 {
		t.Fatalf("expected 1 aggregated tool call, got %d", len(states))
	}
	if got := states[0].args.String(); got != `{"city":"Paris","unit":"C"}` {
		t.Fatalf("unexpected args: %q", got)
	}
}

func TestAggregateStreamToolCallMergesLaterMetadataOnSameID(t *testing.T) {
	t.Parallel()

	var states []streamToolCallState

	// Header carries only the id; a later same-id fragment supplies the
	// name and type the header omitted. Both must be folded into the state.
	fragments := []anyllm.ToolCall{
		{ID: "call_1", Function: anyllm.FunctionCall{Arguments: `{"city":`}},
		{ID: "call_1", Type: toolTypeFunction, Function: anyllm.FunctionCall{Name: "get_weather", Arguments: `"Paris"}`}},
	}
	for _, frag := range fragments {
		if err := aggregateStreamToolCall(&states, frag); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	if len(states) != 1 {
		t.Fatalf("expected 1 aggregated tool call, got %d", len(states))
	}
	if states[0].name != "get_weather" || states[0].toolType != toolTypeFunction {
		t.Fatalf("expected merged name/type, got %#v", states[0])
	}
	if got := states[0].args.String(); got != `{"city":"Paris"}` {
		t.Fatalf("unexpected args: %q", got)
	}
}

func TestAggregateStreamToolCallRejectsConflictingNameOnSameID(t *testing.T) {
	t.Parallel()

	var states []streamToolCallState
	if err := aggregateStreamToolCall(&states, anyllm.ToolCall{
		ID:   "call_1",
		Type: toolTypeFunction,
		Function: anyllm.FunctionCall{
			Name: "get_weather",
		},
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err := aggregateStreamToolCall(&states, anyllm.ToolCall{
		ID: "call_1",
		Function: anyllm.FunctionCall{
			Name: "get_time",
		},
	})
	if err == nil {
		t.Fatal("expected conflicting name error")
	}
	var adapterErr *AdapterError
	if !errors.As(err, &adapterErr) {
		t.Fatalf("expected *AdapterError, got %T", err)
	}
}

func TestAggregateStreamToolCallRejectsConflictingTypeOnSameID(t *testing.T) {
	t.Parallel()

	var states []streamToolCallState
	if err := aggregateStreamToolCall(&states, anyllm.ToolCall{
		ID:   "call_1",
		Type: toolTypeFunction,
		Function: anyllm.FunctionCall{
			Name: "get_weather",
		},
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err := aggregateStreamToolCall(&states, anyllm.ToolCall{
		ID:   "call_1",
		Type: "custom",
	})
	if err == nil {
		t.Fatal("expected conflicting type error")
	}
}

func TestAggregateStreamToolCallRejectsAmbiguousInterleavedContinuation(t *testing.T) {
	t.Parallel()

	var states []streamToolCallState

	// Two parallel tool calls open concurrently (two headers, neither
	// resolved), then a pure continuation fragment arrives with nothing to
	// correlate it against — it cannot be safely attributed to either.
	headers := []anyllm.ToolCall{
		{ID: "call_1", Type: toolTypeFunction, Function: anyllm.FunctionCall{Name: "get_weather"}},
		{ID: "call_2", Type: toolTypeFunction, Function: anyllm.FunctionCall{Name: "get_time"}},
	}
	for _, frag := range headers {
		if err := aggregateStreamToolCall(&states, frag); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	err := aggregateStreamToolCall(&states, anyllm.ToolCall{
		Function: anyllm.FunctionCall{Arguments: `{"city":"Paris"}`},
	})
	if err == nil {
		t.Fatal("expected ambiguous continuation error")
	}
	var adapterErr *AdapterError
	if !errors.As(err, &adapterErr) {
		t.Fatalf("expected *AdapterError, got %T", err)
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

// TestGenerateStreamDoesNotDeadlockOnUnbufferedErrsBeforeChunksClose
// regression-tests a provider that uses an unbuffered errs channel and
// sends its terminal error before closing chunks. Draining chunks to
// completion before ever reading errs would deadlock here: the provider's
// send on errs blocks forever with nothing reading it, so chunks never
// gets closed either.
func TestGenerateStreamDoesNotDeadlockOnUnbufferedErrsBeforeChunksClose(t *testing.T) {
	t.Parallel()

	expectedErr := errors.New("early boom")
	m := &Model{
		provider: &fakeProvider{
			streamFn: func(_ context.Context, _ anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
				var (
					chunks = make(chan anyllm.ChatCompletionChunk)
					errs   = make(chan error) // unbuffered, on purpose
				)

				go func() {
					defer close(chunks)
					defer close(errs)
					errs <- expectedErr
				}()

				return chunks, errs
			},
		},
	}

	var (
		gotErr error
		done   = make(chan struct{})
	)
	go func() {
		defer close(done)

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
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("generateStream deadlocked draining an unbuffered errs channel before chunks closed")
	}

	if !errors.Is(gotErr, expectedErr) {
		t.Fatalf("expected wrapped terminal error, got %v", gotErr)
	}
}

func TestContentFromStreamAggregationSynthesizesEmptyToolCallID(t *testing.T) {
	t.Parallel()

	state := streamToolCallState{name: "get_weather", args: new(strings.Builder)}
	state.args.WriteString(`{"city":"Paris"}`)

	content, err := contentFromStreamAggregation("", "", []streamToolCallState{state}, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fc := content.Parts[0].FunctionCall
	if fc == nil || fc.ID == "" {
		t.Fatalf("expected synthesized non-empty id, got %#v", fc)
	}
}

func TestContentFromStreamAggregationSynthesizesUniqueIDsAcrossMultipleToolCalls(t *testing.T) {
	t.Parallel()

	newState := func(name string) streamToolCallState {
		s := streamToolCallState{name: name, args: new(strings.Builder)}
		s.args.WriteString(`{}`)
		return s
	}

	content, err := contentFromStreamAggregation("", "", []streamToolCallState{
		newState("get_weather"),
		newState("get_weather"),
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

func TestContentFromStreamAggregationPreservesNonEmptyToolCallID(t *testing.T) {
	t.Parallel()

	state := streamToolCallState{id: "call_1", name: "get_weather", args: new(strings.Builder)}
	state.args.WriteString(`{}`)

	content, err := contentFromStreamAggregation("", "", []streamToolCallState{state}, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := content.Parts[0].FunctionCall.ID; got != "call_1" {
		t.Fatalf("ID=%q expected call_1", got)
	}
}

// TestContentFromStreamAggregationAvoidsCollisionWithExplicitProviderID
// verifies the item-2 fix for the streaming path: a synthesized id for an
// ID-less tool call is checked against the explicit ids of other tool calls
// in the same aggregation, and advanced past a collision rather than
// reusing one.
func TestContentFromStreamAggregationAvoidsCollisionWithExplicitProviderID(t *testing.T) {
	t.Parallel()

	first := streamToolCallState{id: "call_lookup_2", name: "lookup", args: new(strings.Builder)}
	first.args.WriteString(`{}`)
	second := streamToolCallState{name: "lookup", args: new(strings.Builder)}
	second.args.WriteString(`{}`)

	content, err := contentFromStreamAggregation("", "", []streamToolCallState{first, second}, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	id1 := content.Parts[0].FunctionCall.ID
	id2 := content.Parts[1].FunctionCall.ID
	if id1 != "call_lookup_2" {
		t.Fatalf("id1=%q expected explicit call_lookup_2 preserved", id1)
	}
	if id2 != "call_lookup_3" {
		t.Fatalf("id2=%q expected advanced past collision to call_lookup_3", id2)
	}
}

// TestContentFromStreamAggregationRejectsDuplicateExplicitToolCallIDs
// verifies the item-3 fix for the streaming path: two aggregated tool-call
// states sharing the same explicit, non-empty id are rejected during
// pre-registration rather than silently preserved.
func TestContentFromStreamAggregationRejectsDuplicateExplicitToolCallIDs(t *testing.T) {
	t.Parallel()

	first := streamToolCallState{id: "call_1", name: "get_weather", args: new(strings.Builder)}
	first.args.WriteString(`{}`)
	second := streamToolCallState{id: "call_1", name: "get_time", args: new(strings.Builder)}
	second.args.WriteString(`{}`)

	_, err := contentFromStreamAggregation("", "", []streamToolCallState{first, second}, false)
	var adapterErr *AdapterError
	if !errors.As(err, &adapterErr) {
		t.Fatalf("expected *AdapterError for duplicate explicit tool call id, got %v", err)
	}
}

// TestContentFromStreamAggregationSuppressesReasoningWhenNotIncluded
// verifies the item-2 fix for the streaming path: with includeThoughts=false
// a provider's aggregated reasoning must not surface as a Thought part.
func TestContentFromStreamAggregationSuppressesReasoningWhenNotIncluded(t *testing.T) {
	t.Parallel()

	content, err := contentFromStreamAggregation("answer", "thinking", nil, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(content.Parts) != 1 {
		t.Fatalf("expected reasoning to be suppressed, got %d parts: %#v", len(content.Parts), content.Parts)
	}
	for _, part := range content.Parts {
		if part.Thought {
			t.Fatalf("unexpected Thought part when includeThoughts=false: %#v", part)
		}
	}
}

// TestContentFromStreamAggregationIncludesReasoningWhenRequested pins the
// counterpart: includeThoughts=true still returns the aggregated reasoning
// as a Thought part.
func TestContentFromStreamAggregationIncludesReasoningWhenRequested(t *testing.T) {
	t.Parallel()

	content, err := contentFromStreamAggregation("answer", "thinking", nil, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(content.Parts) != 2 || !content.Parts[1].Thought || content.Parts[1].Text != "thinking" {
		t.Fatalf("unexpected parts: %#v", content.Parts)
	}
}

// TestGenerateStreamSuppressesReasoningWhenNotIncluded is an end-to-end
// check that GenerateContent(stream=true) suppresses a provider's streamed
// reasoning delta when the request's ThinkingConfig.IncludeThoughts is
// false, even though reasoning effort is otherwise enabled.
func TestGenerateStreamSuppressesReasoningWhenNotIncluded(t *testing.T) {
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
								Content:   "Hello",
								Reasoning: &anyllm.Reasoning{Content: "thinking"},
							},
							FinishReason: anyllm.FinishReasonStop,
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
		Config: &genai.GenerateContentConfig{
			ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: false},
		},
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
	if final == nil || final.Content == nil {
		t.Fatal("expected final content")
	}
	for _, part := range final.Content.Parts {
		if part.Thought {
			t.Fatalf("unexpected Thought part when IncludeThoughts=false: %#v", part)
		}
	}
}

// TestGenerateStreamRejectsMultipleChoicesInChunk verifies that a provider
// streaming more than one choice per chunk fails loudly instead of silently
// discarding every choice past Choices[0].
func TestGenerateStreamRejectsMultipleChoicesInChunk(t *testing.T) {
	t.Parallel()

	m := &Model{
		provider: &fakeProvider{
			streamFn: func(_ context.Context, _ anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
				chunks, errs := fakeStreamChannels()
				go func() {
					defer close(chunks)
					defer close(errs)
					chunks <- anyllm.ChatCompletionChunk{
						Choices: []anyllm.ChunkChoice{
							{Delta: anyllm.ChunkDelta{Content: "first"}},
							{Delta: anyllm.ChunkDelta{Content: "second"}},
						},
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
	if !errors.Is(gotErr, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", gotErr)
	}
}
