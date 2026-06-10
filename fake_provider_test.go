package adkanyllm

import (
	"context"
	"sync"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

type fakeProvider struct {
	completionFn  func(ctx context.Context, params anyllm.CompletionParams) (*anyllm.ChatCompletion, error)
	streamFn      func(ctx context.Context, params anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error)
	streamStarted chan struct{}
	name          string
	mu            sync.Mutex
}

func fakeStreamChannels() (chan anyllm.ChatCompletionChunk, chan error) {
	var (
		chunks = make(chan anyllm.ChatCompletionChunk)
		errs   = make(chan error, 1)
	)

	return chunks, errs
}

func (f *fakeProvider) Name() string {
	if f.name == "" {
		return "fake"
	}

	return f.name
}

func (f *fakeProvider) Completion(
	ctx context.Context,
	params anyllm.CompletionParams,
) (*anyllm.ChatCompletion, error) {
	if f.completionFn != nil {
		return f.completionFn(ctx, params)
	}

	return &anyllm.ChatCompletion{
		Model: params.Model,
		Choices: []anyllm.Choice{
			{
				Message: anyllm.Message{
					Role:    anyllm.RoleAssistant,
					Content: "ok",
				},
				FinishReason: anyllm.FinishReasonStop,
			},
		},
	}, nil
}

func (f *fakeProvider) CompletionStream(
	ctx context.Context,
	params anyllm.CompletionParams,
) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
	if f.streamFn != nil {
		return f.streamFn(ctx, params)
	}

	chunks, errs := fakeStreamChannels()

	go func() {
		defer close(chunks)
		defer close(errs)

		f.mu.Lock()
		if f.streamStarted != nil {
			close(f.streamStarted)
		}
		f.mu.Unlock()

		select {
		case chunks <- anyllm.ChatCompletionChunk{
			Model: params.Model,
			Choices: []anyllm.ChunkChoice{
				{
					Delta: anyllm.ChunkDelta{Content: "Hel"},
				},
			},
		}:
		case <-ctx.Done():
			errs <- ctx.Err()
			return
		}

		select {
		case chunks <- anyllm.ChatCompletionChunk{
			Model: params.Model,
			Choices: []anyllm.ChunkChoice{
				{
					Delta:        anyllm.ChunkDelta{Content: "lo"},
					FinishReason: anyllm.FinishReasonStop,
				},
			},
		}:
		case <-ctx.Done():
			errs <- ctx.Err()
			return
		}
	}()

	return chunks, errs
}
