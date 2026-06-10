package adkanyllm

import (
	"context"
	"iter"
	"maps"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/adk/model"
)

var _ model.LLM = (*Model)(nil)

// Model implements ADK's model.LLM interface using an AnyLLM provider.
type Model struct {
	provider     anyllm.Provider
	extra        map[string]any
	defaultModel string
}

// New constructs a new ADK-compatible AnyLLM model adapter.
func New(opts ...Option) (*Model, error) {
	cfg := &config{}

	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt(cfg); err != nil {
			return nil, err
		}
	}

	if err := cfg.validate(); err != nil {
		return nil, err
	}

	return &Model{
		provider:     cfg.provider,
		defaultModel: cfg.model,
		extra:        maps.Clone(cfg.extra),
	}, nil
}

// Name returns the configured default model.
func (m *Model) Name() string {
	return m.defaultModel
}

// GenerateContent converts ADK requests to AnyLLM completions.
func (m *Model) GenerateContent(
	ctx context.Context,
	req *model.LLMRequest,
	stream bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m == nil || m.provider == nil {
			yield(nil, newError("model is not configured"))
			return
		}
		if req == nil {
			yield(nil, newError("nil request"))
			return
		}

		params, err := buildCompletionParams(req, m.defaultModel, m.extra)
		if err != nil {
			yield(nil, err)
			return
		}

		if stream {
			m.generateStream(ctx, params, yield)
			return
		}

		m.generateOnce(ctx, params, yield)
	}
}
