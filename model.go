package adkanyllm

import (
	"context"
	"iter"
	"reflect"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/adk/model"
)

// isNilValue reports whether v is nil, either because the interface itself
// is nil or because it wraps a nil pointer, map, slice, channel, or func. A
// plain "v == nil" check misses the second case: a concrete typed nil (e.g.
// (*T)(nil)) boxed into an interface value is itself non-nil, so it looks
// non-nil to a caller comparing only against the untyped nil literal.
func isNilValue(v any) bool {
	if v == nil {
		return true
	}

	rv := reflect.ValueOf(v)
	if !rv.IsValid() {
		return true
	}

	switch rv.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func:
		return rv.IsNil()
	default:
		return false
	}
}

var _ model.LLM = (*Model)(nil)

// Model implements ADK's model.LLM interface using an AnyLLM provider.
type Model struct {
	provider     anyllm.Provider
	extra        map[string]any
	defaultModel string
}

// New constructs a new ADK-compatible AnyLLM model adapter.
// provider is required; model name and extra fields are configured via WithModel and WithExtra.
func New(provider anyllm.Provider, opts ...Option) (*Model, error) {
	if isNilValue(provider) {
		return nil, newError("provider is required")
	}

	cfg := &config{}

	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt(cfg); err != nil {
			return nil, err
		}
	}

	return &Model{
		provider:     provider,
		defaultModel: cfg.model,
		// cfg is local to this call, and WithExtra already clones its
		// argument into cfg.extra (or leaves it nil), so no further clone
		// is needed here.
		extra: cfg.extra,
	}, nil
}

// Name returns the configured default model.
func (m *Model) Name() string {
	if m == nil {
		return ""
	}

	return m.defaultModel
}

// GenerateContent converts ADK requests to AnyLLM completions.
func (m *Model) GenerateContent(
	ctx context.Context,
	req *model.LLMRequest,
	stream bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m == nil {
			yield(nil, newError("model is not configured"))
			return
		}
		if req == nil {
			yield(nil, newError("nil request"))
			return
		}
		// New rejects a nil provider, but a zero-valued &Model{} (e.g.
		// constructed directly rather than via New) has a nil provider
		// interface; guard it here rather than panicking inside
		// m.provider.Completion. isNilValue also catches a non-nil interface
		// wrapping a typed nil pointer, which a plain "== nil" check would miss.
		if isNilValue(m.provider) {
			yield(nil, newError("model is not configured: nil provider"))
			return
		}

		params, err := buildCompletionParams(req, m.defaultModel, m.extra)
		if err != nil {
			yield(nil, err)
			return
		}

		includeThoughts := includeThoughtsFromConfig(req.Config)

		if stream {
			m.generateStream(ctx, params, includeThoughts, yield)
			return
		}

		m.generateOnce(ctx, params, includeThoughts, yield)
	}
}
