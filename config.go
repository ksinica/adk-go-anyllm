// Package adkanyllm adapts AnyLLM providers to Google ADK's model.LLM interface,
// converting between genai request/response types and any-llm-go completion parameters.
package adkanyllm

import (
	"maps"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

// Option configures a Model.
type Option func(*config) error

type config struct {
	provider anyllm.Provider
	extra    map[string]any
	model    string
}

// WithProvider sets the AnyLLM provider used for completions.
//
// This option is required.
func WithProvider(provider anyllm.Provider) Option {
	return func(c *config) error {
		if provider == nil {
			return newError("provider is required")
		}

		c.provider = provider
		return nil
	}
}

// WithModel sets the default model name used when LLMRequest.Model is empty.
func WithModel(model string) Option {
	return func(c *config) error {
		c.model = model
		return nil
	}
}

// WithExtra clones and merges provider-specific request fields into each completion.
func WithExtra(extra map[string]any) Option {
	return func(c *config) error {
		if extra == nil {
			c.extra = nil
			return nil
		}

		c.extra = maps.Clone(extra)
		return nil
	}
}

func (c *config) validate() error {
	if c.provider == nil {
		return newError("provider is required")
	}

	return nil
}
