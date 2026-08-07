// Package adkanyllm adapts AnyLLM providers to Google ADK's model.LLM interface,
// converting between genai request/response types and any-llm-go completion parameters.
package adkanyllm

import (
	"maps"
)

// Option configures a Model.
type Option func(*config) error

type config struct {
	extra map[string]any
	model string
}

// WithModel sets the default model name used when LLMRequest.Model is empty.
func WithModel(model string) Option {
	return func(c *config) error {
		c.model = model
		return nil
	}
}

// WithExtra clones extra and sets it as the provider-specific request fields
// included in each completion, replacing any value set by a previous WithExtra call.
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
