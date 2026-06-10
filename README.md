# adk-go-anyllm

[![Go Version](https://img.shields.io/github/go-mod/go-version/ksinica/adk-go-anyllm)](https://go.dev/) [![License](https://img.shields.io/github/license/ksinica/adk-go-anyllm)](./LICENSE) [![Build Status](https://img.shields.io/github/actions/workflow/status/ksinica/adk-go-anyllm/ci.yml?branch=master)](https://github.com/ksinica/adk-go-anyllm/actions) [![Go Reference](https://pkg.go.dev/badge/github.com/ksinica/adk-go-anyllm.svg)](https://pkg.go.dev/github.com/ksinica/adk-go-anyllm)

**One adapter. Every model. Zero lock-in.**

Plug any [AnyLLM Go](https://github.com/mozilla-ai/any-llm-go) provider — OpenAI, Anthropic, Ollama, and friends — straight into [Google's Agent Development Kit for Go](https://google.golang.org/adk). `adk-go-anyllm` is a thin, faithful translation layer that implements `google.golang.org/adk/model.LLM` so your agents stop caring which vendor is behind the curtain.

Write your agent once. Swap the model whenever you like.

## Why you'll like it

- **Provider-agnostic by design.** Anything that satisfies `anyllm.Provider` just works. Switching from `gpt-4o-mini` to Claude or Ollama is a one-line change.
- **Faithful translation, not a leaky abstraction.** ADK `LLMRequest` / `LLMResponse` types map cleanly onto AnyLLM `CompletionParams` and chat completion responses.
- **Streaming that streams.** Token-by-token text partials over AnyLLM channel streaming, with reasoning and tool calls resolved in the final event.
- **Honest about its limits.** Unsupported features fail loudly and typed with `ErrUnsupportedFeature` — never silently dropped.
- **Tiny surface area.** Three options, one constructor. That's the whole API.

## Quickstart

```go
// Pick any AnyLLM provider.
provider, _ := openai.New(anyllm.WithAPIKey(os.Getenv("OPENAI_API_KEY")))

// Wrap it as an ADK model.
llm, _ := adkanyllm.New(
	adkanyllm.WithProvider(provider),
	adkanyllm.WithModel("gpt-4o-mini"),
)

// Drop it into any ADK agent.
assistant, _ := llmagent.New(llmagent.Config{
	Name:        "assistant",
	Model:       llm,
	Instruction: "You are a helpful assistant.",
})
```

Want Claude or Ollama instead? Swap the `openai.New(...)` line for the provider of your choice and pass it to `WithProvider`. Nothing else changes.

## Getting Started

```bash
go get github.com/ksinica/adk-go-anyllm
```

## Configuration

The entire API is three functional options:

| Option | What it does |
| --- | --- |
| `WithProvider(p)` | **Required.** The AnyLLM provider used for completions. |
| `WithModel(name)` | Fallback model used when `LLMRequest.Model` is empty. |
| `WithExtra(map)` | Provider-specific request fields, cloned and merged into every completion. |

## What's supported

| Area | Supported |
| --- | --- |
| User / system / assistant text | ✅ |
| System instruction (`GenerateContentConfig.SystemInstruction`) | ✅ |
| Reasoning / thought parts | ✅ |
| Function calls and tool responses | ✅ |
| Inline / file image parts (user role) | ✅ |
| JSON object / JSON schema response formats | ✅ |
| Temperature, top-p, max tokens, stop, seed | ✅ |
| Tool choice modes (`none`, `any`, `required`, `validated`) | ✅ |
| Streaming text partials | ✅ |
| Top-k, penalties, safety settings, cached content, etc. | ❌ `ErrUnsupportedFeature` |

## Errors you can reason about

Every failure is typed, so you can branch on it instead of parsing strings:

- **`adkanyllm.AdapterError`** — validation and conversion failures (missing model, invalid JSON, token overflow, malformed tool calls).
- **`adkanyllm.ErrUnsupportedFeature`** — a genai/ADK field the adapter doesn't implement. Match with `errors.Is`.
- **`adkanyllm.UnsupportedFeatureError`** — carries the exact unsupported field name. Inspect with `errors.As`.

```go
if errors.Is(err, adkanyllm.ErrUnsupportedFeature) {
    var featureErr *adkanyllm.UnsupportedFeatureError
    if errors.As(err, &featureErr) {
        log.Printf("unsupported: %s", featureErr.Feature)
    }
}
```

Adapter validation errors (unsupported role, invalid part variants, schema conflicts) return `*AdapterError` but deliberately do **not** match `ErrUnsupportedFeature` — a malformed request is your bug, not a missing capability.

## License

MIT. See [LICENSE](LICENSE).
