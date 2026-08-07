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
- **Tiny surface area.** One required argument, two options, one constructor. That's the whole API.

## Quickstart

```go
// Pick any AnyLLM provider.
provider, _ := openai.New(anyllm.WithAPIKey(os.Getenv("OPENAI_API_KEY")))

// Wrap it as an ADK model.
llm, _ := adkanyllm.New(
 provider,
 adkanyllm.WithModel("gpt-4o-mini"),
)

// Drop it into any ADK agent.
assistant, _ := llmagent.New(llmagent.Config{
 Name:        "assistant",
 Model:       llm,
 Instruction: "You are a helpful assistant.",
})
```

Want Claude or Ollama instead? Swap the `openai.New(...)` line for the provider of your choice. Nothing else changes.

## Configuration

The entire API is `New(provider, opts...)` — one required argument and two options:

| Argument / Option | What it does |
| --- | --- |
| `provider` (first arg) | **Required.** The AnyLLM provider used for completions. |
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
| Tool choice modes (`auto`, `none`, `any`, including allowed function names) | ✅ |
| Tool choice mode (`validated`) | ❌ `ErrUnsupportedFeature` — AnyLLM cannot enforce schema-validated calls |
| Streaming text partials | ✅ |
| Streaming a single tool call (with or without id-per-fragment) | ✅ |
| Streaming **parallel** tool calls from a provider that omits ids on continuation fragments | ❌ fails loudly — see below |
| Top-k, penalties, safety settings, cached content, etc. | ❌ `ErrUnsupportedFeature` |

### Streaming tool-call limitation

any-llm-go's streamed `ToolCall` has no wire "index" field — the OpenAI-compatible
protocol uses it to route a continuation fragment back to the call it belongs to,
but any-llm-go's provider drops it when building deltas. As long as providers keep
sending the id on every fragment, or only one tool call streams at a time, fragments
can still be correlated by id (or attributed to the sole in-flight call). Once two or
more tool calls are open concurrently *and* the provider sends bare, id-less
continuation fragments, there is no way to tell which call a fragment belongs to.
Rather than guess, the adapter returns an error in that case instead of silently
mis-attributing (or dropping) arguments. Non-streaming responses are unaffected.

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
