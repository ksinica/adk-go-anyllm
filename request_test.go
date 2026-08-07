package adkanyllm

import (
	"errors"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func ptrFloat32(v float32) *float32 {
	return &v
}

func TestBuildCompletionParamsRequiresModel(t *testing.T) {
	t.Parallel()

	_, err := buildCompletionParams(&model.LLMRequest{}, "", nil)
	if err == nil {
		t.Fatal("expected error for missing model name")
	}
}

func TestBuildCompletionParamsRejectsEmptyMessages(t *testing.T) {
	t.Parallel()

	_, err := buildCompletionParams(&model.LLMRequest{
		Model: "gpt-4o-mini",
	}, "", nil)
	if err == nil {
		t.Fatal("expected error for request with no resolved messages")
	}
}

func TestBuildCompletionParamsUserText(t *testing.T) {
	t.Parallel()

	params, err := buildCompletionParams(&model.LLMRequest{
		Model: "gpt-4o-mini",
		Contents: []*genai.Content{
			genai.NewContentFromText("hello", genai.RoleUser),
		},
	}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(params.Messages))
	}
	if params.Messages[0].Role != anyllm.RoleUser {
		t.Fatalf("role=%q expected user", params.Messages[0].Role)
	}
	if params.Messages[0].ContentString() != "hello" {
		t.Fatalf("content=%q expected hello", params.Messages[0].ContentString())
	}
}

func TestBuildCompletionParamsSystemInstruction(t *testing.T) {
	t.Parallel()

	params, err := buildCompletionParams(&model.LLMRequest{
		Model: "gpt-4o-mini",
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("be helpful", genai.RoleUser),
		},
		Contents: []*genai.Content{
			genai.NewContentFromText("hello", genai.RoleUser),
		},
	}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(params.Messages))
	}
	if params.Messages[0].Role != anyllm.RoleSystem || params.Messages[0].ContentString() != "be helpful" {
		t.Fatalf("unexpected system message: %#v", params.Messages[0])
	}
}

func TestBuildCompletionParamsThoughtRoundTripInput(t *testing.T) {
	t.Parallel()

	params, err := buildCompletionParams(&model.LLMRequest{
		Model: "gpt-4o-mini",
		Contents: []*genai.Content{
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{
					{Text: "thinking", Thought: true},
				},
			},
		},
	}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.Messages[0].Reasoning == nil || params.Messages[0].Reasoning.Content != "thinking" {
		t.Fatalf("expected reasoning content, got %#v", params.Messages[0].Reasoning)
	}
}

func TestApplyConfigToParamsUnsupportedTopK(t *testing.T) {
	t.Parallel()

	var params anyllm.CompletionParams
	cfg := &genai.GenerateContentConfig{
		TopK: ptrFloat32(4),
	}

	err := applyConfigToParams(&params, cfg)
	if err == nil {
		t.Fatal("expected unsupported topK error")
	}
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestApplyConfigToParamsRejectsPresencePenalty(t *testing.T) {
	t.Parallel()

	penalty := float32(0.5)
	var params anyllm.CompletionParams
	err := applyConfigToParams(&params, &genai.GenerateContentConfig{
		PresencePenalty: &penalty,
	})
	if err == nil {
		t.Fatal("expected unsupported presencePenalty error")
	}
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestApplyToolConfigUnsupportedBranches(t *testing.T) {
	t.Parallel()

	// RetrievalConfig.
	err := applyToolConfig(&anyllm.CompletionParams{}, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			RetrievalConfig: &genai.RetrievalConfig{},
		},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}

	// IncludeServerSideToolInvocations.
	err = applyToolConfig(&anyllm.CompletionParams{}, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			IncludeServerSideToolInvocations: func() *bool { v := true; return &v }(),
		},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}

	// StreamFunctionCallArguments.
	err = applyToolConfig(&anyllm.CompletionParams{}, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				StreamFunctionCallArguments: func() *bool { v := true; return &v }(),
			},
		},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}

	// Unsupported mode (lowercase mismatch -> hits default case).
	err = applyToolConfig(&anyllm.CompletionParams{}, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: "CUSTOM_MODE",
			},
		},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestApplyToolConfigAcceptsExplicitFalseBoolFields(t *testing.T) {
	t.Parallel()

	falsePtr := func() *bool { v := false; return &v }

	// IncludeServerSideToolInvocations=false is the normal "off" state, not a signal.
	err := applyToolConfig(&anyllm.CompletionParams{}, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			IncludeServerSideToolInvocations: falsePtr(),
		},
	})
	if err != nil {
		t.Fatalf("unexpected error for IncludeServerSideToolInvocations=false: %v", err)
	}

	// StreamFunctionCallArguments=false is the normal "off" state, not a signal.
	err = applyToolConfig(&anyllm.CompletionParams{}, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				StreamFunctionCallArguments: falsePtr(),
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error for StreamFunctionCallArguments=false: %v", err)
	}
}

func TestResponseFormatFromConfig(t *testing.T) {
	t.Parallel()

	tests := []struct {
		cfg                *genai.GenerateContentConfig
		name               string
		expectedNil        bool
		expectedErr        bool
		expectedJSONObject bool
		expectedJSONSchema bool
	}{
		{
			name:        "no response mime type",
			cfg:         &genai.GenerateContentConfig{},
			expectedNil: true,
		},
		{
			name: "json object mode",
			cfg: &genai.GenerateContentConfig{
				ResponseMIMEType: "application/json",
			},
			expectedJSONObject: true,
		},
		{
			name: "unsupported mime type",
			cfg: &genai.GenerateContentConfig{
				ResponseMIMEType: "application/xml",
			},
			expectedErr: true,
		},
		{
			name: "schema without mime defaults to json schema mode",
			cfg: &genai.GenerateContentConfig{
				ResponseSchema: &genai.Schema{Type: genai.TypeObject},
			},
			expectedJSONSchema: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, err := responseFormatFromConfig(tt.cfg)
			if tt.expectedErr {
				if err == nil {
					t.Fatal("expected error but got nil")
				}
				if !errors.Is(err, ErrUnsupportedFeature) {
					t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.expectedNil && got != nil {
				t.Fatal("expected nil response format")
			}
			if tt.expectedJSONObject && (got == nil || got.Type != responseFormatJSONObject) {
				t.Fatalf("expected json_object format, got %#v", got)
			}
			if tt.expectedJSONSchema && (got == nil || got.Type != responseFormatJSONSchema) {
				t.Fatalf("expected json_schema format, got %#v", got)
			}
		})
	}
}

func TestResponseFormatFromConfigSchemaAndJsonSchemaMutuallyExclusive(t *testing.T) {
	t.Parallel()

	_, err := responseFormatFromConfig(&genai.GenerateContentConfig{
		ResponseSchema:     &genai.Schema{Type: genai.TypeObject},
		ResponseJsonSchema: &genai.Schema{Type: genai.TypeObject},
	})
	if err == nil {
		t.Fatal("expected error for mutually exclusive schemas")
	}
}

// TestResponseFormatFromConfigTypedNilResponseJsonSchemaIgnored verifies
// that a typed-nil ResponseJsonSchema (e.g. a nil *jsonschema.Schema boxed
// into the any field) is treated as absent: it must not trip the
// mutual-exclusivity check against a real ResponseSchema, nor produce a
// json_schema format built from a nil schema.
func TestResponseFormatFromConfigTypedNilResponseJsonSchemaIgnored(t *testing.T) {
	t.Parallel()

	var typedNilJSONSchema *jsonschema.Schema

	got, err := responseFormatFromConfig(&genai.GenerateContentConfig{
		ResponseSchema:     &genai.Schema{Type: genai.TypeObject},
		ResponseJsonSchema: typedNilJSONSchema,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got == nil || got.Type != responseFormatJSONSchema {
		t.Fatalf("expected json_schema built from ResponseSchema, got %#v", got)
	}
}

func TestResponseFormatFromConfigSchemaWithNonJsonMime(t *testing.T) {
	t.Parallel()

	_, err := responseFormatFromConfig(&genai.GenerateContentConfig{
		ResponseMIMEType: "application/xml",
		ResponseSchema:   &genai.Schema{Type: genai.TypeObject},
	})
	if err == nil {
		t.Fatal("expected error for schema with non-json mime")
	}
}

func TestResponseFormatFromConfigUnsupportedMimeType(t *testing.T) {
	t.Parallel()

	_, err := responseFormatFromConfig(&genai.GenerateContentConfig{
		ResponseMIMEType: "application/xml",
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestResponseFormatFromConfigJsonSchemaMode(t *testing.T) {
	t.Parallel()

	got, err := responseFormatFromConfig(&genai.GenerateContentConfig{
		ResponseJsonSchema: &genai.Schema{Type: genai.TypeObject},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Type != responseFormatJSONSchema {
		t.Fatalf("expected json_schema, got %#v", got)
	}
}

func TestResponseFormatFromConfigSchemaWithApplicationJson(t *testing.T) {
	t.Parallel()

	got, err := responseFormatFromConfig(&genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		ResponseSchema:   &genai.Schema{Type: genai.TypeObject},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Type != responseFormatJSONSchema {
		t.Fatalf("expected json_schema, got %#v", got)
	}
}

func TestApplyToolConfigModeNone(t *testing.T) {
	t.Parallel()

	var params anyllm.CompletionParams
	err := applyToolConfig(&params, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeNone,
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.ToolChoice != "none" {
		t.Fatalf("ToolChoice=%#v expected none", params.ToolChoice)
	}
}

func TestApplyToolConfigModeAnyWithAllowedName(t *testing.T) {
	t.Parallel()

	params := anyllm.CompletionParams{
		Tools: []anyllm.Tool{
			{Type: toolTypeFunction, Function: anyllm.Function{Name: "get_weather"}},
		},
	}
	err := applyToolConfig(&params, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{"get_weather"},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	choice, ok := params.ToolChoice.(anyllm.ToolChoice)
	if !ok || choice.Function == nil || choice.Function.Name != "get_weather" {
		t.Fatalf("unexpected tool choice: %#v", params.ToolChoice)
	}
}

// TestApplyToolConfigModeAnySingleNameNotDeclaredIsUnsupported verifies that
// a single allowedFunctionNames entry that names no declared tool fails
// loudly instead of silently pinning ToolChoice to a function the model was
// never told about.
func TestApplyToolConfigModeAnySingleNameNotDeclaredIsUnsupported(t *testing.T) {
	t.Parallel()

	params := anyllm.CompletionParams{
		Tools: []anyllm.Tool{
			{Type: toolTypeFunction, Function: anyllm.Function{Name: "get_time"}},
		},
	}
	err := applyToolConfig(&params, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{"get_weather"},
			},
		},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

// TestApplyToolConfigModeAnyNoAllowedNamesRequiresDeclaredTools verifies that
// mode ANY with no allowedFunctionNames (an unrestricted "any tool") fails
// loudly when there are no declared tools to require use of, rather than
// setting ToolChoice=required with nothing for the provider to choose from.
func TestApplyToolConfigModeAnyNoAllowedNamesRequiresDeclaredTools(t *testing.T) {
	t.Parallel()

	var params anyllm.CompletionParams
	err := applyToolConfig(&params, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAny,
			},
		},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

// TestApplyToolConfigModeAnyNoAllowedNamesWithDeclaredTools verifies the
// success path for mode ANY with no allowedFunctionNames: it requires tool
// use from among whatever tools are already declared.
func TestApplyToolConfigModeAnyNoAllowedNamesWithDeclaredTools(t *testing.T) {
	t.Parallel()

	params := anyllm.CompletionParams{
		Tools: []anyllm.Tool{
			{Type: toolTypeFunction, Function: anyllm.Function{Name: "get_weather"}},
		},
	}
	err := applyToolConfig(&params, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAny,
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.ToolChoice != toolChoiceRequired {
		t.Fatalf("ToolChoice=%#v expected required", params.ToolChoice)
	}
}

func TestConvertContentFunctionCall(t *testing.T) {
	t.Parallel()

	messages, err := convertContent(&genai.Content{
		Role: genai.RoleModel,
		Parts: []*genai.Part{{
			FunctionCall: &genai.FunctionCall{
				ID:   "call_1",
				Name: "get_weather",
				Args: map[string]any{"city": "Paris"},
			},
		}},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 1 || len(messages[0].ToolCalls) != 1 {
		t.Fatalf("unexpected messages: %#v", messages)
	}
	if messages[0].ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("tool name=%q", messages[0].ToolCalls[0].Function.Name)
	}
}

func TestConvertContentFunctionResponse(t *testing.T) {
	t.Parallel()

	messages, err := convertContent(&genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       "call_1",
				Response: map[string]any{"temp": 20},
			},
		}},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 1 || messages[0].Role != anyllm.RoleTool {
		t.Fatalf("unexpected messages: %#v", messages)
	}
}

func TestConvertContentRejectsMixedToolAndText(t *testing.T) {
	t.Parallel()

	_, err := convertContent(&genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			{Text: "hello"},
			{FunctionResponse: &genai.FunctionResponse{ID: "call_1"}},
		},
	}, nil)
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestConvertContentImageFileData(t *testing.T) {
	t.Parallel()

	messages, err := convertContent(&genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			FileData: &genai.FileData{
				FileURI:  "https://example.com/image.png",
				MIMEType: "image/png",
			},
		}},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts, ok := messages[0].Content.([]anyllm.ContentPart)
	if !ok || parts[0].ImageURL == nil || parts[0].ImageURL.URL != "https://example.com/image.png" {
		t.Fatalf("unexpected message content: %#v", messages[0].Content)
	}
}

func TestBuildMessagesUnsupportedRole(t *testing.T) {
	t.Parallel()

	_, err := buildMessages(&model.LLMRequest{
		Contents: []*genai.Content{{
			Role:  "narrator",
			Parts: []*genai.Part{{Text: "hello"}},
		}},
	})
	if err == nil {
		t.Fatal("expected unsupported role error")
	}
	var adapterErr *AdapterError
	if !errors.As(err, &adapterErr) {
		t.Fatalf("expected *AdapterError, got %T", err)
	}
}

// TestApplyToolConfigModeValidatedIsUnsupported verifies that VALIDATED mode
// is rejected loudly rather than silently downgraded to AUTO: AnyLLM has no
// strict/schema-validated tool-choice equivalent, so silently downgrading
// would mislead the caller into believing calls are constrained to
// schema-valid arguments when they are not.
func TestApplyToolConfigModeValidatedIsUnsupported(t *testing.T) {
	t.Parallel()

	var params anyllm.CompletionParams
	err := applyToolConfig(&params, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeValidated,
			},
		},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestApplyToolConfigModeAnyWithoutDeclaredToolsIsUnsupported(t *testing.T) {
	t.Parallel()

	// With multiple allowed names and no declared tools to restrict,
	// AnyLLM cannot represent the allow-list; broadening to unrestricted
	// "required" would silently drop the restriction, so this must fail loudly.
	var params anyllm.CompletionParams
	err := applyToolConfig(&params, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{"a", "b"},
			},
		},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestApplyToolConfigModeAnyMultipleNamesFiltersTools(t *testing.T) {
	t.Parallel()

	params := anyllm.CompletionParams{
		Tools: []anyllm.Tool{
			{Type: toolTypeFunction, Function: anyllm.Function{Name: "a"}},
			{Type: toolTypeFunction, Function: anyllm.Function{Name: "b"}},
			{Type: toolTypeFunction, Function: anyllm.Function{Name: "c"}},
		},
	}
	err := applyToolConfig(&params, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{"a", "b"},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.ToolChoice != toolChoiceRequired {
		t.Fatalf("ToolChoice=%#v expected required", params.ToolChoice)
	}
	if len(params.Tools) != 2 {
		t.Fatalf("expected tools filtered to allow-list, got %#v", params.Tools)
	}
	for _, tool := range params.Tools {
		if tool.Function.Name != "a" && tool.Function.Name != "b" {
			t.Fatalf("unexpected tool leaked through allow-list filter: %#v", tool)
		}
	}
}

func TestApplyToolConfigModeAnyMultipleNamesMatchesNoneIsUnsupported(t *testing.T) {
	t.Parallel()

	params := anyllm.CompletionParams{
		Tools: []anyllm.Tool{
			{Type: toolTypeFunction, Function: anyllm.Function{Name: "c"}},
		},
	}
	err := applyToolConfig(&params, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{"a", "b"},
			},
		},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestConvertContentImageInlineData(t *testing.T) {
	t.Parallel()

	messages, err := convertContent(&genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			InlineData: &genai.Blob{
				MIMEType: "image/png",
				Data:     []byte{0x89, 0x50, 0x4e, 0x47},
			},
		}},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts, ok := messages[0].Content.([]anyllm.ContentPart)
	if !ok || len(parts) != 1 || parts[0].Type != contentTypeImageURL {
		t.Fatalf("unexpected message content: %#v", messages[0].Content)
	}
}

func TestApplyConfigToParamsHappyPath(t *testing.T) {
	t.Parallel()

	temp := float32(0.5)
	topP := float32(0.5)
	seed := int32(42)
	var params anyllm.CompletionParams
	err := applyConfigToParams(&params, &genai.GenerateContentConfig{
		Temperature:     &temp,
		TopP:            &topP,
		MaxOutputTokens: 2048,
		StopSequences:   []string{".", "!"},
		Seed:            &seed,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.Temperature == nil || *params.Temperature != 0.5 {
		t.Fatalf("Temperature=%#v", params.Temperature)
	}
	if params.TopP == nil || *params.TopP != 0.5 {
		t.Fatalf("TopP=%#v", params.TopP)
	}
	if params.MaxTokens == nil || *params.MaxTokens != 2048 {
		t.Fatalf("MaxTokens=%#v", params.MaxTokens)
	}
	if len(params.Stop) != 2 || params.Stop[0] != "." {
		t.Fatalf("Stop=%#v", params.Stop)
	}
	if params.Seed == nil || *params.Seed != 42 {
		t.Fatalf("Seed=%#v", params.Seed)
	}
}

func TestApplyConfigToParamsUnsupportedHttpOptions(t *testing.T) {
	t.Parallel()

	err := applyConfigToParams(&anyllm.CompletionParams{}, &genai.GenerateContentConfig{
		HTTPOptions: &genai.HTTPOptions{},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestApplyConfigToParamsUnsupportedRouting(t *testing.T) {
	t.Parallel()

	err := applyConfigToParams(&anyllm.CompletionParams{}, &genai.GenerateContentConfig{
		RoutingConfig: &genai.GenerationConfigRoutingConfig{},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestApplyConfigToParamsMaxTokens(t *testing.T) {
	t.Parallel()

	cfg := &genai.GenerateContentConfig{
		CandidateCount: 2,
	}
	err := applyConfigToParams(&anyllm.CompletionParams{}, cfg)
	if err == nil {
		t.Fatal("expected error for CandidateCount > 1")
	}
}

// TestApplyConfigToParamsRejectsNegativeCandidateCount verifies the item-4
// fix: a negative CandidateCount fails loudly instead of being silently
// treated as unset.
func TestApplyConfigToParamsRejectsNegativeCandidateCount(t *testing.T) {
	t.Parallel()

	err := applyConfigToParams(&anyllm.CompletionParams{}, &genai.GenerateContentConfig{
		CandidateCount: -1,
	})
	if err == nil {
		t.Fatal("expected error for negative CandidateCount")
	}
}

// TestApplyConfigToParamsRejectsNegativeMaxOutputTokens verifies the item-4
// fix: a negative MaxOutputTokens fails loudly instead of being silently
// treated as unset.
func TestApplyConfigToParamsRejectsNegativeMaxOutputTokens(t *testing.T) {
	t.Parallel()

	err := applyConfigToParams(&anyllm.CompletionParams{}, &genai.GenerateContentConfig{
		MaxOutputTokens: -1,
	})
	if err == nil {
		t.Fatal("expected error for negative MaxOutputTokens")
	}
}

func TestApplyConfigToParamsUnsupportedFeatures(t *testing.T) {
	t.Parallel()

	// Helper to quickly set a bool pointer field and create config.
	ptr := func(v bool) *bool { return &v }

	tests := []struct {
		name string
		cfg  *genai.GenerateContentConfig
	}{
		{"modelSelectionConfig", &genai.GenerateContentConfig{ModelSelectionConfig: &genai.ModelSelectionConfig{}}},
		{"safetySettings", &genai.GenerateContentConfig{SafetySettings: []*genai.SafetySetting{{Category: genai.HarmCategoryHarassment, Threshold: genai.HarmBlockThresholdBlockLowAndAbove}}}},
		{"cachedContent", &genai.GenerateContentConfig{CachedContent: "cache-001"}},
		{"responseModalities", &genai.GenerateContentConfig{ResponseModalities: []string{"text"}}},
		{"mediaResolution", &genai.GenerateContentConfig{MediaResolution: genai.MediaResolutionHigh}},
		{"speechConfig", &genai.GenerateContentConfig{SpeechConfig: &genai.SpeechConfig{}}},
		{"audioTimestamp", &genai.GenerateContentConfig{AudioTimestamp: true}},
		{"imageConfig", &genai.GenerateContentConfig{ImageConfig: &genai.ImageConfig{}}},
		{"enableEnhancedCivicAnswers", &genai.GenerateContentConfig{EnableEnhancedCivicAnswers: ptr(true)}},
		{"modelArmorConfig", &genai.GenerateContentConfig{ModelArmorConfig: &genai.ModelArmorConfig{}}},
		{"serviceTier", &genai.GenerateContentConfig{ServiceTier: "STANDARD"}},
		{"labels", &genai.GenerateContentConfig{Labels: map[string]string{"env": "test"}}},
		{"responseLogprobs", &genai.GenerateContentConfig{ResponseLogprobs: true}},
		{"logprobs", &genai.GenerateContentConfig{Logprobs: int32Ptr(3)}},
		{"frequencyPenalty", &genai.GenerateContentConfig{FrequencyPenalty: float32Ptr(0.5)}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := applyConfigToParams(&anyllm.CompletionParams{}, tt.cfg)
			if !errors.Is(err, ErrUnsupportedFeature) {
				t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
			}
		})
	}
}

func TestApplyConfigToParamsAcceptsEnableEnhancedCivicAnswersFalse(t *testing.T) {
	t.Parallel()

	falseVal := false
	err := applyConfigToParams(&anyllm.CompletionParams{}, &genai.GenerateContentConfig{
		EnableEnhancedCivicAnswers: &falseVal,
	})
	if err != nil {
		t.Fatalf("unexpected error for EnableEnhancedCivicAnswers=false: %v", err)
	}
}

func TestApplyThinkingConfig(t *testing.T) {
	t.Parallel()

	minimal := genai.ThinkingLevelMinimal
	low := genai.ThinkingLevelLow
	medium := genai.ThinkingLevelMedium
	high := genai.ThinkingLevelHigh
	unspecified := genai.ThinkingLevelUnspecified

	budget := func(v int32) *int32 { return &v }

	tests := []struct {
		name          string
		cfg           *genai.GenerateContentConfig
		wantEffort    anyllm.ReasoningEffort
		wantErr       bool
		wantErrUnsupp bool
	}{
		// Level-based mapping (most explicit signal).
		{
			name: "nil config",
			cfg:  nil,
		},
		{
			name: "nil ThinkingConfig",
			cfg:  &genai.GenerateContentConfig{},
		},
		{
			name: "all defaults no-op",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{},
			},
		},
		{
			name: "IncludeThoughts=false no-op",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: false},
			},
		},
		{
			name: "ThinkingLevelMinimal",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: minimal},
			},
			wantEffort: anyllm.ReasoningEffortLow,
		},
		{
			name: "ThinkingLevelLow",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: low},
			},
			wantEffort: anyllm.ReasoningEffortLow,
		},
		{
			name: "ThinkingLevelMedium",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: medium},
			},
			wantEffort: anyllm.ReasoningEffortMedium,
		},
		{
			name: "ThinkingLevelHigh",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: high},
			},
			wantEffort: anyllm.ReasoningEffortHigh,
		},
		{
			name: "IncludeThoughts with no level or budget defaults to auto",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: true},
			},
			wantEffort: anyllm.ReasoningEffortAuto,
		},
		// Budget-based fallback (no ThinkingLevel set).
		{
			name: "budget 0 disables thinking",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: budget(0)},
			},
			wantEffort: anyllm.ReasoningEffortNone,
		},
		{
			name: "budget negative no-op",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: budget(-1)},
			},
		},
		{
			name: "budget 500 -> low",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: budget(500)},
			},
			wantEffort: anyllm.ReasoningEffortLow,
		},
		{
			name: "budget 1024 -> low",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: budget(1024)},
			},
			wantEffort: anyllm.ReasoningEffortLow,
		},
		{
			name: "budget 5000 -> medium",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: budget(5000)},
			},
			wantEffort: anyllm.ReasoningEffortMedium,
		},
		{
			name: "budget 8192 -> medium",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: budget(8192)},
			},
			wantEffort: anyllm.ReasoningEffortMedium,
		},
		{
			name: "budget 10000 -> medium",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: budget(10000)},
			},
			wantEffort: anyllm.ReasoningEffortMedium,
		},
		{
			name: "budget 20000 -> high",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: budget(20000)},
			},
			wantEffort: anyllm.ReasoningEffortHigh,
		},
		{
			name: "budget 24576 -> high",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: budget(24576)},
			},
			wantEffort: anyllm.ReasoningEffortHigh,
		},
		// Level takes precedence over budget when both are set.
		{
			name: "level+Medium overrides budget 500",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingBudget: budget(500),
					ThinkingLevel:  medium,
				},
			},
			wantEffort: anyllm.ReasoningEffortMedium,
		},
		// Unspecified level with budget maps by budget.
		{
			name: "unspecified level + big budget maps by budget",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingBudget: budget(24576),
					ThinkingLevel:  unspecified,
				},
			},
			wantEffort: anyllm.ReasoningEffortHigh,
		},
		// Invalid ThinkingLevel is rejected.
		{
			name: "invalid ThinkingLevel",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingLevel: genai.ThinkingLevel("SUPER"),
				},
			},
			wantErrUnsupp: true,
		},
		// Integration: applyConfigToParams happy path with ThinkingConfig.
		{
			name: "level via applyConfigToParams sets ReasoningEffort",
			cfg: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: high},
			},
			wantEffort: anyllm.ReasoningEffortHigh,
		},
	}

	// The last test uses the outer applyConfigToParams; the rest call applyThinkingConfig directly.
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var params anyllm.CompletionParams

			var err error
			// Use applyConfigToParams for the integration test, applyThinkingConfig for unit tests.
			if tt.name == "level via applyConfigToParams sets ReasoningEffort" {
				err = applyConfigToParams(&params, tt.cfg)
			} else {
				err = applyThinkingConfig(&params, tt.cfg)
			}

			if tt.wantErrUnsupp {
				if !errors.Is(err, ErrUnsupportedFeature) {
					t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
				}
				return
			}

			if tt.wantErr && err == nil {
				t.Fatal("expected error but got nil")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if err != nil {
				return
			}

			if params.ReasoningEffort != tt.wantEffort {
				t.Fatalf("ReasoningEffort=%q, want %q", params.ReasoningEffort, tt.wantEffort)
			}
		})
	}
}

func int32Ptr(v int32) *int32       { return &v }
func float32Ptr(v float32) *float32 { return &v }

func TestApplyConfigToParamsToolsAccepted(t *testing.T) {
	t.Parallel()

	// Tools in Config.Tools are redundant (handled via req.Tools)
	// and must be silently accepted, not rejected.
	err := applyConfigToParams(&anyllm.CompletionParams{}, &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{Name: "ping", Description: "test"},
			},
		}},
	})
	if err != nil {
		t.Fatalf("expected no error for tools in Config.Tools, got: %v", err)
	}
}

func TestBuildCompletionParamsAllowedFunctionNamesFiltersDeclaredTools(t *testing.T) {
	t.Parallel()

	params, err := buildCompletionParams(&model.LLMRequest{
		Model: "gpt-4o-mini",
		Contents: []*genai.Content{
			genai.NewContentFromText("hello", genai.RoleUser),
		},
		Tools: map[string]any{
			"a": map[string]any{"name": "a"},
			"b": map[string]any{"name": "b"},
			"c": map[string]any{"name": "c"},
		},
		Config: &genai.GenerateContentConfig{
			ToolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"a", "b"},
				},
			},
		},
	}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Tools) != 2 {
		t.Fatalf("expected tools filtered to allow-list, got %#v", params.Tools)
	}
	if params.ToolChoice != toolChoiceRequired {
		t.Fatalf("ToolChoice=%#v expected required", params.ToolChoice)
	}
}

func TestToolMessageFromFunctionResponse(t *testing.T) {
	t.Parallel()

	// nil response.
	_, err := toolMessageFromFunctionResponse(nil, nil)
	if err == nil {
		t.Fatal("expected error for nil response")
	}

	// Missing ID with no pending id-less call to resolve against.
	_, err = toolMessageFromFunctionResponse(&genai.FunctionResponse{
		Response: map[string]any{"ok": true},
	}, nil)
	if err == nil {
		t.Fatal("expected error for missing id")
	}

	// Success with response payload.
	msg, err := toolMessageFromFunctionResponse(&genai.FunctionResponse{
		ID:       "call_1",
		Response: map[string]any{"temp": 20},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if msg.Role != anyllm.RoleTool || msg.ToolCallID != "call_1" {
		t.Fatalf("unexpected message: %#v", msg)
	}
	if msg.Content == "" {
		t.Fatal("expected serialised response content")
	}

	// WillContinue unsupported.
	willContinue := true
	_, err = toolMessageFromFunctionResponse(&genai.FunctionResponse{
		ID:           "call_1",
		WillContinue: &willContinue,
	}, nil)
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}

	// WillContinue=false is the normal "off" state, not a signal.
	willContinueFalse := false
	_, err = toolMessageFromFunctionResponse(&genai.FunctionResponse{
		ID:           "call_1",
		Response:     map[string]any{"ok": true},
		WillContinue: &willContinueFalse,
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error for WillContinue=false: %v", err)
	}

	// Parts unsupported.
	_, err = toolMessageFromFunctionResponse(&genai.FunctionResponse{
		ID:    "call_1",
		Parts: []*genai.FunctionResponsePart{genai.NewFunctionResponsePartFromBytes([]byte("data"), "text/plain")},
	}, nil)
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

// TestToolMessageFromFunctionResponseResolvesIDLessPendingCall verifies that
// an ID-less FunctionResponse resolves back to the synthetic id that was
// assigned to a preceding ID-less FunctionCall for the same function name.
func TestToolMessageFromFunctionResponseResolvesIDLessPendingCall(t *testing.T) {
	t.Parallel()

	pending := newToolCallIDTracker()
	pending.add("get_weather", "get_weather_1")

	msg, err := toolMessageFromFunctionResponse(&genai.FunctionResponse{
		Name:     "get_weather",
		Response: map[string]any{"temp": 20},
	}, pending)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if msg.ToolCallID != "get_weather_1" {
		t.Fatalf("ToolCallID=%q expected get_weather_1", msg.ToolCallID)
	}
}

// TestToolMessageFromFunctionResponseNoMatchingPendingCallErrors verifies
// that an ID-less FunctionResponse with no matching pending id-less call
// fails loudly instead of silently proceeding with an empty tool call id.
func TestToolMessageFromFunctionResponseNoMatchingPendingCallErrors(t *testing.T) {
	t.Parallel()

	pending := newToolCallIDTracker()
	pending.add("get_time", "get_time_1")

	_, err := toolMessageFromFunctionResponse(&genai.FunctionResponse{
		Name:     "get_weather",
		Response: map[string]any{"temp": 20},
	}, pending)
	if err == nil {
		t.Fatal("expected error for function response with no matching pending call")
	}
}

// TestBuildMessagesIDLessFunctionCallAndResponseRoundTrip verifies the
// end-to-end fix: a Gemini-style history with an ID-less FunctionCall
// followed by an ID-less FunctionResponse for the same function round-trips
// through buildMessages, with the response resolving to the exact synthetic
// id assigned to the call.
func TestBuildMessagesIDLessFunctionCallAndResponseRoundTrip(t *testing.T) {
	t.Parallel()

	messages, err := buildMessages(&model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						Name: "get_weather",
						Args: map[string]any{"city": "Paris"},
					},
				}},
			},
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						Name:     "get_weather",
						Response: map[string]any{"temp": 20},
					},
				}},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}

	callID := messages[0].ToolCalls[0].ID
	if callID == "" {
		t.Fatal("expected a synthesized non-empty call id")
	}
	if messages[1].ToolCallID != callID {
		t.Fatalf("ToolCallID=%q expected to match call id %q", messages[1].ToolCallID, callID)
	}
}

// TestBuildMessagesIDLessFunctionCallAndResponseRoundTripMultipleNames
// verifies that two ID-less calls with distinct names each resolve to their
// own response by name, rather than mismatching in call order.
func TestBuildMessagesIDLessFunctionCallAndResponseRoundTripMultipleNames(t *testing.T) {
	t.Parallel()

	messages, err := buildMessages(&model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{Name: "get_weather"}},
					{FunctionCall: &genai.FunctionCall{Name: "get_time"}},
				},
			},
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						Name:     "get_time",
						Response: map[string]any{"time": "10:00"},
					},
				}},
			},
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						Name:     "get_weather",
						Response: map[string]any{"temp": 20},
					},
				}},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(messages))
	}

	weatherCallID := messages[0].ToolCalls[0].ID
	timeCallID := messages[0].ToolCalls[1].ID
	if weatherCallID == timeCallID {
		t.Fatalf("expected distinct synthesized ids, got %q and %q", weatherCallID, timeCallID)
	}
	if messages[1].ToolCallID != timeCallID {
		t.Fatalf("first response ToolCallID=%q expected to match get_time call id %q", messages[1].ToolCallID, timeCallID)
	}
	if messages[2].ToolCallID != weatherCallID {
		t.Fatalf("second response ToolCallID=%q expected to match get_weather call id %q", messages[2].ToolCallID, weatherCallID)
	}
}

// TestBuildMessagesIDLessFunctionCallDistinctIDsAcrossTurns verifies the
// item-1 fix: two ID-less calls to the same function name in different
// assistant turns get distinct request-wide ids (rather than both resetting
// to e.g. "lookup_1" because id allocation reset per Content), and each
// turn's ID-less response still resolves to the matching call.
func TestBuildMessagesIDLessFunctionCallDistinctIDsAcrossTurns(t *testing.T) {
	t.Parallel()

	messages, err := buildMessages(&model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{Name: "lookup", Args: map[string]any{"q": "first"}},
				}},
			},
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{Name: "lookup", Response: map[string]any{"result": "one"}},
				}},
			},
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{Name: "lookup", Args: map[string]any{"q": "second"}},
				}},
			},
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{Name: "lookup", Response: map[string]any{"result": "two"}},
				}},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(messages))
	}

	firstCallID := messages[0].ToolCalls[0].ID
	secondCallID := messages[2].ToolCalls[0].ID
	if firstCallID == "" || secondCallID == "" || firstCallID == secondCallID {
		t.Fatalf("expected distinct synthesized ids across turns, got %q and %q", firstCallID, secondCallID)
	}
	if messages[1].ToolCallID != firstCallID {
		t.Fatalf("first response ToolCallID=%q expected to match first call id %q", messages[1].ToolCallID, firstCallID)
	}
	if messages[3].ToolCallID != secondCallID {
		t.Fatalf("second response ToolCallID=%q expected to match second call id %q", messages[3].ToolCallID, secondCallID)
	}
}

// TestBuildMessagesRejectsDuplicateExplicitFunctionCallIDs verifies that two
// genai.FunctionCall parts sharing the same explicit, non-empty id anywhere
// in the request are rejected rather than silently accepted, since a
// duplicate id makes the request's tool-call history ambiguous.
func TestBuildMessagesRejectsDuplicateExplicitFunctionCallIDs(t *testing.T) {
	t.Parallel()

	_, err := buildMessages(&model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup", Args: map[string]any{"q": "first"}},
				}},
			},
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup", Args: map[string]any{"q": "second"}},
				}},
			},
		},
	})
	var adapterErr *AdapterError
	if !errors.As(err, &adapterErr) {
		t.Fatalf("expected *AdapterError for duplicate explicit id, got %v", err)
	}
}

func TestToolCallFromFunctionCall(t *testing.T) {
	t.Parallel()

	// nil call.
	_, err := toolCallFromFunctionCall(nil, nil, 0)
	if err == nil {
		t.Fatal("expected error for nil call")
	}

	// Missing name.
	_, err = toolCallFromFunctionCall(&genai.FunctionCall{ID: "c1"}, nil, 0)
	if err == nil {
		t.Fatal("expected error for empty name")
	}

	// Auto-generated ID when missing, via a nil tracker (isolated call, no
	// request-wide collision checking).
	tc, err := toolCallFromFunctionCall(&genai.FunctionCall{Name: "get_weather"}, nil, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tc.ID != "get_weather_1" || tc.Function.Name != "get_weather" || tc.Function.Arguments != "{}" {
		t.Fatalf("unexpected tool call: %#v", tc)
	}

	// Args marshalling.
	tc, err = toolCallFromFunctionCall(&genai.FunctionCall{
		ID:   "c2",
		Name: "search",
		Args: map[string]any{"q": "hello"},
	}, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tc.Function.Arguments != `{"q":"hello"}` {
		t.Fatalf("unexpected args: %s", tc.Function.Arguments)
	}

	// PartialArgs unsupported.
	_, err = toolCallFromFunctionCall(&genai.FunctionCall{
		Name:        "fn",
		PartialArgs: []*genai.PartialArg{{}},
	}, nil, 0)
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}

	// WillContinue=true unsupported.
	willContinueTrue := true
	_, err = toolCallFromFunctionCall(&genai.FunctionCall{
		Name:         "fn",
		WillContinue: &willContinueTrue,
	}, nil, 0)
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}

	// WillContinue=false is the normal "off" state, not a signal.
	willContinueFalse := false
	_, err = toolCallFromFunctionCall(&genai.FunctionCall{
		Name:         "fn",
		WillContinue: &willContinueFalse,
	}, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error for WillContinue=false: %v", err)
	}
}

// TestToolCallIDTrackerAllocateAdvancesPastSeededCollision verifies that
// allocate advances past a candidate id already seeded from an explicit
// genai.FunctionCall.ID elsewhere in the request, rather than handing out a
// colliding synthetic id.
func TestToolCallIDTrackerAllocateAdvancesPastSeededCollision(t *testing.T) {
	t.Parallel()

	tracker := newToolCallIDTracker()
	if err := tracker.seed("lookup_1"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if id := tracker.allocate("lookup", 1); id != "lookup_2" {
		t.Fatalf("id=%q expected lookup_2", id)
	}
}

// TestToolCallIDTrackerSeedRejectsDuplicateExplicitID verifies that seeding
// the same explicit id twice returns a validation error instead of
// silently accepting an ambiguous request-wide id.
func TestToolCallIDTrackerSeedRejectsDuplicateExplicitID(t *testing.T) {
	t.Parallel()

	tracker := newToolCallIDTracker()
	if err := tracker.seed("call_1"); err != nil {
		t.Fatalf("unexpected error on first seed: %v", err)
	}

	err := tracker.seed("call_1")
	var adapterErr *AdapterError
	if !errors.As(err, &adapterErr) {
		t.Fatalf("expected *AdapterError for duplicate seed, got %v", err)
	}
}

func TestContentToText(t *testing.T) {
	t.Parallel()

	// Nil content.
	got, err := contentToText(nil)
	if err != nil || got != "" {
		t.Fatalf("expected empty, got %q err=%v", got, err)
	}

	// Single text part.
	got, err = contentToText(genai.NewContentFromText("hello", genai.RoleUser))
	if err != nil || got != "hello" {
		t.Fatalf("expected hello, got %q err=%v", got, err)
	}

	// Multiple text parts.
	got, err = contentToText(&genai.Content{
		Parts: []*genai.Part{
			{Text: "first"},
			{Text: "second"},
		},
	})
	if err != nil || got != "first\nsecond" {
		t.Fatalf("expected first\\nsecond, got %q err=%v", got, err)
	}

	// Non-text variant in variant count = 1 returns error.
	_, err = contentToText(&genai.Content{
		Parts: []*genai.Part{{
			InlineData: &genai.Blob{MIMEType: "application/octet-stream", Data: []byte{1}},
		}},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestImageURLFromFileData(t *testing.T) {
	t.Parallel()

	// Nil.
	if got := imageURLFromFileData(nil); got != nil {
		t.Fatal("expected nil")
	}

	// Empty URI.
	if got := imageURLFromFileData(&genai.FileData{}); got != nil {
		t.Fatal("expected nil")
	}

	// Non-image mime type.
	if got := imageURLFromFileData(&genai.FileData{
		FileURI:  "https://example.com/data.pdf",
		MIMEType: "application/pdf",
	}); got != nil {
		t.Fatal("expected nil for non-image mime")
	}

	// Valid image file data.
	got := imageURLFromFileData(&genai.FileData{
		FileURI:  "https://example.com/img.png",
		MIMEType: "image/png",
	})
	if got == nil || got.URL != "https://example.com/img.png" {
		t.Fatalf("unexpected imageURL: %#v", got)
	}
}

func TestConvertContentUnsupportedPartVariants(t *testing.T) {
	t.Parallel()

	_, err := convertContent(&genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			Text:       "hello",
			InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}},
		}},
	}, nil)
	if err == nil {
		t.Fatal("expected error for multiple part variants")
	}
}

func TestConvertContentUnsupportedParts(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		part *genai.Part
	}{
		{"video", &genai.Part{VideoMetadata: &genai.VideoMetadata{}}},
		{"executableCode", &genai.Part{ExecutableCode: &genai.ExecutableCode{}}},
		{"codeExecutionResult", &genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{}}},
		{"toolCall", &genai.Part{ToolCall: &genai.ToolCall{}}},
		{"toolResponse", &genai.Part{ToolResponse: &genai.ToolResponse{}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := convertContent(&genai.Content{
				Role:  genai.RoleUser,
				Parts: []*genai.Part{tt.part},
			}, nil)
			if !errors.Is(err, ErrUnsupportedFeature) {
				t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
			}
		})
	}
}

func TestConvertContentUserPureTextJoinsWithNewline(t *testing.T) {
	t.Parallel()

	messages, err := convertContent(&genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			{Text: "first"},
			{Text: "second"},
		},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	content, ok := messages[0].Content.(string)
	if !ok || content != "first\nsecond" {
		t.Fatalf("expected newline-joined string content, got %#v", messages[0].Content)
	}
}

func TestConvertContentThoughtPartRejectsExtraVariant(t *testing.T) {
	t.Parallel()

	_, err := convertContent(&genai.Content{
		Role: genai.RoleModel,
		Parts: []*genai.Part{{
			Thought: true,
			Text:    "thinking",
			FunctionCall: &genai.FunctionCall{
				Name: "get_weather",
			},
		}},
	}, nil)
	if err == nil {
		t.Fatal("expected error for thought part with an extra variant set")
	}
}

func TestConvertContentThoughtPartRejectsModifiers(t *testing.T) {
	t.Parallel()

	_, err := convertContent(&genai.Content{
		Role: genai.RoleModel,
		Parts: []*genai.Part{{
			Thought:       true,
			Text:          "thinking",
			VideoMetadata: &genai.VideoMetadata{},
		}},
	}, nil)
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

// TestConvertContentRejectsThoughtSignatureOnFunctionCallPart verifies that
// a non-empty ThoughtSignature is rejected on a function-call part, not just
// on a thought part: rejectPartModifiers runs for every part, so the
// signature can no longer be silently discarded on text/function-call parts.
func TestConvertContentRejectsThoughtSignatureOnFunctionCallPart(t *testing.T) {
	t.Parallel()

	_, err := convertContent(&genai.Content{
		Role: genai.RoleModel,
		Parts: []*genai.Part{{
			FunctionCall: &genai.FunctionCall{
				Name: "get_weather",
			},
			ThoughtSignature: []byte("opaque-signature"),
		}},
	}, nil)
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

// TestConvertContentRejectsThoughtSignatureOnTextPart mirrors the
// function-call case for a plain text part.
func TestConvertContentRejectsThoughtSignatureOnTextPart(t *testing.T) {
	t.Parallel()

	_, err := convertContent(&genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			Text:             "hello",
			ThoughtSignature: []byte("opaque-signature"),
		}},
	}, nil)
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestContentToTextRejectsModifiers(t *testing.T) {
	t.Parallel()

	_, err := contentToText(&genai.Content{
		Parts: []*genai.Part{{
			Text:          "hello",
			VideoMetadata: &genai.VideoMetadata{},
		}},
	})
	if err == nil {
		t.Fatal("expected error for text part with a modifier set")
	}

	_, err = contentToText(&genai.Content{
		Parts: []*genai.Part{{
			MediaResolution: &genai.PartMediaResolution{},
		}},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestConvertContentUserMultimodal(t *testing.T) {
	t.Parallel()

	messages, err := convertContent(&genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			{Text: "describe this"},
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2, 3}}},
		},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts, ok := messages[0].Content.([]anyllm.ContentPart)
	if !ok || len(parts) != 2 {
		t.Fatalf("expected 2 content parts, got %#v", messages[0].Content)
	}
	if parts[0].Type != contentTypeText || parts[1].Type != contentTypeImageURL {
		t.Fatalf("unexpected part types: %#v", parts)
	}
}
