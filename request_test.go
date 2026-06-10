package adkanyllm

import (
	"errors"
	"testing"

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

	var params anyllm.CompletionParams
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
	})
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
	})
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
	})
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
	})
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

func TestApplyToolConfigModeValidated(t *testing.T) {
	t.Parallel()

	var params anyllm.CompletionParams
	err := applyToolConfig(&params, &genai.GenerateContentConfig{
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeValidated,
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.ToolChoice != "auto" {
		t.Fatalf("ToolChoice=%#v expected auto", params.ToolChoice)
	}
}

func TestApplyToolConfigModeAnyRequiresToolChoice(t *testing.T) {
	t.Parallel()

	var params anyllm.CompletionParams
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
	if params.ToolChoice != "required" {
		t.Fatalf("ToolChoice=%#v expected required", params.ToolChoice)
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
	})
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
		{"config.tools", &genai.GenerateContentConfig{Tools: []*genai.Tool{{}}}},
		{"cachedContent", &genai.GenerateContentConfig{CachedContent: "cache-001"}},
		{"responseModalities", &genai.GenerateContentConfig{ResponseModalities: []string{"text"}}},
		{"mediaResolution", &genai.GenerateContentConfig{MediaResolution: genai.MediaResolutionHigh}},
		{"speechConfig", &genai.GenerateContentConfig{SpeechConfig: &genai.SpeechConfig{}}},
		{"audioTimestamp", &genai.GenerateContentConfig{AudioTimestamp: true}},
		{"thinkingConfig", &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{}}},
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

func int32Ptr(v int32) *int32       { return &v }
func float32Ptr(v float32) *float32 { return &v }

func TestToolMessageFromFunctionResponse(t *testing.T) {
	t.Parallel()

	// nil response.
	_, err := toolMessageFromFunctionResponse(nil)
	if err == nil {
		t.Fatal("expected error for nil response")
	}

	// Missing ID.
	_, err = toolMessageFromFunctionResponse(&genai.FunctionResponse{
		Response: map[string]any{"ok": true},
	})
	if err == nil {
		t.Fatal("expected error for missing id")
	}

	// Success with response payload.
	msg, err := toolMessageFromFunctionResponse(&genai.FunctionResponse{
		ID:       "call_1",
		Response: map[string]any{"temp": 20},
	})
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
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}

	// Parts unsupported.
	_, err = toolMessageFromFunctionResponse(&genai.FunctionResponse{
		ID:    "call_1",
		Parts: []*genai.FunctionResponsePart{genai.NewFunctionResponsePartFromBytes([]byte("data"), "text/plain")},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

func TestToolCallFromFunctionCall(t *testing.T) {
	t.Parallel()

	// nil call.
	_, err := toolCallFromFunctionCall(nil, 0)
	if err == nil {
		t.Fatal("expected error for nil call")
	}

	// Missing name.
	_, err = toolCallFromFunctionCall(&genai.FunctionCall{ID: "c1"}, 0)
	if err == nil {
		t.Fatal("expected error for empty name")
	}

	// Auto-generated ID when missing.
	tc, err := toolCallFromFunctionCall(&genai.FunctionCall{Name: "get_weather"}, 1)
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
	}, 0)
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
	}, 0)
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
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
	})
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
			})
			if !errors.Is(err, ErrUnsupportedFeature) {
				t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
			}
		})
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
	})
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
