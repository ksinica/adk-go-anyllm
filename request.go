package adkanyllm

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func buildCompletionParams(
	req *model.LLMRequest,
	defaultModel string,
	extra map[string]any,
) (anyllm.CompletionParams, error) {
	modelName := req.Model
	if modelName == "" {
		modelName = defaultModel
	}
	if modelName == "" {
		return anyllm.CompletionParams{}, newError("model name is required")
	}

	messages, err := buildMessages(req)
	if err != nil {
		return anyllm.CompletionParams{}, err
	}
	if len(messages) == 0 {
		return anyllm.CompletionParams{}, newError("at least one message is required")
	}

	params := anyllm.CompletionParams{
		Model:    modelName,
		Messages: messages,
	}

	if err := applyConfigToParams(&params, req.Config); err != nil {
		return anyllm.CompletionParams{}, err
	}

	tools, err := convertTools(req.Tools)
	if err != nil {
		return anyllm.CompletionParams{}, err
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	if len(extra) > 0 {
		params.Extra = maps.Clone(extra)
	}

	return params, nil
}

func buildMessages(req *model.LLMRequest) ([]anyllm.Message, error) {
	messages := make([]anyllm.Message, 0, len(req.Contents)+1)

	if req.Config != nil && req.Config.SystemInstruction != nil {
		systemText, err := contentToText(req.Config.SystemInstruction)
		if err != nil {
			return nil, err
		}
		if systemText != "" {
			messages = append(messages, anyllm.Message{
				Role:    anyllm.RoleSystem,
				Content: systemText,
			})
		}
	}

	for _, content := range req.Contents {
		converted, err := convertContent(content)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted...)
	}

	return messages, nil
}

func convertContent(content *genai.Content) ([]anyllm.Message, error) {
	if content == nil {
		return nil, nil
	}

	role := strings.ToLower(content.Role)
	if role == "" {
		role = genai.RoleUser
	}
	switch role {
	case genai.RoleUser, genai.RoleModel, anyllm.RoleAssistant, anyllm.RoleSystem:
	default:
		return nil, newErrorf("unsupported role %q", content.Role)
	}

	var (
		textParts           []string
		userMultimodalParts []anyllm.ContentPart
		toolCalls           []anyllm.ToolCall
		messages            []anyllm.Message
		reasoningText       strings.Builder
		hasToolReply        bool
		hasNonToolPart      bool
	)

	for _, part := range content.Parts {
		if part == nil {
			continue
		}

		if part.Thought {
			if len(part.ThoughtSignature) > 0 {
				return nil, unsupportedFeatureError("thoughtSignature")
			}
			if part.Text == "" {
				return nil, unsupportedFeatureError("thought part without text")
			}
			hasNonToolPart = true
			if reasoningText.Len() > 0 {
				reasoningText.WriteByte('\n')
			}
			reasoningText.WriteString(part.Text)
			continue
		}

		if variantCount := partVariantCount(part); variantCount > 1 {
			return nil, newErrorf("invalid part with %d variants set", variantCount)
		}
		switch {
		case part.VideoMetadata != nil:
			return nil, unsupportedFeatureError("videoMetadata part")
		case part.MediaResolution != nil:
			return nil, unsupportedFeatureError("mediaResolution part")
		case len(part.PartMetadata) > 0:
			return nil, unsupportedFeatureError("partMetadata")
		}

		if part.FunctionResponse != nil {
			toolResponseMessage, err := toolMessageFromFunctionResponse(part.FunctionResponse)
			if err != nil {
				return nil, err
			}
			messages = append(messages, toolResponseMessage)
			hasToolReply = true
			continue
		}

		if part.FunctionCall != nil {
			hasNonToolPart = true
			toolCall, err := toolCallFromFunctionCall(part.FunctionCall, len(toolCalls)+1)
			if err != nil {
				return nil, err
			}
			toolCalls = append(toolCalls, toolCall)
			continue
		}

		if part.Text != "" {
			hasNonToolPart = true
			textParts = append(textParts, part.Text)
			userMultimodalParts = append(userMultimodalParts, anyllm.ContentPart{
				Type: contentTypeText,
				Text: part.Text,
			})
			continue
		}

		if imageURL := imageURLFromInlineData(part.InlineData); imageURL != nil {
			if role != genai.RoleUser {
				return nil, unsupportedFeatureErrorf("image part in %q role content", role)
			}
			hasNonToolPart = true
			userMultimodalParts = append(userMultimodalParts, anyllm.ContentPart{
				Type:     contentTypeImageURL,
				ImageURL: imageURL,
			})
			continue
		}
		if imageURL := imageURLFromFileData(part.FileData); imageURL != nil {
			if role != genai.RoleUser {
				return nil, unsupportedFeatureErrorf("image part in %q role content", role)
			}
			hasNonToolPart = true
			userMultimodalParts = append(userMultimodalParts, anyllm.ContentPart{
				Type:     contentTypeImageURL,
				ImageURL: imageURL,
			})
			continue
		}

		switch {
		case part.InlineData != nil:
			return nil, unsupportedFeatureErrorf("inlineData mime type %q", part.InlineData.MIMEType)
		case part.FileData != nil:
			return nil, unsupportedFeatureErrorf("fileData mime type %q", part.FileData.MIMEType)
		case part.ExecutableCode != nil:
			return nil, unsupportedFeatureError("executableCode part")
		case part.CodeExecutionResult != nil:
			return nil, unsupportedFeatureError("codeExecutionResult part")
		case part.ToolCall != nil:
			return nil, unsupportedFeatureError("toolCall part")
		case part.ToolResponse != nil:
			return nil, unsupportedFeatureError("toolResponse part")
		}
	}

	if hasToolReply && hasNonToolPart {
		return nil, unsupportedFeatureError("mixed function response parts with text/images/function calls in one content")
	}

	if hasToolReply {
		return messages, nil
	}

	isAssistantContent := role == genai.RoleModel || role == anyllm.RoleAssistant ||
		len(toolCalls) > 0 || reasoningText.Len() > 0
	if isAssistantContent {
		msg := anyllm.Message{Role: anyllm.RoleAssistant}
		if len(textParts) > 0 {
			msg.Content = strings.Join(textParts, "\n")
		}
		if len(toolCalls) > 0 {
			msg.ToolCalls = toolCalls
		}
		if reasoningText.Len() > 0 {
			msg.Reasoning = &anyllm.Reasoning{Content: reasoningText.String()}
		}
		// Extract the 3-operand condition into a named boolean (code-style rule).
		hasMessageContent := msg.Content != nil || len(msg.ToolCalls) > 0 || msg.Reasoning != nil
		if hasMessageContent {
			messages = append(messages, msg)
		}

		return messages, nil
	}

	if role == anyllm.RoleSystem {
		if len(textParts) > 0 {
			messages = append(messages, anyllm.Message{
				Role:    anyllm.RoleSystem,
				Content: strings.Join(textParts, "\n"),
			})
		}

		return messages, nil
	}

	switch {
	case len(userMultimodalParts) > 1 || (len(userMultimodalParts) == 1 && userMultimodalParts[0].Type == contentTypeImageURL):
		messages = append(messages, anyllm.Message{
			Role:    anyllm.RoleUser,
			Content: userMultimodalParts,
		})
	case len(textParts) > 0:
		messages = append(messages, anyllm.Message{
			Role:    anyllm.RoleUser,
			Content: strings.Join(textParts, "\n"),
		})
	}

	return messages, nil
}

func toolMessageFromFunctionResponse(
	functionResponse *genai.FunctionResponse,
) (anyllm.Message, error) {
	if functionResponse == nil {
		return anyllm.Message{}, newError("nil function response")
	}

	if functionResponse.WillContinue != nil {
		return anyllm.Message{}, unsupportedFeatureError("function response willContinue")
	}

	if functionResponse.Scheduling != "" &&
		functionResponse.Scheduling != genai.FunctionResponseSchedulingUnspecified {
		return anyllm.Message{}, unsupportedFeatureError("function response scheduling")
	}

	if len(functionResponse.Parts) > 0 {
		return anyllm.Message{}, unsupportedFeatureError("function response parts")
	}

	var (
		content    = ""
		toolCallID = functionResponse.ID
	)

	if functionResponse.Response != nil {
		payload, err := json.Marshal(functionResponse.Response)
		if err != nil {
			return anyllm.Message{}, wrapError("marshal function response", err)
		}

		content = string(payload)
	}

	if toolCallID == "" {
		return anyllm.Message{}, newError("function response missing tool call id")
	}

	return anyllm.Message{
		Role:       anyllm.RoleTool,
		ToolCallID: toolCallID,
		Content:    content,
	}, nil
}

func toolCallFromFunctionCall(
	functionCall *genai.FunctionCall,
	fallbackOrdinal int,
) (anyllm.ToolCall, error) {
	if functionCall == nil {
		return anyllm.ToolCall{}, newError("nil function call")
	}
	if len(functionCall.PartialArgs) > 0 {
		return anyllm.ToolCall{}, unsupportedFeatureError("function call partialArgs")
	}
	if functionCall.WillContinue != nil {
		return anyllm.ToolCall{}, unsupportedFeatureError("function call willContinue")
	}

	var (
		id   = functionCall.ID
		name = functionCall.Name
		args = "{}"
	)

	if id == "" {
		id = fmt.Sprintf("%s_%d", name, fallbackOrdinal)
	}

	if name == "" {
		return anyllm.ToolCall{}, newError("function call name is required")
	}

	if functionCall.Args != nil {
		rawArgs, err := json.Marshal(functionCall.Args)
		if err != nil {
			return anyllm.ToolCall{}, wrapErrorf("marshal function call args for %q", err, name)
		}

		args = string(rawArgs)
	}

	return anyllm.ToolCall{
		ID:   id,
		Type: toolTypeFunction,
		Function: anyllm.FunctionCall{
			Name:      name,
			Arguments: args,
		},
	}, nil
}

func contentToText(content *genai.Content) (string, error) {
	if content == nil {
		return "", nil
	}

	var (
		builder      strings.Builder
		hasPriorText bool
	)
	for _, part := range content.Parts {
		if part == nil {
			continue
		}
		variantCount := partVariantCount(part)
		if variantCount > 1 {
			return "", newErrorf("invalid part with %d variants set", variantCount)
		}
		if part.Text != "" {
			if hasPriorText {
				builder.WriteByte('\n')
			}
			builder.WriteString(part.Text)
			hasPriorText = true
			continue
		}
		if variantCount == 1 {
			return "", unsupportedFeatureError("non-text system instruction part")
		}
	}

	return builder.String(), nil
}

func imageURLFromInlineData(blob *genai.Blob) *anyllm.ImageURL {
	// Extract the 3-operand guard into a named boolean (code-style rule).
	isEmpty := blob == nil || blob.MIMEType == "" || len(blob.Data) == 0
	if isEmpty {
		return nil
	}

	if !strings.HasPrefix(blob.MIMEType, "image/") {
		return nil
	}

	encoded := base64.StdEncoding.EncodeToString(blob.Data)

	var b strings.Builder
	b.WriteString("data:")
	b.WriteString(blob.MIMEType)
	b.WriteString(";base64,")
	b.WriteString(encoded)

	return &anyllm.ImageURL{
		URL: b.String(),
	}
}

func imageURLFromFileData(fileData *genai.FileData) *anyllm.ImageURL {
	if fileData == nil || fileData.FileURI == "" {
		return nil
	}

	if fileData.MIMEType != "" && !strings.HasPrefix(fileData.MIMEType, "image/") {
		return nil
	}

	return &anyllm.ImageURL{URL: fileData.FileURI}
}

func partVariantCount(part *genai.Part) int {
	var count int
	for _, hasVariant := range []bool{
		part.Text != "",
		part.InlineData != nil,
		part.FileData != nil,
		part.FunctionCall != nil,
		part.FunctionResponse != nil,
		part.ExecutableCode != nil,
		part.CodeExecutionResult != nil,
		part.ToolCall != nil,
		part.ToolResponse != nil,
	} {
		if hasVariant {
			count++
		}
	}

	return count
}

func applyConfigToParams(
	params *anyllm.CompletionParams,
	cfg *genai.GenerateContentConfig,
) error {
	if cfg == nil {
		return nil
	}

	switch {
	case cfg.CandidateCount > 1:
		return newError("candidate count greater than 1 is not supported")
	case cfg.TopK != nil:
		return unsupportedFeatureError("topK")
	case cfg.HTTPOptions != nil:
		return unsupportedFeatureError("httpOptions")
	case cfg.RoutingConfig != nil:
		return unsupportedFeatureError("routingConfig")
	case cfg.ModelSelectionConfig != nil:
		return unsupportedFeatureError("modelSelectionConfig")
	case len(cfg.SafetySettings) > 0:
		return unsupportedFeatureError("safetySettings")
	case len(cfg.Tools) > 0:
		return unsupportedFeatureError("config.tools")
	case cfg.CachedContent != "":
		return unsupportedFeatureError("cachedContent")
	case len(cfg.ResponseModalities) > 0:
		return unsupportedFeatureError("responseModalities")
	case cfg.MediaResolution != "":
		return unsupportedFeatureError("mediaResolution")
	case cfg.SpeechConfig != nil:
		return unsupportedFeatureError("speechConfig")
	case cfg.AudioTimestamp:
		return unsupportedFeatureError("audioTimestamp")
	case cfg.ThinkingConfig != nil:
		return unsupportedFeatureError("thinkingConfig")
	case cfg.ImageConfig != nil:
		return unsupportedFeatureError("imageConfig")
	case cfg.EnableEnhancedCivicAnswers != nil:
		return unsupportedFeatureError("enableEnhancedCivicAnswers")
	case cfg.ModelArmorConfig != nil:
		return unsupportedFeatureError("modelArmorConfig")
	case cfg.ServiceTier != "":
		return unsupportedFeatureError("serviceTier")
	case len(cfg.Labels) > 0:
		return unsupportedFeatureError("labels")
	case cfg.ResponseLogprobs:
		return unsupportedFeatureError("responseLogprobs")
	case cfg.Logprobs != nil:
		return unsupportedFeatureError("logprobs")
	case cfg.PresencePenalty != nil:
		return unsupportedFeatureError("presencePenalty")
	case cfg.FrequencyPenalty != nil:
		return unsupportedFeatureError("frequencyPenalty")
	}

	if cfg.Temperature != nil {
		value := float64(*cfg.Temperature)
		params.Temperature = &value
	}
	if cfg.TopP != nil {
		value := float64(*cfg.TopP)
		params.TopP = &value
	}
	if cfg.MaxOutputTokens > 0 {
		value := int(cfg.MaxOutputTokens)
		params.MaxTokens = &value
	}
	if len(cfg.StopSequences) > 0 {
		params.Stop = append([]string(nil), cfg.StopSequences...)
	}
	if cfg.Seed != nil {
		value := int(*cfg.Seed)
		params.Seed = &value
	}

	if err := applyToolConfig(params, cfg); err != nil {
		return err
	}

	responseFormat, err := responseFormatFromConfig(cfg)
	if err != nil {
		return err
	}
	if responseFormat != nil {
		params.ResponseFormat = responseFormat
	}

	return nil
}

func applyToolConfig(
	params *anyllm.CompletionParams,
	cfg *genai.GenerateContentConfig,
) error {
	if cfg == nil || cfg.ToolConfig == nil {
		return nil
	}
	if cfg.ToolConfig.RetrievalConfig != nil {
		return unsupportedFeatureError("toolConfig.retrievalConfig")
	}
	if cfg.ToolConfig.IncludeServerSideToolInvocations != nil {
		return unsupportedFeatureError("toolConfig.includeServerSideToolInvocations")
	}
	if cfg.ToolConfig.FunctionCallingConfig == nil {
		return nil
	}

	functionConfig := cfg.ToolConfig.FunctionCallingConfig
	if functionConfig.StreamFunctionCallArguments != nil {
		return unsupportedFeatureError("toolConfig.functionCallingConfig.streamFunctionCallArguments")
	}

	switch functionConfig.Mode {
	case "", genai.FunctionCallingConfigModeUnspecified, genai.FunctionCallingConfigModeAuto:
		return nil
	case genai.FunctionCallingConfigModeNone:
		params.ToolChoice = toolChoiceNone
		return nil
	case genai.FunctionCallingConfigModeAny:
		return applyAnyToolMode(params, functionConfig)
	case genai.FunctionCallingConfigModeValidated:
		params.ToolChoice = toolChoiceAuto
		return nil
	default:
		return unsupportedFeatureErrorf("function calling mode %q", functionConfig.Mode)
	}
}

func applyAnyToolMode(
	params *anyllm.CompletionParams,
	functionConfig *genai.FunctionCallingConfig,
) error {
	if len(functionConfig.AllowedFunctionNames) == 1 {
		params.ToolChoice = anyllm.ToolChoice{
			Type: toolTypeFunction,
			Function: &anyllm.ToolChoiceFunction{
				Name: functionConfig.AllowedFunctionNames[0],
			},
		}
		return nil
	}

	params.ToolChoice = toolChoiceRequired
	return nil
}

func responseFormatFromConfig(
	cfg *genai.GenerateContentConfig,
) (*anyllm.ResponseFormat, error) {
	if cfg == nil {
		return nil, nil
	}
	if cfg.ResponseSchema != nil && cfg.ResponseJsonSchema != nil {
		return nil, newError("responseSchema and responseJsonSchema are mutually exclusive")
	}

	mimeType := cfg.ResponseMIMEType
	hasSchema := cfg.ResponseSchema != nil || cfg.ResponseJsonSchema != nil
	if hasSchema {
		if mimeType == "" {
			mimeType = mimeTypeApplicationJSON
		}
		if mimeType != mimeTypeApplicationJSON {
			return nil, newError("responseMimeType must be application/json when response schema is provided")
		}
	}

	if !hasSchema {
		if mimeType == "" || mimeType == mimeTypeTextPlain {
			return nil, nil
		}
		if mimeType != mimeTypeApplicationJSON {
			return nil, unsupportedFeatureErrorf("responseMimeType %q", mimeType)
		}
	}

	if cfg.ResponseSchema == nil && cfg.ResponseJsonSchema == nil {
		return &anyllm.ResponseFormat{Type: responseFormatJSONObject}, nil
	}

	schemaSource := any(cfg.ResponseSchema)
	if cfg.ResponseJsonSchema != nil {
		schemaSource = cfg.ResponseJsonSchema
	}

	schemaMap, err := normalizeSchema(schemaSource)
	if err != nil {
		return nil, wrapError("normalize response schema", err)
	}

	return &anyllm.ResponseFormat{
		Type: responseFormatJSONSchema,
		JSONSchema: &anyllm.JSONSchema{
			Name:   defaultSchemaName,
			Schema: schemaMap,
		},
	}, nil
}
