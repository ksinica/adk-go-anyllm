package adkanyllm

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/adk/v2/model"
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

	tools, err := convertTools(req.Tools)
	if err != nil {
		return anyllm.CompletionParams{}, err
	}

	params := anyllm.CompletionParams{
		Model:    modelName,
		Messages: messages,
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	// applyConfigToParams runs after tools are attached so that tool-choice
	// restrictions (e.g. allowedFunctionNames) can filter params.Tools.
	if err := applyConfigToParams(&params, req.Config); err != nil {
		return anyllm.CompletionParams{}, err
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

	// Seeding every Content before converting any of them guarantees a
	// synthetic id generated below for one Content never collides with an
	// explicit id used by another Content, regardless of which one appears
	// first in req.Contents.
	tracker := newToolCallIDTracker()
	for _, content := range req.Contents {
		if err := seedFunctionCallIDs(tracker, content); err != nil {
			return nil, err
		}
	}

	for _, content := range req.Contents {
		converted, err := convertContent(content, tracker)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted...)
	}

	return messages, nil
}

// seedFunctionCallIDs registers every explicit, non-empty
// genai.FunctionCall id in content with tracker.
func seedFunctionCallIDs(tracker *toolCallIDTracker, content *genai.Content) error {
	if content == nil {
		return nil
	}

	for _, part := range content.Parts {
		if part != nil && part.FunctionCall != nil {
			if err := tracker.seed(part.FunctionCall.ID); err != nil {
				return err
			}
		}
	}

	return nil
}

// toolCallIDTracker synthesizes request-wide unique ids for ID-less
// genai.FunctionCall parts, and tracks them so a later ID-less
// genai.FunctionResponse can resolve back to the exact call it answers.
//
// used holds every id already claimed anywhere in the request: explicit
// ids seeded up front by seedFunctionCallIDs before any Content is
// converted, plus every synthetic id allocate has already handed out. A
// synthetic id is checked against used and advanced until it collides with
// neither, so two ID-less calls to the same function name — even across
// different assistant turns in the same request — always get distinct
// ids.
//
// byName records the synthetic ids assigned to ID-less calls, in call
// order per function name, so a later ID-less genai.FunctionResponse for
// the same function resolves back to the exact id its call was given,
// instead of hard-failing for having no id of its own.
//
// A nil receiver behaves as an empty, unseeded tracker: allocate returns
// its first candidate id without checking for collisions, and resolve
// never finds a pending call. This lets callers that convert a single
// genai.Content in isolation (e.g. tests) pass nil.
type toolCallIDTracker struct {
	used   map[string]struct{}
	byName map[string][]string
}

func newToolCallIDTracker() *toolCallIDTracker {
	return &toolCallIDTracker{
		used:   make(map[string]struct{}),
		byName: make(map[string][]string),
	}
}

// seed registers an explicit, non-empty function-call id so no synthetic
// id allocated later collides with it. It returns a validation error when
// id was already seeded elsewhere in the request: two function calls
// sharing the same explicit id make the request's tool-call history
// ambiguous, so that must fail loudly rather than being silently accepted.
func (t *toolCallIDTracker) seed(id string) error {
	if t == nil || id == "" {
		return nil
	}

	if _, exists := t.used[id]; exists {
		return newErrorf("duplicate explicit function call id %q", id)
	}

	t.used[id] = struct{}{}
	return nil
}

// allocate returns a request-wide unique synthetic id for an ID-less call
// to name. startOrdinal is the call's 1-based position among the tool
// calls already converted in the same Content, used as the starting point
// for the search; it is advanced until the candidate collides with
// neither a seeded explicit id nor an earlier synthetic id. The id is
// recorded as used and as pending for name, so a later ID-less
// FunctionResponse for name can resolve back to it via resolve.
func (t *toolCallIDTracker) allocate(name string, startOrdinal int) string {
	if t == nil {
		return fmt.Sprintf("%s_%d", name, startOrdinal)
	}

	for ordinal := startOrdinal; ; ordinal++ {
		candidate := fmt.Sprintf("%s_%d", name, ordinal)
		if _, exists := t.used[candidate]; exists {
			continue
		}

		t.used[candidate] = struct{}{}
		t.byName[name] = append(t.byName[name], candidate)
		return candidate
	}
}

// add records the synthetic id assigned to an ID-less function call.
func (t *toolCallIDTracker) add(name, id string) {
	if t == nil {
		return
	}

	t.byName[name] = append(t.byName[name], id)
}

// resolve pops the oldest pending id recorded for name, in FIFO order, so
// that ID-less calls sharing a name are matched to their responses in the
// order they were made.
func (t *toolCallIDTracker) resolve(name string) (string, bool) {
	if t == nil {
		return "", false
	}

	ids := t.byName[name]
	if len(ids) == 0 {
		return "", false
	}

	t.byName[name] = ids[1:]
	return ids[0], true
}

func convertContent(content *genai.Content, pending *toolCallIDTracker) ([]anyllm.Message, error) {
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

		if err := rejectPartModifiers(part); err != nil {
			return nil, err
		}

		if variantCount := partVariantCount(part); variantCount > 1 {
			return nil, newErrorf("invalid part with %d variants set", variantCount)
		}

		if part.Thought {
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

		if part.FunctionResponse != nil {
			toolResponseMessage, err := toolMessageFromFunctionResponse(part.FunctionResponse, pending)
			if err != nil {
				return nil, err
			}
			messages = append(messages, toolResponseMessage)
			hasToolReply = true
			continue
		}

		if part.FunctionCall != nil {
			hasNonToolPart = true
			toolCall, err := toolCallFromFunctionCall(part.FunctionCall, pending, len(toolCalls)+1)
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

	// Only route through the multimodal array form when an image is present;
	// multiple pure-text parts still join into a single string, matching the
	// system/assistant handling above.
	hasImage := slices.ContainsFunc(userMultimodalParts, func(p anyllm.ContentPart) bool {
		return p.Type == contentTypeImageURL
	})

	switch {
	case hasImage:
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
	pending *toolCallIDTracker,
) (anyllm.Message, error) {
	if functionResponse == nil {
		return anyllm.Message{}, newError("nil function response")
	}

	if functionResponse.WillContinue != nil && *functionResponse.WillContinue {
		return anyllm.Message{}, unsupportedFeatureError("function response willContinue")
	}

	if functionResponse.Scheduling != "" &&
		functionResponse.Scheduling != genai.FunctionResponseSchedulingUnspecified {
		return anyllm.Message{}, unsupportedFeatureError("function response scheduling")
	}

	if len(functionResponse.Parts) > 0 {
		return anyllm.Message{}, unsupportedFeatureError("function response parts")
	}

	toolCallID := functionResponse.ID
	if toolCallID == "" {
		// Gemini-style history may omit ids on both the FunctionCall and its
		// matching FunctionResponse; resolve back to the synthetic id
		// toolCallFromFunctionCall assigned the pending call with this name,
		// in call order. Only fail when there is genuinely no matching call.
		resolved, ok := pending.resolve(functionResponse.Name)
		if !ok {
			return anyllm.Message{}, newError("function response missing tool call id and matches no pending id-less function call")
		}

		toolCallID = resolved
	}

	content := ""
	if functionResponse.Response != nil {
		payload, err := json.Marshal(functionResponse.Response)
		if err != nil {
			return anyllm.Message{}, wrapError("marshal function response", err)
		}

		content = string(payload)
	}

	return anyllm.Message{
		Role:       anyllm.RoleTool,
		ToolCallID: toolCallID,
		Content:    content,
	}, nil
}

// toolCallFromFunctionCall converts a genai.FunctionCall into an
// anyllm.ToolCall. When functionCall.ID is empty, tracker allocates a
// request-wide unique synthetic id for it (see toolCallIDTracker);
// fallbackOrdinal is that allocation's starting point, the call's 1-based
// position among the tool calls already converted in the same Content.
func toolCallFromFunctionCall(
	functionCall *genai.FunctionCall,
	tracker *toolCallIDTracker,
	fallbackOrdinal int,
) (anyllm.ToolCall, error) {
	if functionCall == nil {
		return anyllm.ToolCall{}, newError("nil function call")
	}
	if len(functionCall.PartialArgs) > 0 {
		return anyllm.ToolCall{}, unsupportedFeatureError("function call partialArgs")
	}
	if functionCall.WillContinue != nil && *functionCall.WillContinue {
		return anyllm.ToolCall{}, unsupportedFeatureError("function call willContinue")
	}

	name := functionCall.Name
	if name == "" {
		return anyllm.ToolCall{}, newError("function call name is required")
	}

	id := functionCall.ID
	if id == "" {
		id = tracker.allocate(name, fallbackOrdinal)
	}

	args := "{}"
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

		if err := rejectPartModifiers(part); err != nil {
			return "", err
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

// rejectPartModifiers rejects genai.Part modifier fields the adapter cannot
// translate. It is shared by convertContent and contentToText so that both
// regular content and system-instruction parts get the same validation.
func rejectPartModifiers(part *genai.Part) error {
	switch {
	case part.VideoMetadata != nil:
		return unsupportedFeatureError("videoMetadata part")
	case part.MediaResolution != nil:
		return unsupportedFeatureError("mediaResolution part")
	case len(part.PartMetadata) > 0:
		return unsupportedFeatureError("partMetadata")
	case len(part.ThoughtSignature) > 0:
		return unsupportedFeatureError("thoughtSignature")
	}

	return nil
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

	// cfg.Tools is intentionally not checked here: the ADK writes tools into
	// both cfg.Tools (genai format) and req.Tools (map format), and we handle
	// tools through req.Tools in convertTools(). It must not become a switch
	// case below — an empty matching case would stop the switch from
	// evaluating every later case, bypassing their validation entirely.
	switch {
	case cfg.CandidateCount > 1:
		return newError("candidate count greater than 1 is not supported")
	case cfg.CandidateCount < 0:
		return newError("candidate count must not be negative")
	case cfg.MaxOutputTokens < 0:
		return newError("max output tokens must not be negative")
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
	// ThinkingConfig is handled below in applyThinkingConfig.
	case cfg.ImageConfig != nil:
		return unsupportedFeatureError("imageConfig")
	case cfg.EnableEnhancedCivicAnswers != nil && *cfg.EnableEnhancedCivicAnswers:
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
		params.Stop = slices.Clone(cfg.StopSequences)
	}
	if cfg.Seed != nil {
		value := int(*cfg.Seed)
		params.Seed = &value
	}

	if err := applyThinkingConfig(params, cfg); err != nil {
		return err
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

// thinkingBudgetThresholds maps ReasoningEffort levels to their minimum token budgets.
// Used to pick the closest effort level when ThinkingBudget is set without a ThinkingLevel.
// Values mirror the AnyLLM Gemini provider's hardcoded budgets (1024/8192/24576).
var thinkingBudgetThresholds = []struct {
	minBudget int32
	effort    anyllm.ReasoningEffort
}{
	// Thresholds use midpoints between the Gemini provider's budget values
	// (Low=1024, Medium=8192, High=24576) for intuitive tier selection: the
	// Low/Medium midpoint is 4608, the Medium/High midpoint is 16384.
	// A budget of 10000 lands in Medium (4608–16383), a budget of 20000 in High (16384+).
	{minBudget: 16384, effort: anyllm.ReasoningEffortHigh},
	{minBudget: 4608, effort: anyllm.ReasoningEffortMedium},
	{minBudget: 1, effort: anyllm.ReasoningEffortLow},
}

// includeThoughtsFromConfig reports whether the request asked for thought
// summaries to be returned, mirroring genai's IncludeThoughts semantics:
// thoughts are only returned when explicitly requested, regardless of
// whether reasoning effort itself is enabled via ThinkingBudget or
// ThinkingLevel.
func includeThoughtsFromConfig(cfg *genai.GenerateContentConfig) bool {
	return cfg != nil && cfg.ThinkingConfig != nil && cfg.ThinkingConfig.IncludeThoughts
}

// applyThinkingConfig translates ADK's ThinkingConfig to AnyLLM's ReasoningEffort.
func applyThinkingConfig(
	params *anyllm.CompletionParams,
	cfg *genai.GenerateContentConfig,
) error {
	if cfg == nil || cfg.ThinkingConfig == nil {
		return nil
	}

	tc := cfg.ThinkingConfig

	// If IncludeThoughts is false and no other signal is set, treat as disabled.
	// ThinkingLevel zero value is empty string, not ThinkingLevelUnspecified.
	isDefaultLevel := tc.ThinkingLevel == "" || tc.ThinkingLevel == genai.ThinkingLevelUnspecified
	if !tc.IncludeThoughts && tc.ThinkingBudget == nil && isDefaultLevel {
		return nil
	}

	// Map by ThinkingLevel first (most explicit signal).
	if tc.ThinkingLevel != "" && tc.ThinkingLevel != genai.ThinkingLevelUnspecified {
		switch tc.ThinkingLevel {
		case genai.ThinkingLevelMinimal, genai.ThinkingLevelLow:
			params.ReasoningEffort = anyllm.ReasoningEffortLow
		case genai.ThinkingLevelMedium:
			params.ReasoningEffort = anyllm.ReasoningEffortMedium
		case genai.ThinkingLevelHigh:
			params.ReasoningEffort = anyllm.ReasoningEffortHigh
		default:
			return unsupportedFeatureErrorf("thinkingLevel %q", tc.ThinkingLevel)
		}

		return nil
	}

	// Fall back to ThinkingBudget if set without a level.
	if tc.ThinkingBudget != nil {
		budget := *tc.ThinkingBudget

		// A budget of exactly 0 is genai's explicit "disable thinking"
		// signal; AnyLLM can represent that directly. Negative budgets (e.g.
		// Gemini's "dynamic" -1) have no clear equivalent, so they are left
		// as a no-op rather than guessed at.
		switch {
		case budget == 0:
			params.ReasoningEffort = anyllm.ReasoningEffortNone
			return nil
		case budget < 0:
			return nil
		}

		for _, tier := range thinkingBudgetThresholds {
			if budget >= tier.minBudget {
				params.ReasoningEffort = tier.effort
				return nil
			}
		}

		params.ReasoningEffort = anyllm.ReasoningEffortLow
		return nil
	}

	// IncludeThoughts=true with no budget or level means model default.
	params.ReasoningEffort = anyllm.ReasoningEffortAuto
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

	if cfg.ToolConfig.IncludeServerSideToolInvocations != nil && *cfg.ToolConfig.IncludeServerSideToolInvocations {
		return unsupportedFeatureError("toolConfig.includeServerSideToolInvocations")
	}

	if cfg.ToolConfig.FunctionCallingConfig == nil {
		return nil
	}

	functionConfig := cfg.ToolConfig.FunctionCallingConfig
	if functionConfig.StreamFunctionCallArguments != nil && *functionConfig.StreamFunctionCallArguments {
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
		// AnyLLM has no strict/schema-validated tool-choice equivalent;
		// downgrading to AUTO would silently drop VALIDATED's
		// schema-conformance guarantee, so this must fail loudly instead.
		return unsupportedFeatureError("toolConfig.functionCallingConfig mode validated")
	default:
		return unsupportedFeatureErrorf("function calling mode %q", functionConfig.Mode)
	}
}

func applyAnyToolMode(
	params *anyllm.CompletionParams,
	functionConfig *genai.FunctionCallingConfig,
) error {
	switch len(functionConfig.AllowedFunctionNames) {
	case 0:
		if len(params.Tools) == 0 {
			return unsupportedFeatureError("toolConfig.functionCallingConfig mode any with no declared tools to require")
		}

		params.ToolChoice = toolChoiceRequired
		return nil
	case 1:
		name := functionConfig.AllowedFunctionNames[0]
		if !slices.ContainsFunc(params.Tools, func(tool anyllm.Tool) bool {
			return tool.Function.Name == name
		}) {
			return unsupportedFeatureError("toolConfig.functionCallingConfig.allowedFunctionNames matches none of the declared tools")
		}

		params.ToolChoice = anyllm.ToolChoice{
			Type: toolTypeFunction,
			Function: &anyllm.ToolChoiceFunction{
				Name: name,
			},
		}
		return nil
	default:
		return restrictToolsToAllowedNames(params, functionConfig.AllowedFunctionNames)
	}
}

// restrictToolsToAllowedNames honors a multi-name allowedFunctionNames list.
// AnyLLM's ToolChoice can only pin a single named function or an
// unrestricted "required"/"auto"/"none" — it has no way to express "any of
// these N named tools". Collapsing that to unrestricted "required" would
// silently broaden the permission the caller asked for, so instead this
// narrows params.Tools to the allowed subset and forces tool use; if none of
// the declared tools match, it fails loudly rather than guessing.
func restrictToolsToAllowedNames(params *anyllm.CompletionParams, allowedNames []string) error {
	if len(params.Tools) == 0 {
		return unsupportedFeatureError("toolConfig.functionCallingConfig.allowedFunctionNames with multiple names and no declared tools to restrict")
	}

	allowed := make(map[string]struct{}, len(allowedNames))
	for _, name := range allowedNames {
		allowed[name] = struct{}{}
	}

	filtered := slices.DeleteFunc(slices.Clone(params.Tools), func(tool anyllm.Tool) bool {
		_, ok := allowed[tool.Function.Name]
		return !ok
	})
	if len(filtered) == 0 {
		return unsupportedFeatureError("toolConfig.functionCallingConfig.allowedFunctionNames matches none of the declared tools")
	}

	params.Tools = filtered
	params.ToolChoice = toolChoiceRequired
	return nil
}

func responseFormatFromConfig(
	cfg *genai.GenerateContentConfig,
) (*anyllm.ResponseFormat, error) {
	if cfg == nil {
		return nil, nil
	}

	// cfg.ResponseJsonSchema arrives as any, so a typed-nil value (e.g. a nil
	// *jsonschema.Schema boxed into the interface) must be treated as absent
	// via isNilValue rather than a plain "!= nil" comparison, which would see
	// it as present and wrongly trip the mutual-exclusivity check below.
	hasJSONSchema := !isNilValue(cfg.ResponseJsonSchema)
	hasResponseSchema := cfg.ResponseSchema != nil

	if hasResponseSchema && hasJSONSchema {
		return nil, newError("responseSchema and responseJsonSchema are mutually exclusive")
	}

	mimeType := cfg.ResponseMIMEType
	hasSchema := hasResponseSchema || hasJSONSchema
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

	if !hasSchema {
		return &anyllm.ResponseFormat{Type: responseFormatJSONObject}, nil
	}

	var schemaSource any = cfg.ResponseSchema
	if hasJSONSchema {
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
