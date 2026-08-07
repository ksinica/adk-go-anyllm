package adkanyllm

import (
	"encoding/json"
	"maps"
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/genai"
)

func convertTools(tools map[string]any) ([]anyllm.Tool, error) {
	if len(tools) == 0 {
		return nil, nil
	}

	type functionDeclarer interface {
		Declaration() *genai.FunctionDeclaration
	}

	var (
		converted     = make([]anyllm.Tool, 0, len(tools))
		names         = slices.Sorted(maps.Keys(tools))
		resolvedNames = make(map[string]struct{}, len(tools))
	)

	for _, name := range names {
		var tool anyllm.Tool

		switch v := tools[name].(type) {
		case anyllm.Tool:
			tool = v
		case functionDeclarer:
			// A typed-nil functionDeclarer (e.g. a nil pointer implementing
			// the interface) is non-nil as an interface value, so calling
			// Declaration() on it would dereference a nil receiver. Guard
			// with isNilValue rather than a plain "== nil" comparison.
			if isNilValue(v) {
				return nil, newErrorf("nil tool declarer for %q", name)
			}

			decl := v.Declaration()
			if decl == nil {
				return nil, newErrorf("missing declaration for tool %q", name)
			}

			fn, err := functionFromDeclaration(name, decl)
			if err != nil {
				return nil, err
			}

			tool = anyllm.Tool{Type: toolTypeFunction, Function: fn}
		case map[string]any:
			fn, err := functionFromMap(name, v)
			if err != nil {
				return nil, err
			}

			tool = anyllm.Tool{Type: toolTypeFunction, Function: fn}
		default:
			return nil, newErrorf("unsupported tool definition type %T for %q", v, name)
		}

		// Two differently-keyed tool declarations can resolve to the same
		// Function.Name (e.g. an explicit override matching another tool's
		// default). AnyLLM has no way to disambiguate two same-named
		// function tools, so a provider would silently see only one of
		// them; reject the collision loudly instead.
		if _, dup := resolvedNames[tool.Function.Name]; dup {
			return nil, newErrorf("duplicate resolved tool name %q", tool.Function.Name)
		}
		resolvedNames[tool.Function.Name] = struct{}{}

		converted = append(converted, tool)
	}

	return converted, nil
}

func functionFromDeclaration(
	defaultName string,
	declaration *genai.FunctionDeclaration,
) (anyllm.Function, error) {
	name := declaration.Name
	if name == "" {
		name = defaultName
	}
	if name == "" {
		return anyllm.Function{}, newError("tool name is required")
	}

	// Behavior, Response, and ResponseJsonSchema have no AnyLLM tool
	// equivalent. BehaviorBlocking is the adapter's implicit default
	// (synchronous request/response tool calling), so it is accepted; any
	// other populated Behavior, or a populated Response/ResponseJsonSchema,
	// would otherwise be silently dropped, so reject it loudly instead.
	switch declaration.Behavior {
	case "", genai.BehaviorUnspecified, genai.BehaviorBlocking:
	default:
		return anyllm.Function{}, unsupportedFeatureErrorf("function declaration behavior %q", declaration.Behavior)
	}
	if declaration.Response != nil {
		return anyllm.Function{}, unsupportedFeatureError("function declaration response")
	}
	if !isNilValue(declaration.ResponseJsonSchema) {
		return anyllm.Function{}, unsupportedFeatureError("function declaration responseJsonSchema")
	}

	fn := anyllm.Function{Name: name}
	if declaration.Description != "" {
		fn.Description = declaration.Description
	}

	params, err := normalizeSchemaFromDeclaration(
		declaration.ParametersJsonSchema,
		declaration.Parameters,
	)
	if err != nil {
		return anyllm.Function{}, err
	}
	if params != nil {
		fn.Parameters = params
	}

	return fn, nil
}

func functionFromMap(
	defaultName string,
	toolDef map[string]any,
) (anyllm.Function, error) {
	name := defaultName
	if n, ok := toolDef["name"].(string); ok && n != "" {
		name = n
	}
	if name == "" {
		return anyllm.Function{}, newError("tool name is required")
	}

	fn := anyllm.Function{Name: name}
	if desc, ok := toolDef["description"].(string); ok && desc != "" {
		fn.Description = desc
	}
	if p, ok := toolDef["parameters"]; ok {
		params, err := normalizeSchema(p)
		if err != nil {
			return anyllm.Function{}, wrapErrorf("invalid parameters for %q", err, name)
		}
		if params != nil {
			fn.Parameters = params
		}
	}

	return fn, nil
}

func normalizeSchemaFromDeclaration(
	parametersJSONSchema any,
	parametersSchema *genai.Schema,
) (map[string]any, error) {
	// parametersJSONSchema arrives as any, so a typed-nil value (e.g. a nil
	// *jsonschema.Schema boxed into the interface) must be treated as absent
	// via isNilValue rather than a plain "!= nil" comparison, which would see
	// it as present and wrongly reject it as conflicting with parametersSchema.
	hasJSONSchema := !isNilValue(parametersJSONSchema)
	hasSchema := parametersSchema != nil

	if hasJSONSchema && hasSchema {
		return nil, newError("parameters and parametersJsonSchema are mutually exclusive")
	}

	switch {
	case hasJSONSchema:
		return normalizeSchema(parametersJSONSchema)
	case hasSchema:
		return sanitizeSchemaMap(genaiSchemaToMap(parametersSchema)), nil
	default:
		return nil, nil
	}
}

func normalizeSchema(raw any) (map[string]any, error) {
	switch typed := raw.(type) {
	case nil:
		return nil, nil
	case map[string]any:
		return sanitizeSchemaMap(typed), nil
	case json.RawMessage:
		return jsonRawMessageToMap(typed)
	case *genai.Schema:
		return sanitizeSchemaMap(genaiSchemaToMap(typed)), nil
	case *jsonschema.Schema:
		return jsonSchemaToMap(typed)
	default:
		return nil, newErrorf("unsupported schema type %T", raw)
	}
}

func jsonRawMessageToMap(raw json.RawMessage) (map[string]any, error) {
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, wrapError("invalid json schema", err)
	}

	// json.Unmarshal leaves out nil, without an error, for a raw "null"
	// schema; treating that as "no schema" would silently drop the
	// caller-provided parameters/response schema, so reject it loudly instead.
	if out == nil {
		return nil, newError("json schema must not be null")
	}

	return sanitizeSchemaMap(out), nil
}

func jsonSchemaToMap(schema *jsonschema.Schema) (map[string]any, error) {
	if schema == nil {
		return nil, nil
	}

	payload, err := json.Marshal(schema)
	if err != nil {
		return nil, wrapError("marshal jsonschema", err)
	}

	return jsonRawMessageToMap(payload)
}

func genaiSchemaToMap(schema *genai.Schema) map[string]any {
	if schema == nil {
		return nil
	}

	var (
		out            = map[string]any{}
		normalizedType string
	)
	if schema.Type != "" {
		normalizedType = strings.ToLower(string(schema.Type))
		out["type"] = normalizedType
	}
	if schema.Format != "" {
		out["format"] = schema.Format
	}
	if schema.Description != "" {
		out["description"] = schema.Description
	}
	if schema.Title != "" {
		out["title"] = schema.Title
	}
	if len(schema.Enum) > 0 {
		out["enum"] = slices.Clone(schema.Enum)
	}
	if schema.Default != nil {
		out["default"] = schema.Default
	}
	if schema.Example != nil {
		out["example"] = schema.Example
	}
	if schema.Pattern != "" {
		out["pattern"] = schema.Pattern
	}

	if schema.Items != nil {
		out["items"] = genaiSchemaToMap(schema.Items)
	}
	if len(schema.AnyOf) > 0 {
		anyOf := make([]map[string]any, 0, len(schema.AnyOf))
		for _, item := range schema.AnyOf {
			anyOf = append(anyOf, genaiSchemaToMap(item))
		}
		out["anyOf"] = anyOf
	}
	if len(schema.Properties) > 0 {
		properties := make(map[string]any, len(schema.Properties))
		for key, propertySchema := range schema.Properties {
			properties[key] = genaiSchemaToMap(propertySchema)
		}
		out["properties"] = properties
	}
	if len(schema.Required) > 0 {
		out["required"] = slices.Clone(schema.Required)
	}
	if schema.MinItems != nil {
		out["minItems"] = *schema.MinItems
	}
	if schema.MaxItems != nil {
		out["maxItems"] = *schema.MaxItems
	}
	if schema.MinLength != nil {
		out["minLength"] = *schema.MinLength
	}
	if schema.MaxLength != nil {
		out["maxLength"] = *schema.MaxLength
	}
	if schema.MinProperties != nil {
		out["minProperties"] = *schema.MinProperties
	}
	if schema.MaxProperties != nil {
		out["maxProperties"] = *schema.MaxProperties
	}
	if schema.Minimum != nil {
		out["minimum"] = *schema.Minimum
	}
	if schema.Maximum != nil {
		out["maximum"] = *schema.Maximum
	}

	if schema.Nullable != nil && *schema.Nullable {
		out = applySchemaNullable(out, normalizedType, len(schema.AnyOf) > 0, len(schema.Enum) > 0)
	}

	return out
}

// applySchemaNullable adjusts a fully-built non-null schema map so that a
// JSON null value also validates, mirroring genai's Nullable flag. Widening
// the type in place (e.g. {"type":["string","null"]}) is enough on its own
// only in the plain case: no other keyword constrains the instance's value
// regardless of its type. enum and anyOf are both keywords handled here that
// do: enum requires an exact match against its value list, and anyOf
// requires a match against one of its subschemas, so a widened type still
// leaves null failing either check. In that case the complete non-null
// schema is wrapped instead: {"anyOf": [<original>, {"type": "null"}]}.
func applySchemaNullable(out map[string]any, normalizedType string, hasAnyOf, hasValueConstraint bool) map[string]any {
	switch {
	case hasValueConstraint || hasAnyOf:
		return map[string]any{
			"anyOf": []map[string]any{maps.Clone(out), {"type": "null"}},
		}
	case normalizedType != "":
		out["type"] = []string{normalizedType, "null"}
		return out
	default:
		return out
	}
}

func sanitizeSchemaMap(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}

	output := make(map[string]any, len(input))
	for key, value := range input {
		if key == "propertyOrdering" {
			continue
		}
		output[key] = sanitizeSchemaValue(value)
	}

	return output
}

func sanitizeSchemaValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		return sanitizeSchemaMap(typed)
	case []any:
		out := make([]any, len(typed))
		for idx, item := range typed {
			out[idx] = sanitizeSchemaValue(item)
		}
		return out
	case []map[string]any:
		out := make([]map[string]any, len(typed))
		for idx, item := range typed {
			out[idx] = sanitizeSchemaMap(item)
		}
		return out
	default:
		return value
	}
}
