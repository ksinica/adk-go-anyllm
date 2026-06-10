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
		converted = make([]anyllm.Tool, 0, len(tools))
		names     = slices.Sorted(maps.Keys(tools))
	)

	for _, name := range names {
		switch v := tools[name].(type) {
		case anyllm.Tool:
			converted = append(converted, v)
		case functionDeclarer:
			decl := v.Declaration()
			if decl == nil {
				return nil, newErrorf("missing declaration for tool %q", name)
			}

			fn, err := functionFromDeclaration(name, decl)
			if err != nil {
				return nil, err
			}

			converted = append(converted, anyllm.Tool{
				Type:     toolTypeFunction,
				Function: fn,
			})
		case map[string]any:
			fn, err := functionFromMap(name, v)
			if err != nil {
				return nil, err
			}

			converted = append(converted, anyllm.Tool{
				Type:     toolTypeFunction,
				Function: fn,
			})
		default:
			return nil, newErrorf("unsupported tool definition type %T for %q", v, name)
		}
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
	switch {
	case parametersJSONSchema != nil:
		return normalizeSchema(parametersJSONSchema)
	case parametersSchema != nil:
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
		out["enum"] = append([]string(nil), schema.Enum...)
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
		out["required"] = append([]string(nil), schema.Required...)
	}
	if schema.Nullable != nil && *schema.Nullable {
		types := []string{"null"}
		if normalizedType != "" {
			types = []string{normalizedType, "null"}
		}
		out["type"] = types
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

	return out
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
