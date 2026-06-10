package adkanyllm

import (
	"encoding/json"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	anyllm "github.com/mozilla-ai/any-llm-go"
	"google.golang.org/genai"
)

type fakeTool struct {
	decl *genai.FunctionDeclaration
}

func (f fakeTool) Declaration() *genai.FunctionDeclaration {
	return f.decl
}

func TestConvertToolsFromDeclaration(t *testing.T) {
	t.Parallel()

	tools, err := convertTools(map[string]any{
		"weather": fakeTool{
			decl: &genai.FunctionDeclaration{
				Name:        "get_weather",
				Description: "Get weather",
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"city": {Type: genai.TypeString},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if tools[0].Function.Name != "get_weather" {
		t.Fatalf("name=%q expected get_weather", tools[0].Function.Name)
	}
}

func TestConvertToolsDeterministicOrder(t *testing.T) {
	t.Parallel()

	tools, err := convertTools(map[string]any{
		"z_fn": map[string]any{"name": "z"},
		"a_fn": map[string]any{"name": "a"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}
	if tools[0].Function.Name != "a" || tools[1].Function.Name != "z" {
		t.Fatalf("unexpected tool order: %#v", tools)
	}
}

func TestConvertToolsAcceptsAnyLLMTool(t *testing.T) {
	t.Parallel()

	input := anyllm.Tool{
		Type: "function",
		Function: anyllm.Function{
			Name: "ping",
		},
	}
	tools, err := convertTools(map[string]any{"ping": input})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) != 1 || tools[0].Function.Name != "ping" {
		t.Fatalf("unexpected tools: %#v", tools)
	}
}

func TestConvertToolsMissingDeclaration(t *testing.T) {
	t.Parallel()

	_, err := convertTools(map[string]any{
		"weather": fakeTool{},
	})
	if err == nil {
		t.Fatal("expected missing declaration error")
	}
}

func TestNormalizeSchemaFromMap(t *testing.T) {
	t.Parallel()

	got, err := normalizeSchema(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{"type": "string"},
		},
		"propertyOrdering": []string{"city"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := got["propertyOrdering"]; ok {
		t.Fatal("expected propertyOrdering to be stripped")
	}
}

func TestNormalizeSchemaFromJSONSchema(t *testing.T) {
	t.Parallel()

	schema, err := jsonschema.For[struct {
		City string `json:"city"`
	}](nil)
	if err != nil {
		t.Fatalf("jsonschema.For failed: %v", err)
	}

	got, err := normalizeSchema(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	properties, ok := got["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties=%#v expected object properties", got["properties"])
	}
	if _, ok := properties["city"]; !ok {
		t.Fatalf("properties=%#v expected city field", properties)
	}
}

func TestNormalizeSchemaFromFunctionDeclaration(t *testing.T) {
	t.Parallel()

	schema, err := jsonschema.For[struct {
		City string `json:"city"`
	}](nil)
	if err != nil {
		t.Fatalf("jsonschema.For failed: %v", err)
	}

	fn, err := functionFromDeclaration("weather", &genai.FunctionDeclaration{
		Name:                 "get_weather",
		ParametersJsonSchema: schema,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if fn.Parameters == nil {
		t.Fatal("expected parameters schema")
	}
	if fn.Parameters["type"] != "object" {
		t.Fatalf("type=%#v expected object", fn.Parameters["type"])
	}
}

func TestNormalizeSchemaUnsupportedType(t *testing.T) {
	t.Parallel()

	_, err := normalizeSchema(123)
	if err == nil {
		t.Fatal("expected unsupported schema type error")
	}
}

func TestNormalizeSchemaFromRawMessage(t *testing.T) {
	t.Parallel()

	raw := json.RawMessage(`{"type":"object"}`)
	got, err := normalizeSchema(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got["type"] != "object" {
		t.Fatalf("type=%#v expected object", got["type"])
	}
}

func TestConvertToolsUnsupportedType(t *testing.T) {
	t.Parallel()

	_, err := convertTools(map[string]any{"bad": 123})
	if err == nil {
		t.Fatal("expected unsupported tool type error")
	}
}

func TestGenaiSchemaToMapNullable(t *testing.T) {
	t.Parallel()

	nullable := true
	got := genaiSchemaToMap(&genai.Schema{
		Type:     genai.TypeString,
		Nullable: &nullable,
	})
	types, ok := got["type"].([]string)
	if !ok || len(types) != 2 || types[1] != "null" {
		t.Fatalf("unexpected nullable type: %#v", got["type"])
	}
}

func TestGenaiSchemaToMapFull(t *testing.T) {
	t.Parallel()

	minItems := int64(1)
	maxItems := int64(10)
	minLength := int64(2)
	maxLength := int64(100)
	minProps := int64(1)
	maxProps := int64(20)
	minVal := 0.0
	maxVal := 100.0

	got := genaiSchemaToMap(&genai.Schema{
		Type:          genai.TypeObject,
		Format:        "date",
		Description:   "A date",
		Title:         "Date",
		Enum:          []string{"a", "b"},
		Pattern:       "\\d{4}-\\d{2}-\\d{2}",
		Items:         &genai.Schema{Type: genai.TypeString},
		Properties:    map[string]*genai.Schema{"day": {Type: genai.TypeString}},
		Required:      []string{"day"},
		MinItems:      &minItems,
		MaxItems:      &maxItems,
		MinLength:     &minLength,
		MaxLength:     &maxLength,
		MinProperties: &minProps,
		MaxProperties: &maxProps,
		Minimum:       &minVal,
		Maximum:       &maxVal,
	})
	if got["type"] != "object" {
		t.Fatalf("type=%#v", got["type"])
	}
	if got["format"] != "date" {
		t.Fatalf("format=%#v", got["format"])
	}
	if got["title"] != "Date" {
		t.Fatalf("title=%#v", got["title"])
	}
	if got["pattern"] != "\\d{4}-\\d{2}-\\d{2}" {
		t.Fatalf("pattern=%#v", got["pattern"])
	}
	items, ok := got["items"].(map[string]any)
	if !ok || items["type"] != "string" {
		t.Fatalf("items=%#v", got["items"])
	}
	props, ok := got["properties"].(map[string]any)
	if !ok || props["day"] == nil {
		t.Fatalf("properties=%#v", got["properties"])
	}
	if len(got["enum"].([]string)) != 2 {
		t.Fatalf("enum=%#v", got["enum"])
	}
	if v, ok := got["minItems"].(int64); !ok || v != 1 {
		t.Fatalf("minItems=%#v", got["minItems"])
	}
	if v, ok := got["maxItems"].(int64); !ok || v != 10 {
		t.Fatalf("maxItems=%#v", got["maxItems"])
	}
	if v, ok := got["minLength"].(int64); !ok || v != 2 {
		t.Fatalf("minLength=%#v", got["minLength"])
	}
	if v, ok := got["maxLength"].(int64); !ok || v != 100 {
		t.Fatalf("maxLength=%#v", got["maxLength"])
	}
}

func TestSanitizeSchemaValueRecursive(t *testing.T) {
	t.Parallel()

	input := map[string]any{
		"items": []any{
			map[string]any{"type": "string"},
			"plain",
		},
		"nested": map[string]any{
			"value": 42,
		},
	}
	got := sanitizeSchemaMap(input)
	items, ok := got["items"].([]any)
	if !ok || len(items) != 2 {
		t.Fatalf("items=%#v", got["items"])
	}
	nested, ok := got["nested"].(map[string]any)
	if !ok || nested["value"] != 42 {
		t.Fatalf("nested=%#v", got["nested"])
	}
}

func TestFunctionFromMap(t *testing.T) {
	t.Parallel()

	// Minimum.
	fn, err := functionFromMap("ping", map[string]any{})
	if err != nil || fn.Name != "ping" {
		t.Fatalf("unexpected function: %#v err=%v", fn, err)
	}

	// With description and parameters.
	fn, err = functionFromMap("weather", map[string]any{
		"name":        "get_weather",
		"description": "Get weather data",
		"parameters": map[string]any{
			"type":       "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if fn.Name != "get_weather" || fn.Description != "Get weather data" || fn.Parameters == nil {
		t.Fatalf("unexpected function: %#v", fn)
	}

	// name taken from key when map element name is empty.
	fn, err = functionFromMap("fallback", map[string]any{"name": ""})
	if err != nil || fn.Name != "fallback" {
		t.Fatalf("expected name fallback, got %q err=%v", fn.Name, err)
	}

	// Missing name entirely.
	_, err = functionFromMap("", map[string]any{})
	if err == nil {
		t.Fatal("expected error for empty name")
	}
}

func TestNormalizeSchemaFromGenaiSchema(t *testing.T) {
	t.Parallel()

	got, err := normalizeSchema(&genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"city": {Type: genai.TypeString, Description: "City name"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got["type"] != "object" {
		t.Fatalf("type=%#v", got["type"])
	}
	props, ok := got["properties"].(map[string]any)
	if !ok || props["city"] == nil {
		t.Fatalf("properties=%#v", got["properties"])
	}
}

func TestNormalizeSchemaFromDeclarationWithGenaiSchema(t *testing.T) {
	t.Parallel()

	fn, err := functionFromDeclaration("weather", &genai.FunctionDeclaration{
		Name: "get_weather",
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"city": {Type: genai.TypeString},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if fn.Parameters == nil || fn.Parameters["type"] != "object" {
		t.Fatalf("expected type=object, got %#v", fn.Parameters)
	}
}

func TestJsonSchemaToMapNil(t *testing.T) {
	t.Parallel()

	got, err := jsonSchemaToMap(nil)
	if err != nil || got != nil {
		t.Fatalf("expected nil, got %#v err=%v", got, err)
	}
}

func TestSanitizeSchemaMapNil(t *testing.T) {
	t.Parallel()

	if got := sanitizeSchemaMap(nil); got != nil {
		t.Fatal("expected nil")
	}
}

func TestConvertToolsFunctionFromMapEmptyName(t *testing.T) {
	t.Parallel()

	_, err := convertTools(map[string]any{
		"": map[string]any{},
	})
	if err == nil {
		t.Fatal("expected error for empty tool name")
	}
}

func TestFunctionFromDeclarationEmptyName(t *testing.T) {
	t.Parallel()

	_, err := functionFromDeclaration("", &genai.FunctionDeclaration{})
	if err == nil {
		t.Fatal("expected error for empty name")
	}
}

func TestNormalizeSchemaFromDeclarationNil(t *testing.T) {
	t.Parallel()

	got, err := normalizeSchemaFromDeclaration(nil, nil)
	if err != nil || got != nil {
		t.Fatalf("expected nil, got %#v err=%v", got, err)
	}
}

func TestNormalizeSchemaFromDeclarationWithGenaiSchemaNullable(t *testing.T) {
	t.Parallel()

	got, err := normalizeSchemaFromDeclaration(nil, &genai.Schema{Type: genai.TypeObject})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got["type"] != "object" {
		t.Fatalf("type=%#v", got["type"])
	}
}

func TestGenaiSchemaToMapDefaultAndExample(t *testing.T) {
	t.Parallel()

	got := genaiSchemaToMap(&genai.Schema{
		Default: "fallback",
		Example: "sample",
	})
	if got["default"] != "fallback" || got["example"] != "sample" {
		t.Fatalf("default/example=%#v", got)
	}
}

func TestJsonRawMessageToMapInvalid(t *testing.T) {
	t.Parallel()

	_, err := jsonRawMessageToMap(json.RawMessage(`{invalid}`))
	if err == nil {
		t.Fatal("expected invalid json error")
	}
}
