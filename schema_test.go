package adkanyllm

import (
	"encoding/json"
	"errors"
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

// TestConvertToolsRejectsDuplicateResolvedName verifies the item-3 fix: two
// differently-keyed tool declarations that resolve to the same
// Function.Name fail loudly instead of silently producing two tool
// definitions AnyLLM cannot tell apart.
func TestConvertToolsRejectsDuplicateResolvedName(t *testing.T) {
	t.Parallel()

	_, err := convertTools(map[string]any{
		"a_weather": map[string]any{"name": "get_weather"},
		"b_weather": map[string]any{"name": "get_weather"},
	})
	if err == nil {
		t.Fatal("expected error for duplicate resolved tool name")
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

func TestGenaiSchemaToMapNullableUntyped(t *testing.T) {
	t.Parallel()

	nullable := true
	got := genaiSchemaToMap(&genai.Schema{
		Nullable: &nullable,
	})
	if _, ok := got["type"]; ok {
		t.Fatalf("expected no synthesized type for untyped nullable schema, got %#v", got["type"])
	}
}

// TestGenaiSchemaToMapNullableEnumWrapsAnyOf verifies that a nullable schema
// with sibling value constraints (enum) is wrapped as {"anyOf":[<non-null
// schema>, {"type":"null"}]} rather than merely widening "type" in place.
// Widening alone would produce {"type":["string","null"],"enum":["a"]},
// where null still fails the enum check because enum applies regardless of
// the instance's type.
func TestGenaiSchemaToMapNullableEnumWrapsAnyOf(t *testing.T) {
	t.Parallel()

	nullable := true
	got := genaiSchemaToMap(&genai.Schema{
		Type:     genai.TypeString,
		Enum:     []string{"a"},
		Nullable: &nullable,
	})
	if _, ok := got["type"]; ok {
		t.Fatalf("expected no top-level type when wrapped in anyOf, got %#v", got["type"])
	}
	anyOf, ok := got["anyOf"].([]map[string]any)
	if !ok || len(anyOf) != 2 {
		t.Fatalf("expected anyOf with 2 branches, got %#v", got["anyOf"])
	}
	if anyOf[0]["type"] != "string" {
		t.Fatalf("expected non-null branch to keep type=string, got %#v", anyOf[0])
	}
	enum, ok := anyOf[0]["enum"].([]string)
	if !ok || len(enum) != 1 || enum[0] != "a" {
		t.Fatalf("expected non-null branch to keep enum, got %#v", anyOf[0]["enum"])
	}
	if anyOf[1]["type"] != "null" {
		t.Fatalf("expected null branch, got %#v", anyOf[1])
	}
}

// TestGenaiSchemaToMapNullablePlainWidensTypeInPlace pins the simple case
// (nullable with no sibling value constraints) to the minimal-churn
// representation: widening "type" in place rather than wrapping in anyOf.
func TestGenaiSchemaToMapNullablePlainWidensTypeInPlace(t *testing.T) {
	t.Parallel()

	nullable := true
	got := genaiSchemaToMap(&genai.Schema{
		Type:     genai.TypeString,
		Nullable: &nullable,
	})
	if _, ok := got["anyOf"]; ok {
		t.Fatalf("expected no anyOf wrapping for a plain nullable schema, got %#v", got["anyOf"])
	}
	types, ok := got["type"].([]string)
	if !ok || len(types) != 2 || types[0] != "string" || types[1] != "null" {
		t.Fatalf("unexpected nullable type: %#v", got["type"])
	}
}

// TestGenaiSchemaToMapNullableAnyOf verifies that a nullable schema whose
// only constraint is an anyOf is wrapped as {"anyOf":[<original schema>,
// {"type":"null"}]}, the same general representation used for enum: the
// complete original (non-null) schema becomes one branch, sitting alongside
// a null branch, rather than merely appending a null branch into the
// original anyOf list in place.
func TestGenaiSchemaToMapNullableAnyOf(t *testing.T) {
	t.Parallel()

	nullable := true
	got := genaiSchemaToMap(&genai.Schema{
		Nullable: &nullable,
		AnyOf: []*genai.Schema{
			{Type: genai.TypeString},
			{Type: genai.TypeInteger},
		},
	})
	if _, ok := got["type"]; ok {
		t.Fatalf("expected no top-level type for anyOf nullable schema, got %#v", got["type"])
	}

	anyOf, ok := got["anyOf"].([]map[string]any)
	if !ok || len(anyOf) != 2 {
		t.Fatalf("expected outer anyOf with 2 branches, got %#v", got["anyOf"])
	}
	if anyOf[1]["type"] != "null" {
		t.Fatalf("expected second branch to be null, got %#v", anyOf[1])
	}

	original, ok := anyOf[0]["anyOf"].([]map[string]any)
	if !ok || len(original) != 2 {
		t.Fatalf("expected first branch to keep the original anyOf, got %#v", anyOf[0])
	}
	if original[0]["type"] != "string" || original[1]["type"] != "integer" {
		t.Fatalf("unexpected original anyOf branches: %#v", original)
	}
}

// TestGenaiSchemaToMapNullableUntypedEnumWrapsAnyOf verifies gap (a): an
// UNTYPED schema with enum must also validate null, by wrapping the complete
// original schema in an outer anyOf alongside a null branch.
func TestGenaiSchemaToMapNullableUntypedEnumWrapsAnyOf(t *testing.T) {
	t.Parallel()

	nullable := true
	got := genaiSchemaToMap(&genai.Schema{
		Enum:     []string{"a", "b"},
		Nullable: &nullable,
	})
	if _, ok := got["type"]; ok {
		t.Fatalf("expected no top-level type, got %#v", got["type"])
	}

	anyOf, ok := got["anyOf"].([]map[string]any)
	if !ok || len(anyOf) != 2 {
		t.Fatalf("expected anyOf with 2 branches, got %#v", got["anyOf"])
	}
	enum, ok := anyOf[0]["enum"].([]string)
	if !ok || len(enum) != 2 {
		t.Fatalf("expected first branch to keep enum, got %#v", anyOf[0])
	}
	if anyOf[1]["type"] != "null" {
		t.Fatalf("expected second branch to be null, got %#v", anyOf[1])
	}
}

// TestGenaiSchemaToMapNullableTypedAnyOfWrapsAnyOf verifies gap (b): a TYPED
// schema that also has anyOf must wrap the complete schema (type and anyOf
// together) rather than merely widening "type" in place, which would still
// leave null failing the anyOf branch match.
func TestGenaiSchemaToMapNullableTypedAnyOfWrapsAnyOf(t *testing.T) {
	t.Parallel()

	nullable := true
	got := genaiSchemaToMap(&genai.Schema{
		Type:     genai.TypeString,
		Nullable: &nullable,
		AnyOf: []*genai.Schema{
			{Type: genai.TypeString},
			{Type: genai.TypeInteger},
		},
	})
	if _, ok := got["type"]; ok {
		t.Fatalf("expected no top-level type when wrapped in anyOf, got %#v", got["type"])
	}

	anyOf, ok := got["anyOf"].([]map[string]any)
	if !ok || len(anyOf) != 2 {
		t.Fatalf("expected outer anyOf with 2 branches, got %#v", got["anyOf"])
	}
	if anyOf[0]["type"] != "string" {
		t.Fatalf("expected first branch to keep type=string, got %#v", anyOf[0])
	}
	if _, ok := anyOf[0]["anyOf"]; !ok {
		t.Fatalf("expected first branch to keep the original anyOf, got %#v", anyOf[0])
	}
	if anyOf[1]["type"] != "null" {
		t.Fatalf("expected second branch to be null, got %#v", anyOf[1])
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

func TestNormalizeSchemaFromDeclarationBothSet(t *testing.T) {
	t.Parallel()

	_, err := normalizeSchemaFromDeclaration(
		map[string]any{"type": "object"},
		&genai.Schema{Type: genai.TypeObject},
	)
	if err == nil {
		t.Fatal("expected error when both parameters and parametersJsonSchema are set")
	}
}

func TestFunctionFromDeclarationRejectsBothParameterSchemas(t *testing.T) {
	t.Parallel()

	_, err := functionFromDeclaration("weather", &genai.FunctionDeclaration{
		Name:                 "get_weather",
		Parameters:           &genai.Schema{Type: genai.TypeObject},
		ParametersJsonSchema: map[string]any{"type": "object"},
	})
	if err == nil {
		t.Fatal("expected error when both parameters and parametersJsonSchema are set")
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

// TestNormalizeSchemaFromDeclarationTypedNilJSONSchemaIgnored verifies that a
// typed-nil ParametersJsonSchema (e.g. a nil *jsonschema.Schema boxed into
// the any parameter) is treated as absent rather than "set", so it does not
// wrongly trip the mutually-exclusive check against a real Parameters schema.
func TestNormalizeSchemaFromDeclarationTypedNilJSONSchemaIgnored(t *testing.T) {
	t.Parallel()

	var typedNilJSONSchema *jsonschema.Schema

	got, err := normalizeSchemaFromDeclaration(typedNilJSONSchema, &genai.Schema{
		Type: genai.TypeObject,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got["type"] != "object" {
		t.Fatalf("type=%#v expected object", got["type"])
	}
}

// TestJsonRawMessageToMapRejectsNull verifies item 7: a raw JSON schema of
// literal "null" unmarshals to a nil map without an error, which would
// otherwise be silently treated as "no schema" and drop the caller-provided
// parameters/response schema. It must be rejected instead.
func TestJsonRawMessageToMapRejectsNull(t *testing.T) {
	t.Parallel()

	_, err := jsonRawMessageToMap(json.RawMessage("null"))
	if err == nil {
		t.Fatal("expected error for null json schema")
	}
}

// TestNormalizeSchemaRejectsRawNull pins the same behavior through the
// normalizeSchema entry point used for ParametersJsonSchema/response schemas.
func TestNormalizeSchemaRejectsRawNull(t *testing.T) {
	t.Parallel()

	_, err := normalizeSchema(json.RawMessage("null"))
	if err == nil {
		t.Fatal("expected error for null json schema")
	}
}

// TestFunctionFromDeclarationRejectsNonBlockingBehavior verifies item 5: a
// populated, non-default Behavior (NON_BLOCKING) has no AnyLLM tool
// equivalent and must be rejected rather than silently dropped.
func TestFunctionFromDeclarationRejectsNonBlockingBehavior(t *testing.T) {
	t.Parallel()

	_, err := functionFromDeclaration("weather", &genai.FunctionDeclaration{
		Name:     "get_weather",
		Behavior: genai.BehaviorNonBlocking,
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

// TestFunctionFromDeclarationAcceptsBlockingBehavior verifies that
// BehaviorBlocking, the adapter's implicit default (synchronous
// request/response tool calling), is accepted rather than rejected.
func TestFunctionFromDeclarationAcceptsBlockingBehavior(t *testing.T) {
	t.Parallel()

	_, err := functionFromDeclaration("weather", &genai.FunctionDeclaration{
		Name:     "get_weather",
		Behavior: genai.BehaviorBlocking,
	})
	if err != nil {
		t.Fatalf("unexpected error for BehaviorBlocking: %v", err)
	}
}

// TestFunctionFromDeclarationRejectsResponse verifies item 5: a populated
// Response schema has no AnyLLM tool equivalent and must be rejected rather
// than silently dropped.
func TestFunctionFromDeclarationRejectsResponse(t *testing.T) {
	t.Parallel()

	_, err := functionFromDeclaration("weather", &genai.FunctionDeclaration{
		Name:     "get_weather",
		Response: &genai.Schema{Type: genai.TypeObject},
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

// TestFunctionFromDeclarationRejectsResponseJsonSchema mirrors
// TestFunctionFromDeclarationRejectsResponse for ResponseJsonSchema, and
// uses isNilValue rather than a plain "!= nil" comparison so a typed-nil
// value is correctly treated as absent.
func TestFunctionFromDeclarationRejectsResponseJsonSchema(t *testing.T) {
	t.Parallel()

	schema, err := jsonschema.For[struct {
		Result string `json:"result"`
	}](nil)
	if err != nil {
		t.Fatalf("jsonschema.For failed: %v", err)
	}

	_, err = functionFromDeclaration("weather", &genai.FunctionDeclaration{
		Name:               "get_weather",
		ResponseJsonSchema: schema,
	})
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatalf("expected ErrUnsupportedFeature, got %v", err)
	}
}

// TestFunctionFromDeclarationTypedNilResponseJsonSchemaIgnored verifies that
// a typed-nil ResponseJsonSchema (e.g. a nil *jsonschema.Schema boxed into
// the any field) is treated as absent, not populated.
func TestFunctionFromDeclarationTypedNilResponseJsonSchemaIgnored(t *testing.T) {
	t.Parallel()

	var typedNilJSONSchema *jsonschema.Schema

	_, err := functionFromDeclaration("weather", &genai.FunctionDeclaration{
		Name:               "get_weather",
		ResponseJsonSchema: typedNilJSONSchema,
	})
	if err != nil {
		t.Fatalf("unexpected error for typed-nil ResponseJsonSchema: %v", err)
	}
}

// nilDeclarerTool implements functionDeclarer with a pointer receiver whose
// method dereferences the receiver, so a typed-nil value panics if called
// without a nil guard.
type nilDeclarerTool struct {
	decl *genai.FunctionDeclaration
}

func (t *nilDeclarerTool) Declaration() *genai.FunctionDeclaration {
	return t.decl
}

// TestConvertToolsTypedNilDeclarerReturnsErrorInsteadOfPanicking verifies
// item 9: a typed-nil functionDeclarer (non-nil as an interface value, since
// it wraps a nil *nilDeclarerTool) must be rejected with a validation error
// rather than panicking when Declaration() dereferences the nil receiver.
func TestConvertToolsTypedNilDeclarerReturnsErrorInsteadOfPanicking(t *testing.T) {
	t.Parallel()

	var nilTool *nilDeclarerTool

	_, err := convertTools(map[string]any{"weather": nilTool})
	if err == nil {
		t.Fatal("expected error for typed-nil tool declarer")
	}
}
