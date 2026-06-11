package adkanyllm

import (
	"errors"
	"testing"
)

func TestNewRequiresProvider(t *testing.T) {
	t.Parallel()

	_, err := New(nil)
	if err == nil {
		t.Fatal("expected error for nil provider")
	}
	var adapterErr *AdapterError
	if !errors.As(err, &adapterErr) {
		t.Fatalf("expected *AdapterError, got %T", err)
	}
}

func TestNewWithProviderAndModel(t *testing.T) {
	t.Parallel()

	m, err := New(
		&fakeProvider{},
		WithModel("gpt-4o-mini"),
		WithExtra(map[string]any{"foo": "bar"}),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m.Name() != "gpt-4o-mini" {
		t.Fatalf("Name()=%q expected gpt-4o-mini", m.Name())
	}
	if m.extra["foo"] != "bar" {
		t.Fatalf("expected extra to be cloned")
	}
}

func TestWithExtraNilClearsExtra(t *testing.T) {
	t.Parallel()

	m, err := New(
		&fakeProvider{},
		WithExtra(map[string]any{"foo": "bar"}),
		WithExtra(nil),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m.extra != nil {
		t.Fatalf("expected nil extra, got %#v", m.extra)
	}
}

func TestNewSkipsNilOption(t *testing.T) {
	t.Parallel()

	m, err := New(
		&fakeProvider{},
		nil,
		WithModel("gpt-test"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m.Name() != "gpt-test" {
		t.Fatalf("Name()=%q expected gpt-test", m.Name())
	}
}
