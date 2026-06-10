package adkanyllm

import (
	"errors"
	"testing"
)

func TestAdapterErrorStringIncludesPackagePrefix(t *testing.T) {
	t.Parallel()

	err := newError("provider is required")
	if got := err.Error(); got != "adkanyllm: provider is required" {
		t.Fatalf("Error()=%q expected adkanyllm: provider is required", got)
	}
}

func TestWrapErrorUnwrapsCause(t *testing.T) {
	t.Parallel()

	var (
		cause  = errors.New("provider failed")
		err    = wrapError("provider completion", cause)
		pkgErr *AdapterError
	)
	if !errors.As(err, &pkgErr) {
		t.Fatal("expected *AdapterError")
	}
	if !errors.Is(err, cause) {
		t.Fatal("expected wrapped cause")
	}
}

func TestUnsupportedFeatureErrorIsDetectable(t *testing.T) {
	t.Parallel()

	err := unsupportedFeatureError("topK")

	var featureErr *UnsupportedFeatureError
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatal("expected ErrUnsupportedFeature")
	}
	if !errors.As(err, &featureErr) {
		t.Fatal("expected *UnsupportedFeatureError")
	}
	if featureErr.Feature != "topK" {
		t.Fatalf("Feature=%q expected topK", featureErr.Feature)
	}
	if got := err.Error(); got != "adkanyllm: unsupported feature: topK" {
		t.Fatalf("Error()=%q", got)
	}
}

func TestUnsupportedFeatureErrorfIsDetectable(t *testing.T) {
	t.Parallel()

	err := unsupportedFeatureErrorf("responseMimeType %q", "application/xml")
	if !errors.Is(err, ErrUnsupportedFeature) {
		t.Fatal("expected ErrUnsupportedFeature")
	}
}

func TestWrapErrorf(t *testing.T) {
	t.Parallel()

	cause := errors.New("root")
	err := wrapErrorf("wrapped: %s", cause, "details")
	if err == nil || err.Error() != "adkanyllm: wrapped: details" {
		t.Fatalf("unexpected error: %v", err)
	}
	if !errors.Is(err, cause) {
		t.Fatal("expected errors.Is to traverse to cause")
	}
}
