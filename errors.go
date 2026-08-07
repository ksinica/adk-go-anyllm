package adkanyllm

import (
	"errors"
	"fmt"
)

// ErrUnsupportedFeature indicates a genai or ADK request field the adapter does not implement.
var ErrUnsupportedFeature = errors.New("adkanyllm: unsupported feature")

// AdapterError is the package error type for validation and conversion failures.
type AdapterError struct {
	cause   error
	message string
}

// Error returns the error message prefixed with the package name, including
// the wrapped cause when present so provider and JSON errors stay visible in
// logs instead of being swallowed by the generic wrapper message.
func (e *AdapterError) Error() string {
	if e.cause != nil {
		return fmt.Sprintf("adkanyllm: %s: %s", e.message, e.cause)
	}

	return fmt.Sprintf("adkanyllm: %s", e.message)
}

// Unwrap returns the wrapped cause, enabling errors.Is and errors.As traversal.
func (e *AdapterError) Unwrap() error {
	return e.cause
}

// UnsupportedFeatureError reports a specific unsupported genai or ADK field.
type UnsupportedFeatureError struct {
	// Feature is the name or description of the unsupported field.
	Feature string
}

// Error returns the combined unsupported-feature message including the feature name.
func (e *UnsupportedFeatureError) Error() string {
	return fmt.Sprintf("%s: %s", ErrUnsupportedFeature.Error(), e.Feature)
}

// Is returns true when target is ErrUnsupportedFeature.
func (e *UnsupportedFeatureError) Is(target error) bool {
	return target == ErrUnsupportedFeature
}

func newError(message string) error {
	return &AdapterError{message: message}
}

func newErrorf(format string, args ...any) error {
	return newError(fmt.Sprintf(format, args...))
}

func wrapError(message string, cause error) error {
	return &AdapterError{message: message, cause: cause}
}

func wrapErrorf(format string, cause error, args ...any) error {
	return &AdapterError{message: fmt.Sprintf(format, args...), cause: cause}
}

func unsupportedFeatureError(feature string) error {
	return &UnsupportedFeatureError{Feature: feature}
}

func unsupportedFeatureErrorf(format string, args ...any) error {
	return &UnsupportedFeatureError{Feature: fmt.Sprintf(format, args...)}
}
