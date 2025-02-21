// Package parameters handles generic configuration Params, a map[string]string that the
// user can set.
package parameters

import (
	"github.com/pkg/errors"
	"strconv"
	"strings"
)

// Params represent generic configuration parameters.
type Params map[string]string

// NewFromConfigString create params from user's configuration string.
// See GetParamOr and PopParamOr to parse values from this map.
func NewFromConfigString(config string) map[string]string {
	params := make(map[string]string)
	parts := strings.Split(config, ",")
	for _, part := range parts {
		subParts := strings.SplitN(part, "=", 2) // Split into up to 2 parts to handle '=' in values
		if len(subParts) == 1 {
			params[subParts[0]] = ""
		} else if len(subParts) == 2 {
			params[subParts[0]] = subParts[1]
		}
	}
	return params
}

// PopParamOr is like GetParamOr, but it also deletes from the params map the retrieved parameter.
func PopParamOr[T interface {
	bool | int | float32 | float64 | string
}](params Params, key string, defaultValue T) (T, error) {
	value, err := GetParamOr(params, key, defaultValue)
	if err != nil {
		return value, err
	}
	delete(params, key)
	return value, nil
}

// GetParamOr attempts to parse a parameter to the given type if the key is present, or returns the defaultValue
// if not.
//
// For bool types, a key without a value is interpreted as true.
func GetParamOr[T interface {
	bool | int | float32 | float64 | string
}](params Params, key string, defaultValue T) (T, error) {
	vAny := (any)(defaultValue)
	var t T
	toT := func(v any) T { return v.(T) }
	switch vAny.(type) {
	case string:
		if value, exists := params[key]; exists {
			return toT(value), nil
		}
	case int:
		if value, exists := params[key]; exists && value != "" {
			parsedValue, err := strconv.Atoi(value)
			if err != nil {
				return t, errors.Wrapf(err, "failed to parse configuration %s=%q to int", key, value)
			}
			return toT(parsedValue), nil
		}
	case float32:
		if value, exists := params[key]; exists && value != "" {
			parsedValue, err := strconv.ParseFloat(value, 32)
			if err != nil {
				return t, errors.Wrapf(err, "failed to parse configuration %s=%q to float", key, value)
			}
			return toT(float32(parsedValue)), nil
		}
	case float64:
		if value, exists := params[key]; exists && value != "" {
			parsedValue, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return t, errors.Wrapf(err, "failed to parse configuration %s=%q to float", key, value)
			}
			return toT(parsedValue), nil
		}
	case bool:
		if value, exists := params[key]; exists {
			if value == "" || strings.ToLower(value) == "true" || value == "1" { // Empty value is considered "true"
				return toT(true), nil
			}
			if strings.ToLower(value) == "false" || value == "0" {
				return toT(false), nil
			}
			return defaultValue, errors.New("failed to parse bool")
		}
	}
	return defaultValue, nil
}
