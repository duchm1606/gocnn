package ops

import (
	"math"
	"testing"
)

func TestMean(t *testing.T) {
    testCases := []struct {
        input    []float32
        expected float32
    }{
        {[]float32{1.0, 2.0, 3.0}, 2.0},
        {[]float32{0.0, 0.0, 0.0}, 0.0},
        {[]float32{-1.0, 1.0}, 0.0},
        {[]float32{5.0}, 5.0},
        {[]float32{}, 0.0},
    }
    
    for _, tc := range testCases {
        result := Mean(tc.input)
        if math.Abs(float64(result-tc.expected)) > 1e-6 {
            t.Errorf("Mean(%v) = %f, expected %f", tc.input, result, tc.expected)
        }
    }
}

func TestVariance(t *testing.T) {
    input := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
    result := Variance(input)
    expected := float32(2.5) // Sample variance
    
    if math.Abs(float64(result-expected)) > 1e-6 {
        t.Errorf("Variance(%v) = %f, expected %f", input, result, expected)
    }
}

func TestNormalize(t *testing.T) {
    input := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
    result := Normalize(input)
    
    // Check that result has mean ≈ 0 and std ≈ 1
    resultMean := Mean(result)
    resultStd := StandardDeviation(result)
    
    if math.Abs(float64(resultMean)) > 1e-6 {
        t.Errorf("Normalized data mean is not 0: %f", resultMean)
    }
    
    if math.Abs(float64(resultStd-1.0)) > 1e-6 {
        t.Errorf("Normalized data std is not 1: %f", resultStd)
    }
}

func TestClip(t *testing.T) {
    input := []float32{-5.0, -1.0, 0.0, 1.0, 5.0}
    result := Clip(input, -2.0, 2.0)
    expected := []float32{-2.0, -1.0, 0.0, 1.0, 2.0}
    
    for i, val := range result {
        if val != expected[i] {
            t.Errorf("Clip result[%d] = %f, expected %f", i, val, expected[i])
        }
    }
}