package ops

import (
	"math"
	"testing"
)

func TestReLU(t *testing.T) {
    testCases := []struct {
        input    float32
        expected float32
    }{
        {-1.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0},
        {5.5, 5.5},
        {-100.0, 0.0},
    }
    
    for _, tc := range testCases {
        result := ReLU(tc.input)
        if result != tc.expected {
            t.Errorf("ReLU(%f) = %f, expected %f", tc.input, result, tc.expected)
        }
    }
}

func TestSoftmax(t *testing.T) {
    input := []float32{1.0, 2.0, 3.0}
    result := Softmax(input)
    
    // Check that probabilities sum to 1
    sum := Sum(result)
    if math.Abs(float64(sum-1.0)) > 1e-6 {
        t.Errorf("Softmax probabilities don't sum to 1: %f", sum)
    }
    
    // Check that all probabilities are positive
    for i, prob := range result {
        if prob <= 0 {
            t.Errorf("Softmax probability %d is not positive: %f", i, prob)
        }
    }
    
    // Check that larger inputs have larger probabilities
    if result[2] <= result[1] || result[1] <= result[0] {
        t.Error("Softmax doesn't preserve order")
    }
}

func TestSoftmaxNumericalStability(t *testing.T) {
    // Test with large values that could cause overflow
    input := []float32{1000.0, 1001.0, 1002.0}
    result := Softmax(input)
    
    // Should not contain NaN or Inf
    for i, val := range result {
        if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
            t.Errorf("Softmax result contains NaN/Inf at index %d: %f", i, val)
        }
    }
    
    // Should still sum to 1
    sum := Sum(result)
    if math.Abs(float64(sum-1.0)) > 1e-6 {
        t.Errorf("Softmax with large inputs doesn't sum to 1: %f", sum)
    }
}

func TestArgmax(t *testing.T) {
    testCases := []struct {
        input    []float32
        expected int
    }{
        {[]float32{1.0, 2.0, 3.0}, 2},
        {[]float32{3.0, 1.0, 2.0}, 0},
        {[]float32{1.0, 3.0, 2.0}, 1},
        {[]float32{1.0}, 0},
        {[]float32{}, -1},
    }
    
    for _, tc := range testCases {
        result := Argmax(tc.input)
        if result != tc.expected {
            t.Errorf("Argmax(%v) = %d, expected %d", tc.input, result, tc.expected)
        }
    }
}

// Benchmark tests
func BenchmarkReLU(b *testing.B) {
    for i := 0; i < b.N; i++ {
        _ = ReLU(float32(i) - float32(b.N/2))
    }
}

func BenchmarkSoftmax(b *testing.B) {
    input := make([]float32, 10)
    for i := range input {
        input[i] = float32(i)
    }
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = Softmax(input)
    }
}

func BenchmarkSoftmaxInPlace(b *testing.B) {
    input := make([]float32, 10)
    for i := range input {
        input[i] = float32(i)
    }
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        SoftmaxInPlace(input)
    }
}
