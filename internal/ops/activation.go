package ops

import "math"

/**
Why numerical stability matters?
- Large exponentials can cause overflow (inf)
- Subtracting the maximum prevents overflow while preserving mathematical correctness
- `exp(x - max) / sum(exp(x - max))` = `exp(x) / sum(exp(x))`
*/

// ReLU applies Rectified Linear Unit activation
// f(x) = max(0, x)
func ReLU(x float32) float32 {
    if x > 0 {
        return x
    }
    return 0
}

// ReLUInPlace applies ReLU activation to a slice in-place
func ReLUInPlace(data []float32) {
    for i, val := range data {
        if val < 0 {
            data[i] = 0
        }
    }
}

// LeakyReLU applies Leaky ReLU activation
// f(x) = x if x > 0, else alpha * x
func LeakyReLU(x, alpha float32) float32 {
    if x > 0 {
        return x
    }
    return alpha * x
}

// Sigmoid applies sigmoid activation
// f(x) = 1 / (1 + exp(-x))
func Sigmoid(x float32) float32 {
    return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

// Tanh applies hyperbolic tangent activation
// f(x) = tanh(x)
func Tanh(x float32) float32 {
    return float32(math.Tanh(float64(x)))
}

// Softmax applies softmax activation to a slice
// Numerically stable implementation using the log-sum-exp trick
func Softmax(input []float32) []float32 {
    if len(input) == 0 {
        return []float32{}
    }
    
    result := make([]float32, len(input))
    
    // Find maximum for numerical stability
    maxVal := input[0]
    for _, val := range input[1:] {
        if val > maxVal {
            maxVal = val
        }
    }
    
    // Compute exponentials and sum
    var sum float32
    for i, val := range input {
        exp := float32(math.Exp(float64(val - maxVal)))
        result[i] = exp
        sum += exp
    }
    
    // Normalize to get probabilities
    if sum > 0 {
        for i := range result {
            result[i] /= sum
        }
    }
    
    return result
}

// SoftmaxInPlace applies softmax activation in-place
func SoftmaxInPlace(data []float32) {
    if len(data) == 0 {
        return
    }
    
    // Find maximum for numerical stability
    maxVal := data[0]
    for _, val := range data[1:] {
        if val > maxVal {
            maxVal = val
        }
    }
    
    // Compute exponentials and sum
    var sum float32
    for i, val := range data {
        exp := float32(math.Exp(float64(val - maxVal)))
        data[i] = exp
        sum += exp
    }
    
    // Normalize
    if sum > 0 {
        for i := range data {
            data[i] /= sum
        }
    }
}

// LogSoftmax applies log-softmax activation (useful for numerical stability)
func LogSoftmax(input []float32) []float32 {
    if len(input) == 0 {
        return []float32{}
    }
    
    result := make([]float32, len(input))
    
    // Find maximum
    maxVal := input[0]
    for _, val := range input[1:] {
        if val > maxVal {
            maxVal = val
        }
    }
    
    // Compute log-sum-exp
    var sumExp float64
    for _, val := range input {
        sumExp += math.Exp(float64(val - maxVal))
    }
    logSumExp := float32(math.Log(sumExp)) + maxVal
    
    // Compute log-softmax
    for i, val := range input {
        result[i] = val - logSumExp
    }
    
    return result
}