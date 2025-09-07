package tensor

import (
	"fmt"
	"math"
)

// Argmax returns the index of the maximum value in a slice
func Argmax(slice []float32) int {
    if len(slice) == 0 {
        return -1
    }
    
    maxIdx := 0
    maxVal := slice[0]
    
    for i, val := range slice[1:] {
        if val > maxVal {
            maxVal = val
            maxIdx = i + 1
        }
    }
    
    return maxIdx
}

// Max returns the maximum value in a slice
func Max(slice []float32) float32 {
    if len(slice) == 0 {
        return 0
    }
    
    maxVal := slice[0]
    for _, val := range slice[1:] {
        if val > maxVal {
            maxVal = val
        }
    }
    
    return maxVal
}


// Sum returns the sum of all values in a slice
func Sum(slice []float32) float32 {
    var sum float32
    for _, val := range slice {
        sum += val
    }
    return sum
}

// Mean returns the average value of a slice
func Mean(slice []float32) float32 {
    if len(slice) == 0 {
        return 0
    }
    return Sum(slice) / float32(len(slice))
}

// PadFeatureMap creates a new feature map with zero padding
func PadFeatureMap(input *FeatureMap, padding int) *FeatureMap {
    if padding <= 0 {
        return input.Clone()
    }
    
    newHeight := input.Height + 2*padding
    newWidth := input.Width + 2*padding
    
    padded := NewFeatureMap(newHeight, newWidth, input.Channels)
    
    // Copy original data to center of padded feature map
    for c := 0; c < input.Channels; c++ {
        for h := 0; h < input.Height; h++ {
            for w := 0; w < input.Width; w++ {
                value := input.GetUnsafe(c, h, w)
                padded.SetUnsafe(c, h+padding, w+padding, value)
            }
        }
    }
    
    return padded
}

// ValidateFeatureMap checks if a feature map has valid dimensions and data
func ValidateFeatureMap(fm *FeatureMap) error {
    if fm == nil {
        return fmt.Errorf("feature map is nil")
    }
    
    if fm.Height <= 0 || fm.Width <= 0 || fm.Channels <= 0 {
        return fmt.Errorf("invalid dimensions: height=%d, width=%d, channels=%d", 
            fm.Height, fm.Width, fm.Channels)
    }
    
    expectedSize := fm.Height * fm.Width * fm.Channels
    if len(fm.Data) != expectedSize {
        return fmt.Errorf("data size mismatch: expected %d, got %d", expectedSize, len(fm.Data))
    }
    
    // Check for NaN or Inf values
    for i, val := range fm.Data {
        if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
            return fmt.Errorf("invalid value at index %d: %f", i, val)
        }
    }
    
    return nil
}

// ValidateKernel checks if a kernel has valid dimensions and weights
func ValidateKernel(k *Kernel) error {
    if k == nil {
        return fmt.Errorf("kernel is nil")
    }
    
    if k.Size <= 0 || k.Channels <= 0 || k.Filters <= 0 {
        return fmt.Errorf("invalid dimensions: size=%d, channels=%d, filters=%d", 
            k.Size, k.Channels, k.Filters)
    }
    
    expectedSize := k.Size * k.Size * k.Channels * k.Filters
    if len(k.Weights) != expectedSize {
        return fmt.Errorf("weights size mismatch: expected %d, got %d", expectedSize, len(k.Weights))
    }
    
    // Check for NaN or Inf values
    for i, val := range k.Weights {
        if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
            return fmt.Errorf("invalid weight at index %d: %f", i, val)
        }
    }
    
    return nil
}