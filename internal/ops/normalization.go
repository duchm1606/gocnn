package ops

import (
	"duchm1606/gocnn/internal/tensor"
	"math"
)

// Implement batch normalization

// BatchNormParams holds the parameters for batch normalization
type BatchNormParams struct {
    Mean     []float32 // Moving mean for each channel
    Variance []float32 // Moving variance for each channel
    Scale    []float32 // Learned scale (gamma) for each channel
    Shift    []float32 // Learned shift (beta) for each channel
    Epsilon  float32   // Small constant for numerical stability
}

// NewBatchNormParams creates new batch normalization parameters
func NewBatchNormParams(channels int) *BatchNormParams {
    return &BatchNormParams{
        Mean:     make([]float32, channels),
        Variance: make([]float32, channels),
        Scale:    make([]float32, channels),
        Shift:    make([]float32, channels),
        Epsilon:  1e-5,
    }
}

// BatchNormalize applies batch normalization to a feature map
// Formula: y = scale * ((x - mean) / sqrt(variance + epsilon)) + shift
func BatchNormalize(fm *tensor.FeatureMap, params *BatchNormParams) *tensor.FeatureMap {
    if len(params.Mean) != fm.Channels {
        panic("BatchNorm parameters don't match feature map channels")
    }
    
    result := fm.Clone()
    
    for c := 0; c < fm.Channels; c++ {
        mean := params.Mean[c]
        variance := params.Variance[c]
        scale := params.Scale[c]
        shift := params.Shift[c]
        
        // Compute normalization factor
        stdDev := float32(math.Sqrt(float64(variance + params.Epsilon)))
        
        // Apply normalization to all pixels in this channel
        for h := 0; h < fm.Height; h++ {
            for w := 0; w < fm.Width; w++ {
                val := fm.GetUnsafe(c, h, w)
                
                // Normalize: (x - mean) / sqrt(variance + epsilon)
                normalized := (val - mean) / stdDev
                
                // Scale and shift: gamma * normalized + beta
                transformed := scale*normalized + shift
                
                // Apply ReLU activation (common in CNN)
                if transformed < 0 {
                    transformed = 0
                }
                
                result.SetUnsafe(c, h, w, transformed)
            }
        }
    }
    
    return result
}

// BatchNormalizeInPlace applies batch normalization in-place
func BatchNormalizeInPlace(fm *tensor.FeatureMap, params *BatchNormParams) {
    if len(params.Mean) != fm.Channels {
        panic("BatchNorm parameters don't match feature map channels")
    }
    
    for c := 0; c < fm.Channels; c++ {
        mean := params.Mean[c]
        variance := params.Variance[c]
        scale := params.Scale[c]
        shift := params.Shift[c]
        
        stdDev := float32(math.Sqrt(float64(variance + params.Epsilon)))
        
        for h := 0; h < fm.Height; h++ {
            for w := 0; w < fm.Width; w++ {
                val := fm.GetUnsafe(c, h, w)
                normalized := (val - mean) / stdDev
                transformed := scale*normalized + shift
                
                // Apply ReLU
                if transformed < 0 {
                    transformed = 0
                }
                
                fm.SetUnsafe(c, h, w, transformed)
            }
        }
    }
}

// ComputeBatchStatistics computes mean and variance from a batch of feature maps
// Useful for training (not needed for inference, but good for testing)
func ComputeBatchStatistics(batch []*tensor.FeatureMap) ([]float32, []float32) {
    if len(batch) == 0 {
        return nil, nil
    }
    
    channels := batch[0].Channels
    height := batch[0].Height
    width := batch[0].Width
    batchSize := len(batch)
    
    means := make([]float32, channels)
    variances := make([]float32, channels)
    
    // Compute means
    for c := 0; c < channels; c++ {
        var sum float32
        totalElements := batchSize * height * width
        
        for _, fm := range batch {
            for h := 0; h < height; h++ {
                for w := 0; w < width; w++ {
                    sum += fm.GetUnsafe(c, h, w)
                }
            }
        }
        
        means[c] = sum / float32(totalElements)
    }
    
    // Compute variances
    for c := 0; c < channels; c++ {
        var sumSquaredDiff float32
        totalElements := batchSize * height * width
        mean := means[c]
        
        for _, fm := range batch {
            for h := 0; h < height; h++ {
                for w := 0; w < width; w++ {
                    val := fm.GetUnsafe(c, h, w)
                    diff := val - mean
                    sumSquaredDiff += diff * diff
                }
            }
        }
        
        variances[c] = sumSquaredDiff / float32(totalElements)
    }
    
    return means, variances
}