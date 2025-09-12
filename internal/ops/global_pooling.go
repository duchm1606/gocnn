package ops

import (
	"duchm1606/gocnn/internal/tensor"
	"math"
)

// GlobalMaxPooling reduces each feature map to a single maximum value
// Input: (H, W, C) -> Output: (1, 1, C) -> flattened to [C]
func GlobalMaxPooling(input *tensor.FeatureMap) []float32 {
    result := make([]float32, input.Channels)
    
    for c := 0; c < input.Channels; c++ {
        maxVal := input.GetUnsafe(c, 0, 0)
        
        for h := 0; h < input.Height; h++ {
            for w := 0; w < input.Width; w++ {
                val := input.GetUnsafe(c, h, w)
                if val > maxVal {
                    maxVal = val
                }
            }
        }
        
        result[c] = maxVal
    }
    
    return result
}

// GlobalAvgPooling reduces each feature map to a single average value
func GlobalAvgPooling(input *tensor.FeatureMap) []float32 {
    result := make([]float32, input.Channels)
    totalPixels := float32(input.Height * input.Width)
    
    for c := 0; c < input.Channels; c++ {
        var sum float32
        
        for h := 0; h < input.Height; h++ {
            for w := 0; w < input.Width; w++ {
                sum += input.GetUnsafe(c, h, w)
            }
        }
        
        result[c] = sum / totalPixels
    }
    
    return result
}

// GlobalMinPooling reduces each feature map to a single minimum value
func GlobalMinPooling(input *tensor.FeatureMap) []float32 {
    result := make([]float32, input.Channels)
    
    for c := 0; c < input.Channels; c++ {
        minVal := input.GetUnsafe(c, 0, 0)
        
        for h := 0; h < input.Height; h++ {
            for w := 0; w < input.Width; w++ {
                val := input.GetUnsafe(c, h, w)
                if val < minVal {
                    minVal = val
                }
            }
        }
        
        result[c] = minVal
    }
    
    return result
}

// GlobalNormPooling computes L2 norm of each feature map
func GlobalNormPooling(input *tensor.FeatureMap) []float32 {
    result := make([]float32, input.Channels)
    
    for c := 0; c < input.Channels; c++ {
        var sumSquares float64
        
        for h := 0; h < input.Height; h++ {
            for w := 0; w < input.Width; w++ {
                val := float64(input.GetUnsafe(c, h, w))
                sumSquares += val * val
            }
        }
        
        result[c] = float32(math.Sqrt(sumSquares))
    }
    
    return result
}

// GlobalPoolingWithMask performs global pooling with an attention mask
// Useful for masking out irrelevant regions
func GlobalPoolingWithMask(input *tensor.FeatureMap, mask *tensor.FeatureMap, poolType PoolingType) []float32 {
    if mask != nil && (mask.Height != input.Height || mask.Width != input.Width) {
        panic("Mask dimensions must match input dimensions")
    }
    
    result := make([]float32, input.Channels)
    
    for c := 0; c < input.Channels; c++ {
        switch poolType {
        case MaxPooling:
            result[c] = globalMaxWithMask(input, mask, c)
        case AvgPooling:
            result[c] = globalAvgWithMask(input, mask, c)
        default:
            panic("Unsupported pooling type for masked pooling")
        }
    }
    
    return result
}

func globalMaxWithMask(input, mask *tensor.FeatureMap, channel int) float32 {
    maxVal := float32(-math.Inf(1))
    hasValidPixels := false
    
    for h := 0; h < input.Height; h++ {
        for w := 0; w < input.Width; w++ {
            // Check if this pixel is masked (assume mask channel 0)
            if mask == nil || mask.GetUnsafe(0, h, w) > 0 {
                val := input.GetUnsafe(channel, h, w)
                if !hasValidPixels || val > maxVal {
                    maxVal = val
                    hasValidPixels = true
                }
            }
        }
    }
    
    if !hasValidPixels {
        return 0 // or some default value
    }
    
    return maxVal
}

func globalAvgWithMask(input, mask *tensor.FeatureMap, channel int) float32 {
    var sum float32
    var count int
    
    for h := 0; h < input.Height; h++ {
        for w := 0; w < input.Width; w++ {
            if mask == nil || mask.GetUnsafe(0, h, w) > 0 {
                sum += input.GetUnsafe(channel, h, w)
                count++
            }
        }
    }
    
    if count > 0 {
        return sum / float32(count)
    }
    return 0
}

// AdaptivePooling resizes feature maps to specific output dimensions
// Automatically calculates kernel size and stride
func AdaptiveMaxPooling(input *tensor.FeatureMap, outputHeight, outputWidth int) *tensor.FeatureMap {
    if outputHeight <= 0 || outputWidth <= 0 {
        panic("Output dimensions must be positive")
    }
    
    output := tensor.NewFeatureMap(outputHeight, outputWidth, input.Channels)
    
    for c := 0; c < input.Channels; c++ {
        for i := 0; i < outputHeight; i++ {
            for j := 0; j < outputWidth; j++ {
                // Calculate adaptive window boundaries
                startH := (i * input.Height) / outputHeight
                endH := ((i + 1) * input.Height) / outputHeight
                startW := (j * input.Width) / outputWidth
                endW := ((j + 1) * input.Width) / outputWidth
                
                // Ensure we don't go out of bounds
                if endH > input.Height {
                    endH = input.Height
                }
                if endW > input.Width {
                    endW = input.Width
                }
                
                // Find maximum in this adaptive window
                maxVal := input.GetUnsafe(c, startH, startW)
                for h := startH; h < endH; h++ {
                    for w := startW; w < endW; w++ {
                        val := input.GetUnsafe(c, h, w)
                        if val > maxVal {
                            maxVal = val
                        }
                    }
                }
                
                output.SetUnsafe(c, i, j, maxVal)
            }
        }
    }
    
    return output
}

// AdaptiveAvgPooling performs adaptive average pooling
func AdaptiveAvgPooling(input *tensor.FeatureMap, outputHeight, outputWidth int) *tensor.FeatureMap {
    if outputHeight <= 0 || outputWidth <= 0 {
        panic("Output dimensions must be positive")
    }
    
    output := tensor.NewFeatureMap(outputHeight, outputWidth, input.Channels)
    
    for c := 0; c < input.Channels; c++ {
        for i := 0; i < outputHeight; i++ {
            for j := 0; j < outputWidth; j++ {
                // Calculate adaptive window boundaries
                startH := (i * input.Height) / outputHeight
                endH := ((i + 1) * input.Height) / outputHeight
                startW := (j * input.Width) / outputWidth
                endW := ((j + 1) * input.Width) / outputWidth
                
                if endH > input.Height {
                    endH = input.Height
                }
                if endW > input.Width {
                    endW = input.Width
                }
                
                // Calculate average in this adaptive window
                var sum float32
                count := 0
                for h := startH; h < endH; h++ {
                    for w := startW; w < endW; w++ {
                        sum += input.GetUnsafe(c, h, w)
                        count++
                    }
                }
                
                if count > 0 {
                    output.SetUnsafe(c, i, j, sum/float32(count))
                } else {
                    output.SetUnsafe(c, i, j, 0)
                }
            }
        }
    }
    
    return output
}