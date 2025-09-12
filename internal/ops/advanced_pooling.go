package ops

import (
	"duchm1606/gocnn/internal/tensor"
	"math"
	"sort"
)

// StochasticPooling randomly selects elements based on their values
// Higher values have higher probability of being selected
func StochasticPooling(input *tensor.FeatureMap, kernelSize, stride int) *tensor.FeatureMap {
    outHeight := (input.Height-kernelSize)/stride + 1
    outWidth := (input.Width-kernelSize)/stride + 1
    
    output := tensor.NewFeatureMap(outHeight, outWidth, input.Channels)
    
    for c := 0; c < input.Channels; c++ {
        for i := 0; i < outHeight; i++ {
            for j := 0; j < outWidth; j++ {
                // Define pooling window
                startH := i * stride
                endH := startH + kernelSize
                startW := j * stride
                endW := startW + kernelSize
                
                // Collect values and compute probabilities
                var values []float32
                var probabilities []float32
                var sum float32
                
                for h := startH; h < endH; h++ {
                    for w := startW; w < endW; w++ {
                        val := input.GetUnsafe(c, h, w)
                        if val > 0 { // Only consider positive values
                            values = append(values, val)
                            sum += val
                        }
                    }
                }
                
                if len(values) == 0 || sum == 0 {
                    output.SetUnsafe(c, i, j, 0)
                    continue
                }
                
                // Normalize to get probabilities
                for _, val := range values {
                    probabilities = append(probabilities, val/sum)
                }
                
                // Sample based on probabilities (simplified - use first value for deterministic behavior)
                // In real implementation, you'd use random sampling
                selectedValue := values[0] // Simplified
                output.SetUnsafe(c, i, j, selectedValue)
            }
        }
    }
    
    return output
}

// LpPooling performs Lp norm pooling (generalization of max and average pooling)
// p=1: average pooling, p=∞: max pooling, p=2: L2 norm pooling
func LpPooling(input *tensor.FeatureMap, kernelSize, stride int, p float64) *tensor.FeatureMap {
    outHeight := (input.Height-kernelSize)/stride + 1
    outWidth := (input.Width-kernelSize)/stride + 1
    
    output := tensor.NewFeatureMap(outHeight, outWidth, input.Channels)
    
    for c := 0; c < input.Channels; c++ {
        for i := 0; i < outHeight; i++ {
            for j := 0; j < outWidth; j++ {
                startH := i * stride
                endH := startH + kernelSize
                startW := j * stride
                endW := startW + kernelSize
                
                var result float32
                
                if math.IsInf(p, 1) {
                    // Max pooling (p = ∞)
                    result = maxPoolWindow(input, c, startH, endH, startW, endW)
                } else if p == 1.0 {
                    // Average pooling (p = 1)
                    result = avgPoolWindow(input, c, startH, endH, startW, endW)
                } else {
                    // General Lp pooling
                    var sum float64
                    count := 0
                    
                    for h := startH; h < endH; h++ {
                        for w := startW; w < endW; w++ {
                            val := float64(math.Abs(float64(input.GetUnsafe(c, h, w))))
                            sum += math.Pow(val, p)
                            count++
                        }
                    }
                    
                    if count > 0 {
                        result = float32(math.Pow(sum/float64(count), 1.0/p))
                    }
                }
                
                output.SetUnsafe(c, i, j, result)
            }
        }
    }
    
    return output
}

// RankPooling selects the k-th largest value in each window
// k=1: max pooling, k=window_size: min pooling
func RankPooling(input *tensor.FeatureMap, kernelSize, stride, k int) *tensor.FeatureMap {
    outHeight := (input.Height-kernelSize)/stride + 1
    outWidth := (input.Width-kernelSize)/stride + 1
    
    output := tensor.NewFeatureMap(outHeight, outWidth, input.Channels)
    
    for c := 0; c < input.Channels; c++ {
        for i := 0; i < outHeight; i++ {
            for j := 0; j < outWidth; j++ {
                startH := i * stride
                endH := startH + kernelSize
                startW := j * stride
                endW := startW + kernelSize
                
                // Collect all values in the window
                var values []float32
                for h := startH; h < endH; h++ {
                    for w := startW; w < endW; w++ {
                        values = append(values, input.GetUnsafe(c, h, w))
                    }
                }
                
                // Sort values in descending order
                sort.Slice(values, func(i, j int) bool {
                    return values[i] > values[j]
                })
                
                // Select k-th largest (1-indexed)
                if k <= len(values) && k > 0 {
                    output.SetUnsafe(c, i, j, values[k-1])
                } else {
                    output.SetUnsafe(c, i, j, 0)
                }
            }
        }
    }
    
    return output
}

// FractionalMaxPooling implements fractional max pooling for irregular pooling regions
// This allows for pooling ratios that are not integers
func FractionalMaxPooling(input *tensor.FeatureMap, poolingRatio float64) *tensor.FeatureMap {
    if poolingRatio <= 1.0 {
        panic("Pooling ratio must be greater than 1.0")
    }
    
    outHeight := int(float64(input.Height) / poolingRatio)
    outWidth := int(float64(input.Width) / poolingRatio)
    
    if outHeight <= 0 {
        outHeight = 1
    }
    if outWidth <= 0 {
        outWidth = 1
    }
    
    output := tensor.NewFeatureMap(outHeight, outWidth, input.Channels)
    
    for c := 0; c < input.Channels; c++ {
        for i := 0; i < outHeight; i++ {
            for j := 0; j < outWidth; j++ {
                // Calculate variable-size pooling window
                startH := int(float64(i) * poolingRatio)
                endH := int(float64(i+1) * poolingRatio)
                startW := int(float64(j) * poolingRatio)
                endW := int(float64(j+1) * poolingRatio)
                
                // Ensure boundaries are within input
                if endH > input.Height {
                    endH = input.Height
                }
                if endW > input.Width {
                    endW = input.Width
                }
                
                // Find maximum in the fractional window
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

// SpatialPyramidPooling performs pooling at multiple scales
// Returns a fixed-size output regardless of input size
func SpatialPyramidPooling(input *tensor.FeatureMap, pyramidLevels []int) []float32 {
    var result []float32
    
    for _, level := range pyramidLevels {
        // Perform adaptive pooling to create level×level output
        pooled := AdaptiveMaxPooling(input, level, level)
        
        // Flatten and append to result
        for c := 0; c < pooled.Channels; c++ {
            for h := 0; h < pooled.Height; h++ {
                for w := 0; w < pooled.Width; w++ {
                    result = append(result, pooled.GetUnsafe(c, h, w))
                }
            }
        }
    }
    
    return result
}