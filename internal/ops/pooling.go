package ops

import (
	"duchm1606/gocnn/internal/tensor"
	"fmt"
)

// PoolingType defines the type of pooling operation
type PoolingType int

const (
    MaxPooling PoolingType = iota
    AvgPooling
    MinPooling
)

// PoolingConfig holds configuration for pooling operations
type PoolingConfig struct {
    KernelSize int         // Size of pooling window (e.g., 2 for 2x2)
    Stride     int         // Stride for moving the window
    Type       PoolingType // Type of pooling operation
}

// MaxPooling2D performs 2D max pooling operation
// Takes the maximum value in each pooling window
func MaxPooling2D(input *tensor.FeatureMap, kernelSize, stride int) *tensor.FeatureMap {
    config := PoolingConfig{
        KernelSize: kernelSize,
        Stride:     stride,
        Type:       MaxPooling,
    }
    return Pooling2D(input, config)
}

// AvgPooling2D performs 2D average pooling operation
// Takes the average value in each pooling window
func AvgPooling2D(input *tensor.FeatureMap, kernelSize, stride int) *tensor.FeatureMap {
    config := PoolingConfig{
        KernelSize: kernelSize,
        Stride:     stride,
        Type:       AvgPooling,
    }
    return Pooling2D(input, config)
}

// Pooling2D performs general 2D pooling operation
func Pooling2D(input *tensor.FeatureMap, config PoolingConfig) *tensor.FeatureMap {
    // Validate inputs
    if err := validatePoolingInputs(input, config); err != nil {
        panic(fmt.Sprintf("Pooling validation failed: %v", err))
    }
    
    // Calculate output dimensions
    outHeight := (input.Height-config.KernelSize)/config.Stride + 1
    outWidth := (input.Width-config.KernelSize)/config.Stride + 1
    
    // Create output feature map (same number of channels)
    output := tensor.NewFeatureMap(outHeight, outWidth, input.Channels)
    
    // Perform pooling for each channel independently
    for c := 0; c < input.Channels; c++ {
        for i := 0; i < outHeight; i++ {
            for j := 0; j < outWidth; j++ {
                // Define pooling window boundaries
                startH := i * config.Stride
                endH := startH + config.KernelSize
                startW := j * config.Stride
                endW := startW + config.KernelSize
                
                // Apply pooling operation to this window
                value := poolWindow(input, c, startH, endH, startW, endW, config.Type)
                output.SetUnsafe(c, i, j, value)
            }
        }
    }
    
    return output
}

// poolWindow applies pooling operation to a specific window
func poolWindow(input *tensor.FeatureMap, channel, startH, endH, startW, endW int, poolType PoolingType) float32 {
    switch poolType {
    case MaxPooling:
        return maxPoolWindow(input, channel, startH, endH, startW, endW)
    case AvgPooling:
        return avgPoolWindow(input, channel, startH, endH, startW, endW)
    case MinPooling:
        return minPoolWindow(input, channel, startH, endH, startW, endW)
    default:
        panic(fmt.Sprintf("Unknown pooling type: %d", poolType))
    }
}

// maxPoolWindow finds the maximum value in a window
func maxPoolWindow(input *tensor.FeatureMap, channel, startH, endH, startW, endW int) float32 {
    maxVal := input.GetUnsafe(channel, startH, startW)
    
    for h := startH; h < endH; h++ {
        for w := startW; w < endW; w++ {
            val := input.GetUnsafe(channel, h, w)
            if val > maxVal {
                maxVal = val
            }
        }
    }
    
    return maxVal
}

// avgPoolWindow calculates the average value in a window
func avgPoolWindow(input *tensor.FeatureMap, channel, startH, endH, startW, endW int) float32 {
    var sum float32
    count := 0
    
    for h := startH; h < endH; h++ {
        for w := startW; w < endW; w++ {
            sum += input.GetUnsafe(channel, h, w)
            count++
        }
    }
    
    if count > 0 {
        return sum / float32(count)
    }
    return 0
}

// minPoolWindow finds the minimum value in a window
func minPoolWindow(input *tensor.FeatureMap, channel, startH, endH, startW, endW int) float32 {
    minVal := input.GetUnsafe(channel, startH, startW)
    
    for h := startH; h < endH; h++ {
        for w := startW; w < endW; w++ {
            val := input.GetUnsafe(channel, h, w)
            if val < minVal {
                minVal = val
            }
        }
    }
    
    return minVal
}

// validatePoolingInputs validates inputs for pooling operations
func validatePoolingInputs(input *tensor.FeatureMap, config PoolingConfig) error {
    if input == nil {
        return fmt.Errorf("input feature map is nil")
    }
    
    if config.KernelSize <= 0 {
        return fmt.Errorf("kernel size must be positive, got %d", config.KernelSize)
    }
    
    if config.Stride <= 0 {
        return fmt.Errorf("stride must be positive, got %d", config.Stride)
    }
    
    if config.KernelSize > input.Height || config.KernelSize > input.Width {
        return fmt.Errorf("kernel size (%d) larger than input dimensions (%dx%d)", 
            config.KernelSize, input.Height, input.Width)
    }
    
    // Check that we can produce at least one output
    outHeight := (input.Height-config.KernelSize)/config.Stride + 1
    outWidth := (input.Width-config.KernelSize)/config.Stride + 1
    
    if outHeight <= 0 || outWidth <= 0 {
        return fmt.Errorf("pooling configuration produces invalid output dimensions: %dx%d", 
            outHeight, outWidth)
    }
    
    return nil
}

// GetPoolingOutputDims calculates output dimensions for pooling
func GetPoolingOutputDims(inputHeight, inputWidth, kernelSize, stride int) (int, int) {
    outHeight := (inputHeight-kernelSize)/stride + 1
    outWidth := (inputWidth-kernelSize)/stride + 1
    return outHeight, outWidth
}