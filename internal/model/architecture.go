package model

import "fmt"

// LayerType defines the type of neural network layer
type LayerType int

const (
    ConvolutionLayer LayerType = iota
    MaxPoolingLayer
    GlobalMaxPoolingLayer
    SoftmaxLayer
    BatchNormLayer
)

// LayerConfig defines configuration for a single layer
type LayerConfig struct {
    Type       LayerType
    Name       string
    
    // Convolution parameters
    KernelSize int
    Filters    int
    Stride     int
    Padding    int
    
    // Pooling parameters
    PoolSize   int
    PoolStride int
    
    // Other parameters
    ApplyBatchNorm bool
    ApplyActivation bool
}

// TinyCNNArchitecture defines the complete network architecture
type TinyCNNArchitecture struct {
    InputHeight   int
    InputWidth    int
    InputChannels int
    NumClasses    int
    Layers        []LayerConfig
}

// GetTinyCNNArchitecture returns the standard TinyCNN architecture for CIFAR-10
func GetTinyCNNArchitecture() *TinyCNNArchitecture {
    return &TinyCNNArchitecture{
        InputHeight:   32,
        InputWidth:    32,
        InputChannels: 3,
        NumClasses:    10,
        Layers: []LayerConfig{
            {
                Type:            ConvolutionLayer,
                Name:            "conv1",
                KernelSize:      3,
                Filters:         32,
                Stride:          1,
                Padding:         1,
                ApplyBatchNorm:  true,
                ApplyActivation: true,
            },
            {
                Type:            ConvolutionLayer,
                Name:            "conv2", 
                KernelSize:      3,
                Filters:         32,
                Stride:          1,
                Padding:         1,
                ApplyBatchNorm:  true,
                ApplyActivation: true,
            },
            {
                Type:       MaxPoolingLayer,
                Name:       "maxpool1",
                PoolSize:   2,
                PoolStride: 2,
            },
            {
                Type:            ConvolutionLayer,
                Name:            "conv3",
                KernelSize:      3,
                Filters:         64,
                Stride:          1,
                Padding:         1,
                ApplyBatchNorm:  true,
                ApplyActivation: true,
            },
            {
                Type:            ConvolutionLayer,
                Name:            "conv4",
                KernelSize:      3,
                Filters:         64,
                Stride:          1,
                Padding:         1,
                ApplyBatchNorm:  true,
                ApplyActivation: true,
            },
            {
                Type:       MaxPoolingLayer,
                Name:       "maxpool2",
                PoolSize:   2,
                PoolStride: 2,
            },
            {
                Type:            ConvolutionLayer,
                Name:            "conv5",
                KernelSize:      3,
                Filters:         128,
                Stride:          1,
                Padding:         1,
                ApplyBatchNorm:  true,
                ApplyActivation: true,
            },
            {
                Type:            ConvolutionLayer,
                Name:            "conv6",
                KernelSize:      3,
                Filters:         128,
                Stride:          1,
                Padding:         1,
                ApplyBatchNorm:  true,
                ApplyActivation: true,
            },
            {
                Type:       MaxPoolingLayer,
                Name:       "maxpool3",
                PoolSize:   2,
                PoolStride: 2,
            },
            {
                Type:            ConvolutionLayer,
                Name:            "conv7",
                KernelSize:      1,
                Filters:         10,
                Stride:          1,
                Padding:         0,
                ApplyBatchNorm:  false,
                ApplyActivation: false,
            },
            {
                Type: GlobalMaxPoolingLayer,
                Name: "global_maxpool",
            },
            {
                Type: SoftmaxLayer,
                Name: "softmax",
            },
        },
    }
}

// ValidateArchitecture checks if the architecture is valid
func (arch *TinyCNNArchitecture) ValidateArchitecture() error {
    if arch.InputHeight <= 0 || arch.InputWidth <= 0 || arch.InputChannels <= 0 {
        return fmt.Errorf("invalid input dimensions: (%d, %d, %d)", 
            arch.InputHeight, arch.InputWidth, arch.InputChannels)
    }
    
    if arch.NumClasses <= 0 {
        return fmt.Errorf("invalid number of classes: %d", arch.NumClasses)
    }
    
    if len(arch.Layers) == 0 {
        return fmt.Errorf("architecture has no layers")
    }
    
    // Validate each layer
    for i, layer := range arch.Layers {
        err := validateLayerConfig(layer)
        if err != nil {
            return fmt.Errorf("layer %d (%s) is invalid: %w", i, layer.Name, err)
        }
    }
    
    return nil
}

func validateLayerConfig(layer LayerConfig) error {
    switch layer.Type {
    case ConvolutionLayer:
        if layer.KernelSize <= 0 {
            return fmt.Errorf("invalid kernel size: %d", layer.KernelSize)
        }
        if layer.Filters <= 0 {
            return fmt.Errorf("invalid number of filters: %d", layer.Filters)
        }
        if layer.Stride <= 0 {
            return fmt.Errorf("invalid stride: %d", layer.Stride)
        }
        if layer.Padding < 0 {
            return fmt.Errorf("invalid padding: %d", layer.Padding)
        }
        
    case MaxPoolingLayer:
        if layer.PoolSize <= 0 {
            return fmt.Errorf("invalid pool size: %d", layer.PoolSize)
        }
        if layer.PoolStride <= 0 {
            return fmt.Errorf("invalid pool stride: %d", layer.PoolStride)
        }
        
    case GlobalMaxPoolingLayer, SoftmaxLayer:
        // No specific validation needed
        
    default:
        return fmt.Errorf("unknown layer type: %d", layer.Type)
    }
    
    return nil
}

// GetOutputDimensions calculates the output dimensions after each layer
func (arch *TinyCNNArchitecture) GetOutputDimensions() ([][]int, error) {
    dimensions := make([][]int, len(arch.Layers)+1)
    
    // Input dimensions
    currentH := arch.InputHeight
    currentW := arch.InputWidth
    currentC := arch.InputChannels
    dimensions[0] = []int{currentH, currentW, currentC}
    
    for i, layer := range arch.Layers {
        switch layer.Type {
        case ConvolutionLayer:
            // Apply padding, then convolution
            paddedH := currentH + 2*layer.Padding
            paddedW := currentW + 2*layer.Padding
            
            currentH = (paddedH-layer.KernelSize)/layer.Stride + 1
            currentW = (paddedW-layer.KernelSize)/layer.Stride + 1
            currentC = layer.Filters
            
        case MaxPoolingLayer:
            currentH = (currentH-layer.PoolSize)/layer.PoolStride + 1
            currentW = (currentW-layer.PoolSize)/layer.PoolStride + 1
            // Channels unchanged
            
        case GlobalMaxPoolingLayer:
            currentH = 1
            currentW = 1
            // Channels unchanged
            
        case SoftmaxLayer:
            // Dimensions unchanged (applied to flattened vector)
        }
        
        dimensions[i+1] = []int{currentH, currentW, currentC}
    }
    
    return dimensions, nil
}