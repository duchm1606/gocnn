package ops

import "duchm1606/gocnn/internal/tensor"

// DepthwiseConv2D performs depthwise separable convolution
// This is more efficient than standard convolution for mobile applications
func DepthwiseConv2D(input *tensor.FeatureMap, kernel *tensor.Kernel, bias []float32, config Conv2DConfig) *tensor.FeatureMap {
    // Validate that kernel has same number of filters as input channels
    if kernel.Filters != input.Channels {
        panic("Depthwise convolution requires kernel.Filters == input.Channels")
    }
    
    // Apply padding if needed
    paddedInput := input
    if config.Padding > 0 {
        paddedInput = tensor.PadFeatureMap(input, config.Padding)
    }
    
    // Calculate output dimensions
    outHeight := (paddedInput.Height-kernel.Size)/config.Stride + 1
    outWidth := (paddedInput.Width-kernel.Size)/config.Stride + 1
    
    output := tensor.NewFeatureMap(outHeight, outWidth, kernel.Filters)
    
    // For each channel, convolve with corresponding filter
    for c := 0; c < input.Channels; c++ {
        for i := 0; i < outHeight; i++ {
            for j := 0; j < outWidth; j++ {
                var sum float32
                
                // Convolve only with the corresponding channel's filter
                for m := 0; m < kernel.Size; m++ {
                    for n := 0; n < kernel.Size; n++ {
                        inputH := i*config.Stride + m
                        inputW := j*config.Stride + n
                        
                        inputVal := paddedInput.GetUnsafe(c, inputH, inputW)
                        kernelWeight := kernel.GetWeightUnsafe(c, c, m, n) // Same channel for both
                        
                        sum += inputVal * kernelWeight
                    }
                }
                
                output.SetUnsafe(c, i, j, sum+bias[c])
            }
        }
    }
    
    return output
}

// PointwiseConv2D performs 1x1 convolution (pointwise convolution)
// This is used for channel-wise feature combination and dimensionality reduction
func PointwiseConv2D(input *tensor.FeatureMap, kernel *tensor.Kernel, bias []float32) *tensor.FeatureMap {
    if kernel.Size != 1 {
        panic("Pointwise convolution requires 1x1 kernel")
    }
    
    config := Conv2DConfig{Padding: 0, Stride: 1}
    return Conv2D(input, kernel, bias, config)
}

// GroupConv2D performs grouped convolution
// Divides input channels into groups and applies separate convolutions
func GroupConv2D(input *tensor.FeatureMap, kernel *tensor.Kernel, bias []float32, 
                groups int, config Conv2DConfig) *tensor.FeatureMap {
    
    if input.Channels%groups != 0 {
        panic("Input channels must be divisible by number of groups")
    }
    
    if kernel.Filters%groups != 0 {
        panic("Kernel filters must be divisible by number of groups")
    }
    
    channelsPerGroup := input.Channels / groups
    filtersPerGroup := kernel.Filters / groups
    
    // Apply padding if needed
    paddedInput := input
    if config.Padding > 0 {
        paddedInput = tensor.PadFeatureMap(input, config.Padding)
    }
    
    // Calculate output dimensions
    outHeight := (paddedInput.Height-kernel.Size)/config.Stride + 1
    outWidth := (paddedInput.Width-kernel.Size)/config.Stride + 1
    
    output := tensor.NewFeatureMap(outHeight, outWidth, kernel.Filters)
    
    // Process each group separately
    for g := 0; g < groups; g++ {
        inputStartChannel := g * channelsPerGroup
        outputStartFilter := g * filtersPerGroup
        
        // Convolve within this group
        for f := 0; f < filtersPerGroup; f++ {
            outputFilter := outputStartFilter + f
            
            for i := 0; i < outHeight; i++ {
                for j := 0; j < outWidth; j++ {
                    var sum float32
                    
                    // Sum over channels in this group only
                    for c := 0; c < channelsPerGroup; c++ {
                        inputChannel := inputStartChannel + c
                        
                        for m := 0; m < kernel.Size; m++ {
                            for n := 0; n < kernel.Size; n++ {
                                inputH := i*config.Stride + m
                                inputW := j*config.Stride + n
                                
                                inputVal := paddedInput.GetUnsafe(inputChannel, inputH, inputW)
                                kernelWeight := kernel.GetWeightUnsafe(outputFilter, inputChannel, m, n)
                                
                                sum += inputVal * kernelWeight
                            }
                        }
                    }
                    
                    output.SetUnsafe(outputFilter, i, j, sum+bias[outputFilter])
                }
            }
        }
    }
    
    return output
}

// Im2Col converts image patches to columns for matrix multiplication
// This is an alternative implementation approach that can be faster for large kernels
func Im2Col(input *tensor.FeatureMap, kernelSize, padding, stride int) [][]float32 {
    paddedInput := input
    if padding > 0 {
        paddedInput = tensor.PadFeatureMap(input, padding)
    }
    
    outHeight := (paddedInput.Height-kernelSize)/stride + 1
    outWidth := (paddedInput.Width-kernelSize)/stride + 1
    
    // Each row represents one output position
    // Each column represents one element from the receptive field
    rows := outHeight * outWidth
    cols := input.Channels * kernelSize * kernelSize
    
    result := make([][]float32, rows)
    
    rowIdx := 0
    for i := 0; i < outHeight; i++ {
        for j := 0; j < outWidth; j++ {
            result[rowIdx] = make([]float32, cols)
            colIdx := 0
            
            for c := 0; c < input.Channels; c++ {
                for m := 0; m < kernelSize; m++ {
                    for n := 0; n < kernelSize; n++ {
                        inputH := i*stride + m
                        inputW := j*stride + n
                        
                        result[rowIdx][colIdx] = paddedInput.GetUnsafe(c, inputH, inputW)
                        colIdx++
                    }
                }
            }
            
            rowIdx++
        }
    }
    
    return result
}

// Conv2DBackward computes gradients for convolution (used in training)
// Not needed for inference, but useful for understanding and testing
func Conv2DBackward(outputGrad *tensor.FeatureMap, input *tensor.FeatureMap, 
                   kernel *tensor.Kernel, config Conv2DConfig) (*tensor.FeatureMap, *tensor.Kernel, []float32) {
    
    // This is a simplified version - full backpropagation is more complex
    // For inference-only CNN, this is not needed but good for completeness
    
    // Compute input gradients (for chaining backward pass)
    inputGrad := tensor.NewFeatureMap(input.Height, input.Width, input.Channels)
    
    // Compute kernel gradients
    kernelGrad := tensor.NewKernel(kernel.Size, kernel.Channels, kernel.Filters)
    
    // Compute bias gradients
    biasGrad := make([]float32, kernel.Filters)
    
    // Implementation would go here...
    // (Omitted for brevity - full implementation is quite complex)
    
    return inputGrad, kernelGrad, biasGrad
}