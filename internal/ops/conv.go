package ops

import (
	"duchm1606/gocnn/internal/tensor"
	"fmt"
	"runtime"
	"sync"
)

/**
* Background Knowledge

Convolution is the fundamental operation in CNNs:
```
Output[f][i][j] = Σ(Input[c][i*stride+m][j*stride+n] * Kernel[f][c][m][n]) + Bias[f]
```

Where:
- `f` = output filter index
- `c` = input channel index
- `i,j` = output spatial coordinates
- `m,n` = kernel spatial coordinates
- `stride` = step size for sliding the kernel
- Bias is added per filter
*/

// Conv2DConfig holds configuration for convolution operation
type Conv2DConfig struct {
    Padding int // Padding size (0 = no padding, 1 = 1-pixel border, etc.)
    Stride  int // Stride for sliding the kernel (1 = slide by 1 pixel)
}


// Conv2D performs 2D convolution operation
// This is the core CNN operation that detects features in images
func Conv2D(input *tensor.FeatureMap, kernel *tensor.Kernel, bias []float32, config Conv2DConfig) *tensor.FeatureMap {
    // Validate inputs
    if err := validateConv2DInputs(input, kernel, bias, config); err != nil {
        panic(fmt.Sprintf("Conv2D validation failed: %v", err))
    }
    
    // Apply padding if needed
    paddedInput := input
    if config.Padding > 0 {
        paddedInput = tensor.PadFeatureMap(input, config.Padding)
    }
    
    // Calculate output dimensions using the formula:
    // output_size = (input_size - kernel_size + 2*padding) / stride + 1
    outHeight := (paddedInput.Height-kernel.Size)/config.Stride + 1
    outWidth := (paddedInput.Width-kernel.Size)/config.Stride + 1
    
    // Create output feature map
    output := tensor.NewFeatureMap(outHeight, outWidth, kernel.Filters)
    
    // Perform convolution for each output filter
    for f := 0; f < kernel.Filters; f++ {
        convolveFilter(paddedInput, kernel, output, f, bias[f], config)
    }
    
    return output
}

// convolveFilter performs convolution for a single output filter
func convolveFilter(input *tensor.FeatureMap, kernel *tensor.Kernel, output *tensor.FeatureMap, 
	filterIdx int, bias float32, config Conv2DConfig) {
	for i := 0; i < output.Height; i++ {
		for j := 0; j < output.Width; j++ {
		// Compute convolution at position (i, j)
		var sum float32

		// Sum across all input channels
		for c := 0; c < kernel.Channels; c++ {
		// Sum across kernel spatial dimensions
		for m := 0; m < kernel.Size; m++ {
			for n := 0; n < kernel.Size; n++ {
				// Input coordinates
				inputH := i*config.Stride + m
				inputW := j*config.Stride + n
				
				// Get input value and kernel weight
				inputVal := input.GetUnsafe(c, inputH, inputW)
				kernelWeight := kernel.GetWeightUnsafe(filterIdx, c, m, n)
				
				// Accumulate the product
				sum += inputVal * kernelWeight
			}
		}
		}

		// Add bias and store result
		output.SetUnsafe(filterIdx, i, j, sum+bias)
		}
	}
}

/**
* Parallel convolution - Why parallel convolution?
- Each output filter can be computed independently
- Modern CPUs have multiple cores that can work simultaneously
- Go's goroutines make parallelization easy and efficient
- Typical speedup: 2-4x on quad-core systems
*/


// Conv2DParallel performs parallel convolution using goroutines
// This provides significant speedup on multi-core systems
func Conv2DParallel(input *tensor.FeatureMap, kernel *tensor.Kernel, bias []float32, config Conv2DConfig) *tensor.FeatureMap {
    // Validate inputs
    if err := validateConv2DInputs(input, kernel, bias, config); err != nil {
        panic(fmt.Sprintf("Conv2D validation failed: %v", err))
    }
    
    // Apply padding if needed
    paddedInput := input
    if config.Padding > 0 {
        paddedInput = tensor.PadFeatureMap(input, config.Padding)
    }
    
    // Calculate output dimensions
    outHeight := (paddedInput.Height-kernel.Size)/config.Stride + 1
    outWidth := (paddedInput.Width-kernel.Size)/config.Stride + 1
    
    // Create output feature map
    output := tensor.NewFeatureMap(outHeight, outWidth, kernel.Filters)
    
    // Determine number of workers (use all CPU cores)
    numWorkers := runtime.NumCPU()
    if numWorkers > kernel.Filters {
        numWorkers = kernel.Filters
    }
    
    // Create work channels
    jobs := make(chan int, kernel.Filters)
    var wg sync.WaitGroup
    
    // Start worker goroutines
    for w := 0; w < numWorkers; w++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for filterIdx := range jobs {
                convolveFilter(paddedInput, kernel, output, filterIdx, bias[filterIdx], config)
            }
        }()
    }
    
    // Send filter indices to workers
    for f := 0; f < kernel.Filters; f++ {
        jobs <- f
    }
    close(jobs)
    
    // Wait for all workers to complete
    wg.Wait()
    
    return output
}

// validateConv2DInputs validates the inputs for convolution operation
func validateConv2DInputs(input *tensor.FeatureMap, kernel *tensor.Kernel, bias []float32, config Conv2DConfig) error {
    if input == nil {
        return fmt.Errorf("input feature map is nil")
    }
    
    if kernel == nil {
        return fmt.Errorf("kernel is nil")
    }
    
    if len(bias) != kernel.Filters {
        return fmt.Errorf("bias length (%d) doesn't match kernel filters (%d)", len(bias), kernel.Filters)
    }
    
    if input.Channels != kernel.Channels {
        return fmt.Errorf("input channels (%d) don't match kernel channels (%d)", 
            input.Channels, kernel.Channels)
    }
    
    if config.Stride <= 0 {
        return fmt.Errorf("stride must be positive, got %d", config.Stride)
    }
    
    if config.Padding < 0 {
        return fmt.Errorf("padding must be non-negative, got %d", config.Padding)
    }
    
    // Check that output dimensions will be positive
    paddedHeight := input.Height + 2*config.Padding
    paddedWidth := input.Width + 2*config.Padding
    
    if paddedHeight < kernel.Size || paddedWidth < kernel.Size {
        return fmt.Errorf("input too small for kernel size after padding")
    }
    
    return nil
}

// GetConvOutputDims calculates the output dimensions for a convolution
// Useful for planning memory allocation and network architecture
func GetConvOutputDims(inputHeight, inputWidth, kernelSize, padding, stride int) (int, int) {
    outHeight := (inputHeight+2*padding-kernelSize)/stride + 1
    outWidth := (inputWidth+2*padding-kernelSize)/stride + 1
    return outHeight, outWidth
}

// Conv2DValid performs convolution with "valid" padding (no padding)
func Conv2DValid(input *tensor.FeatureMap, kernel *tensor.Kernel, bias []float32) *tensor.FeatureMap {
    config := Conv2DConfig{Padding: 0, Stride: 1}
    return Conv2D(input, kernel, bias, config)
}

// Conv2DSame performs convolution with "same" padding (output size = input size)
func Conv2DSame(input *tensor.FeatureMap, kernel *tensor.Kernel, bias []float32) *tensor.FeatureMap {
    // Calculate padding needed for same-size output
    padding := (kernel.Size - 1) / 2
    config := Conv2DConfig{Padding: padding, Stride: 1}
    return Conv2D(input, kernel, bias, config)
}

// Conv2DWithStride performs strided convolution (commonly used for downsampling)
func Conv2DWithStride(input *tensor.FeatureMap, kernel *tensor.Kernel, bias []float32, stride int) *tensor.FeatureMap {
    config := Conv2DConfig{Padding: 0, Stride: stride}
    return Conv2D(input, kernel, bias, config)
}