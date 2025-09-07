package tensor

import (
	"fmt"
	"math/rand/v2"
)

// Kernel represents convolution weights in 4D format
// Data layout: [filter][channel][height][width] in row-major order
type Kernel struct {
    Size     int       // Kernel size (e.g., 3 for 3x3)
    Channels int       // Number of input channels
    Filters  int       // Number of output filters
    Weights  []float32 // Flat array: len = size * size * channels * filters
}

// NewKernel creates a new kernel with given dimensions
func NewKernel(size, channels, filters int) *Kernel {
    return &Kernel{
        Size:     size,
        Channels: channels,
        Filters:  filters,
        Weights:  make([]float32, size*size*channels*filters),
    }
}

// NewKernelFromData creates a kernel from existing weight data
func NewKernelFromData(weights []float32, size, channels, filters int) (*Kernel, error) {
    expectedSize := size * size * channels * filters
    if len(weights) != expectedSize {
        return nil, fmt.Errorf("weights size mismatch: expected %d, got %d", expectedSize, len(weights))
    }
    
    kernel := &Kernel{
        Size:     size,
        Channels: channels,
        Filters:  filters,
        Weights:  make([]float32, expectedSize),
    }
    
    copy(kernel.Weights, weights)
    return kernel, nil
}

// GetWeight returns the weight at the specified position
// Formula: index = f*channels*size*size + c*size*size + h*size + w
func (k *Kernel) GetWeight(f, c, h, w int) float32 {
    if !k.isValidIndex(f, c, h, w) {
        panic(fmt.Sprintf("kernel index out of bounds: (%d,%d,%d,%d) for shape (%d,%d,%d,%d)", 
            f, c, h, w, k.Filters, k.Channels, k.Size, k.Size))
    }
    return k.Weights[f*k.Channels*k.Size*k.Size + c*k.Size*k.Size + h*k.Size + w]
}

// SetWeight sets the weight at the specified position
func (k *Kernel) SetWeight(f, c, h, w int, value float32) {
    if !k.isValidIndex(f, c, h, w) {
        panic(fmt.Sprintf("kernel index out of bounds: (%d,%d,%d,%d) for shape (%d,%d,%d,%d)", 
            f, c, h, w, k.Filters, k.Channels, k.Size, k.Size))
    }
    k.Weights[f*k.Channels*k.Size*k.Size + c*k.Size*k.Size + h*k.Size + w] = value
}

// GetWeightUnsafe returns the weight without bounds checking
func (k *Kernel) GetWeightUnsafe(f, c, h, w int) float32 {
    return k.Weights[f*k.Channels*k.Size*k.Size + c*k.Size*k.Size + h*k.Size + w]
}

// SetWeightUnsafe sets the weight without bounds checking
func (k *Kernel) SetWeightUnsafe(f, c, h, w int, value float32) {
    k.Weights[f*k.Channels*k.Size*k.Size + c*k.Size*k.Size + h*k.Size + w] = value
}

// isValidIndex checks if the given indices are within bounds
func (k *Kernel) isValidIndex(f, c, h, w int) bool {
    return f >= 0 && f < k.Filters &&
           c >= 0 && c < k.Channels &&
           h >= 0 && h < k.Size &&
           w >= 0 && w < k.Size
}

// Clone creates a deep copy of the kernel
func (k *Kernel) Clone() *Kernel {
    clone := NewKernel(k.Size, k.Channels, k.Filters)
    copy(clone.Weights, k.Weights)
    return clone
}

// Zero fills the kernel with zeros
func (k *Kernel) Zero() {
    for i := range k.Weights {
        k.Weights[i] = 0
    }
}

// RandomFill fills the kernel with random values for testing
func (k *Kernel) RandomFill() {
    for i := range k.Weights {
        k.Weights[i] = rand.Float32()*2 - 1 // Random values between -1 and 1
    }
}

// Shape returns the dimensions as a slice [filters, channels, size, size]
func (k *Kernel) Shape() []int {
    return []int{k.Filters, k.Channels, k.Size, k.Size}
}

// TotalWeights returns the total number of weights
func (k *Kernel) TotalWeights() int {
    return len(k.Weights)
}

// String provides a string representation
func (k *Kernel) String() string {
    return fmt.Sprintf("Kernel{Size: %d, Channels: %d, Filters: %d, Weights: %d}", 
        k.Size, k.Channels, k.Filters, k.TotalWeights())
}