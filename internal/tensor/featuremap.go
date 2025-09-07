package tensor

import (
	"fmt"
	"math/rand/v2"
)

/**
* **Learning Note**: Why flat arrays?
- **Cache Performance**: Contiguous memory layout improves cache hits
- **Simpler Memory Management**: Single allocation instead of nested allocations
- **Better for Vectorization**: Easier to apply SIMD optimizations later
- **No Pointer Chasing**: Direct indexing is faster than pointer dereferencing
*/

// FeatureMap represents a 3D tensor for CNN feature maps
// Data layout: [channel][height][width] in row-major order
type FeatureMap struct {
    Height   int       // Height dimension
    Width    int       // Width dimension  
    Channels int       // Number of channels
    Data     []float32 // Flat array: len = height * width * channels
}

// NewFeatureMap creates a new feature map with given dimensions
func NewFeatureMap(height, width, channels int) *FeatureMap {
    return &FeatureMap{
        Height:   height,
        Width:    width,
        Channels: channels,
        Data:     make([]float32, height*width*channels),
    }
}

// NewFeatureMapFromData creates a feature map from existing data
func NewFeatureMapFromData(data []float32, height, width, channels int) (*FeatureMap, error) {
    expectedSize := height * width * channels
    if len(data) != expectedSize {
        return nil, fmt.Errorf("data size mismatch: expected %d, got %d", expectedSize, len(data))
    }
    
    fm := &FeatureMap{
        Height:   height,
        Width:    width,
        Channels: channels,
        Data:     make([]float32, expectedSize),
    }
    
    copy(fm.Data, data)
    return fm, nil
}

// Get returns the value at the specified position
// Formula: index = c*height*width + h*width + w
func (fm *FeatureMap) Get(c, h, w int) float32 {
    if !fm.isValidIndex(c, h, w) {
        panic(fmt.Sprintf("index out of bounds: (%d,%d,%d) for shape (%d,%d,%d)", 
            c, h, w, fm.Channels, fm.Height, fm.Width))
    }
    return fm.Data[c*fm.Height*fm.Width + h*fm.Width + w]
}

// Set sets the value at the specified position
func (fm *FeatureMap) Set(c, h, w int, value float32) {
    if !fm.isValidIndex(c, h, w) {
        panic(fmt.Sprintf("index out of bounds: (%d,%d,%d) for shape (%d,%d,%d)", 
            c, h, w, fm.Channels, fm.Height, fm.Width))
    }
    fm.Data[c*fm.Height*fm.Width + h*fm.Width + w] = value
}

// GetUnsafe returns the value without bounds checking (for performance)
func (fm *FeatureMap) GetUnsafe(c, h, w int) float32 {
    return fm.Data[c*fm.Height*fm.Width + h*fm.Width + w]
}

// SetUnsafe sets the value without bounds checking (for performance)
func (fm *FeatureMap) SetUnsafe(c, h, w int, value float32) {
    fm.Data[c*fm.Height*fm.Width + h*fm.Width + w] = value
}

// isValidIndex checks if the given indices are within bounds
func (fm *FeatureMap) isValidIndex(c, h, w int) bool {
    return c >= 0 && c < fm.Channels && 
           h >= 0 && h < fm.Height && 
           w >= 0 && w < fm.Width
}

// Clone creates a deep copy of the feature map
func (fm *FeatureMap) Clone() *FeatureMap {
    clone := NewFeatureMap(fm.Height, fm.Width, fm.Channels)
    copy(clone.Data, fm.Data)
    return clone
}

// Zero fills the feature map with zeros
func (fm *FeatureMap) Zero() {
    for i := range fm.Data {
        fm.Data[i] = 0
    }
}

// Fill fills the feature map with a constant value
func (fm *FeatureMap) Fill(value float32) {
    for i := range fm.Data {
        fm.Data[i] = value
    }
}

// RandomFill fills the feature map with random values for testing
func (fm *FeatureMap) RandomFill() {
    for i := range fm.Data {
        fm.Data[i] = rand.Float32()
    }
}

// Shape returns the dimensions as a slice [height, width, channels]
func (fm *FeatureMap) Shape() []int {
    return []int{fm.Height, fm.Width, fm.Channels}
}

// Size returns the total number of elements
func (fm *FeatureMap) Size() int {
    return len(fm.Data)
}

// String provides a string representation (for debugging)
func (fm *FeatureMap) String() string {
    return fmt.Sprintf("FeatureMap{Height: %d, Width: %d, Channels: %d, Size: %d}", 
        fm.Height, fm.Width, fm.Channels, fm.Size())
}