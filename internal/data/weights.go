package data

import (
	"duchm1606/gocnn/internal/tensor"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
)

// WeightLoader handles loading of model weights from files
type WeightLoader struct {
    weightsPath string
    byteOrder   binary.ByteOrder
}

// NewWeightLoader creates a new weight loader
func NewWeightLoader(weightsPath string) *WeightLoader {
    return &WeightLoader{
        weightsPath: weightsPath,
        byteOrder:   binary.LittleEndian, // Match original C implementation
    }
}

// LoadKernel loads convolution kernel weights from a binary file
func (wl *WeightLoader) LoadKernel(filename string, size, channels, filters int) (*tensor.Kernel, error) {
    fullPath := filepath.Join(wl.weightsPath, filename)
    
    // Open file
    file, err := os.Open(fullPath)
    if err != nil {
        return nil, fmt.Errorf("failed to open kernel file %s: %w", fullPath, err)
    }
    defer file.Close()
    
    // Get file size
    fileInfo, err := file.Stat()
    if err != nil {
        return nil, fmt.Errorf("failed to get file info for %s: %w", fullPath, err)
    }
    
    // Validate file size
    expectedElements := size * size * channels * filters
    expectedBytes := int64(expectedElements * 4) // 4 bytes per float32
    
    if fileInfo.Size() != expectedBytes {
        return nil, fmt.Errorf("kernel file %s has wrong size: expected %d bytes, got %d bytes", 
            filename, expectedBytes, fileInfo.Size())
    }
    
    // Create kernel
    kernel := tensor.NewKernel(size, channels, filters)
    
    // Read weights in the correct order: [size][size][channels][filters]
    // This matches the original C implementation's file format
    for h := 0; h < size; h++ {
        for w := 0; w < size; w++ {
            for c := 0; c < channels; c++ {
                for f := 0; f < filters; f++ {
                    var weight float32
                    err := binary.Read(file, wl.byteOrder, &weight)
                    if err != nil {
                        return nil, fmt.Errorf("failed to read weight at (%d,%d,%d,%d) from %s: %w", 
                            h, w, c, f, filename, err)
                    }
                    
                    kernel.SetWeight(f, c, h, w, weight)
                }
            }
        }
    }
    
    // Validate loaded kernel
    err = tensor.ValidateKernel(kernel)
    if err != nil {
        return nil, fmt.Errorf("loaded kernel failed validation: %w", err)
    }
    
    return kernel, nil
}

// LoadBias loads bias values from a binary file
func (wl *WeightLoader) LoadBias(filename string, filters int) ([]float32, error) {
    fullPath := filepath.Join(wl.weightsPath, filename)
    
    file, err := os.Open(fullPath)
    if err != nil {
        return nil, fmt.Errorf("failed to open bias file %s: %w", fullPath, err)
    }
    defer file.Close()
    
    // Validate file size
    fileInfo, err := file.Stat()
    if err != nil {
        return nil, fmt.Errorf("failed to get file info for %s: %w", fullPath, err)
    }
    
    expectedBytes := int64(filters * 4) // 4 bytes per float32
    if fileInfo.Size() != expectedBytes {
        return nil, fmt.Errorf("bias file %s has wrong size: expected %d bytes, got %d bytes", 
            filename, expectedBytes, fileInfo.Size())
    }
    
    // Load bias values
    bias := make([]float32, filters)
    err = binary.Read(file, wl.byteOrder, bias)
    if err != nil {
        return nil, fmt.Errorf("failed to read bias from %s: %w", filename, err)
    }
    
    return bias, nil
}

// LoadBatchNormParams loads batch normalization parameters
func (wl *WeightLoader) LoadBatchNormParams(layerName string, channels int) (*BatchNormParams, error) {
    params := &BatchNormParams{
        Mean:     make([]float32, channels),
        Variance: make([]float32, channels),
        Scale:    make([]float32, channels),
        Shift:    make([]float32, channels),
        Epsilon:  1e-5,
    }
    
    // Load mean
    meanFile := fmt.Sprintf("%s_moving_mean.bin", layerName)
    mean, err := wl.loadFloatArray(meanFile, channels)
    if err != nil {
        return nil, fmt.Errorf("failed to load mean for %s: %w", layerName, err)
    }
    copy(params.Mean, mean)
    
    // Load variance
    varianceFile := fmt.Sprintf("%s_moving_variance.bin", layerName)
    variance, err := wl.loadFloatArray(varianceFile, channels)
    if err != nil {
        return nil, fmt.Errorf("failed to load variance for %s: %w", layerName, err)
    }
    copy(params.Variance, variance)
    
    // Load scale (gamma)
    scaleFile := fmt.Sprintf("%s_gamma.bin", layerName)
    scale, err := wl.loadFloatArray(scaleFile, channels)
    if err != nil {
        return nil, fmt.Errorf("failed to load scale for %s: %w", layerName, err)
    }
    copy(params.Scale, scale)
    
    // Load shift (beta)
    shiftFile := fmt.Sprintf("%s_beta.bin", layerName)
    shift, err := wl.loadFloatArray(shiftFile, channels)
    if err != nil {
        return nil, fmt.Errorf("failed to load shift for %s: %w", layerName, err)
    }
    copy(params.Shift, shift)
    
    return params, nil
}

// loadFloatArray is a helper function to load an array of floats
func (wl *WeightLoader) loadFloatArray(filename string, size int) ([]float32, error) {
    fullPath := filepath.Join(wl.weightsPath, filename)
    
    file, err := os.Open(fullPath)
    if err != nil {
        return nil, fmt.Errorf("failed to open file %s: %w", fullPath, err)
    }
    defer file.Close()
    
    // Validate file size
    fileInfo, err := file.Stat()
    if err != nil {
        return nil, fmt.Errorf("failed to get file info for %s: %w", fullPath, err)
    }
    
    expectedBytes := int64(size * 4)
    if fileInfo.Size() != expectedBytes {
        return nil, fmt.Errorf("file %s has wrong size: expected %d bytes, got %d bytes", 
            filename, expectedBytes, fileInfo.Size())
    }
    
    // Load data
    data := make([]float32, size)
    err = binary.Read(file, wl.byteOrder, data)
    if err != nil {
        return nil, fmt.Errorf("failed to read data from %s: %w", filename, err)
    }
    
    return data, nil
}

// BatchNormParams holds batch normalization parameters
type BatchNormParams struct {
    Mean     []float32
    Variance []float32
    Scale    []float32
    Shift    []float32
    Epsilon  float32
}

// ValidateWeightsDirectory checks if the weights directory contains required files
func ValidateWeightsDirectory(weightsPath string) error {
    // Check if directory exists
    info, err := os.Stat(weightsPath)
    if err != nil {
        return fmt.Errorf("weights directory does not exist: %s", weightsPath)
    }
    
    if !info.IsDir() {
        return fmt.Errorf("weights path is not a directory: %s", weightsPath)
    }
    
    // Check for required files (basic validation)
	// TODO: Check for the correct files
    requiredFiles := []string{
        "conv1_weight.bin",
        "conv1_bias.bin",
        "conv2_weight.bin",
        "conv2_bias.bin",
        // Add more files as needed
    }
    
    for _, filename := range requiredFiles {
        fullPath := filepath.Join(weightsPath, filename)
        if _, err := os.Stat(fullPath); os.IsNotExist(err) {
            return fmt.Errorf("required weight file not found: %s", filename)
        }
    }
    
    return nil
}

// GetWeightFilesInfo returns information about weight files in a directory
func GetWeightFilesInfo(weightsPath string) (map[string]int64, error) {
    files := make(map[string]int64)
    
    err := filepath.Walk(weightsPath, func(path string, info os.FileInfo, err error) error {
        if err != nil {
            return err
        }
        
        if !info.IsDir() && filepath.Ext(path) == ".bin" {
            relPath, _ := filepath.Rel(weightsPath, path)
            files[relPath] = info.Size()
        }
        
        return nil
    })
    
    return files, err
}