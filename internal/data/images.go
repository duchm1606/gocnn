package data

import (
	"duchm1606/gocnn/internal/tensor"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ImageLoader handles loading of image data
type ImageLoader struct {
    imageFormat ImageFormat
    byteOrder   binary.ByteOrder
}

// ImageFormat specifies the format of image files
type ImageFormat int

const (
    BinaryFloat32 ImageFormat = iota // 32x32x3 float32 values
    BinaryUint8                      // 32x32x3 uint8 values (0-255)
)

// NewImageLoader creates a new image loader
func NewImageLoader(format ImageFormat) *ImageLoader {
    return &ImageLoader{
        imageFormat: format,
        byteOrder:   binary.LittleEndian,
    }
}

// LoadImage loads a single image from a binary file
func (il *ImageLoader) LoadImage(filename string, height, width, channels int) (*tensor.FeatureMap, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to open image file %s: %w", filename, err)
    }
    defer file.Close()
    
    // Validate file size based on format
    fileInfo, err := file.Stat()
    if err != nil {
        return nil, fmt.Errorf("failed to get file info for %s: %w", filename, err)
    }
    
    var expectedBytes int64
    switch il.imageFormat {
    case BinaryFloat32:
        expectedBytes = int64(height * width * channels * 4) // 4 bytes per float32
    case BinaryUint8:
        expectedBytes = int64(height * width * channels * 1) // 1 byte per uint8
    default:
        return nil, fmt.Errorf("unsupported image format: %d", il.imageFormat)
    }
    
    if fileInfo.Size() != expectedBytes {
        return nil, fmt.Errorf("image file %s has wrong size: expected %d bytes, got %d bytes", 
            filename, expectedBytes, fileInfo.Size())
    }
    
    // Create feature map
    fm := tensor.NewFeatureMap(height, width, channels)
    
    // Load data based on format
    switch il.imageFormat {
    case BinaryFloat32:
        err = il.loadFloat32Image(file, fm)
    case BinaryUint8:
        err = il.loadUint8Image(file, fm)
    }
    
    if err != nil {
        return nil, fmt.Errorf("failed to load image data from %s: %w", filename, err)
    }
    
    // Validate loaded image
    err = tensor.ValidateFeatureMap(fm)
    if err != nil {
        return nil, fmt.Errorf("loaded image failed validation: %w", err)
    }
    
    return fm, nil
}

// loadFloat32Image loads image data as float32 values
func (il *ImageLoader) loadFloat32Image(file *os.File, fm *tensor.FeatureMap) error {
    // Read data in HWC order (height, width, channels)
    for h := 0; h < fm.Height; h++ {
        for w := 0; w < fm.Width; w++ {
            for c := 0; c < fm.Channels; c++ {
                var pixel float32
                err := binary.Read(file, il.byteOrder, &pixel)
                if err != nil {
                    return fmt.Errorf("failed to read pixel at (%d,%d,%d): %w", h, w, c, err)
                }
                
                fm.SetUnsafe(c, h, w, pixel)
            }
        }
    }
    
    return nil
}

// loadUint8Image loads image data as uint8 values and converts to float32
func (il *ImageLoader) loadUint8Image(file *os.File, fm *tensor.FeatureMap) error {
    // Read data in HWC order
    for h := 0; h < fm.Height; h++ {
        for w := 0; w < fm.Width; w++ {
            for c := 0; c < fm.Channels; c++ {
                var pixel uint8
                err := binary.Read(file, il.byteOrder, &pixel)
                if err != nil {
                    return fmt.Errorf("failed to read pixel at (%d,%d,%d): %w", h, w, c, err)
                }
                
                // Convert to float32 and normalize to [0, 1]
                normalizedPixel := float32(pixel) / 255.0
                fm.SetUnsafe(c, h, w, normalizedPixel)
            }
        }
    }
    
    return nil
}

// LoadImageBatch loads multiple images from a directory
func (il *ImageLoader) LoadImageBatch(imageDir string, numImages, height, width, channels int) ([]*tensor.FeatureMap, error) {
    images := make([]*tensor.FeatureMap, numImages)
    
    for i := 0; i < numImages; i++ {
        filename := filepath.Join(imageDir, fmt.Sprintf("test_img_%d.bin", i))
        
        image, err := il.LoadImage(filename, height, width, channels)
        if err != nil {
            return nil, fmt.Errorf("failed to load image %d: %w", i, err)
        }
        
        images[i] = image
    }
    
    return images, nil
}

// PreprocessImage applies common preprocessing operations
func (il *ImageLoader) PreprocessImage(fm *tensor.FeatureMap, config PreprocessConfig) *tensor.FeatureMap {
    result := fm.Clone()
    
    // Apply normalization
    if config.Normalize {
        il.normalizeImage(result, config.Mean, config.Std)
    }
    
    // Apply other preprocessing as needed
    
    return result
}

// PreprocessConfig holds image preprocessing configuration
type PreprocessConfig struct {
    Normalize bool        // Whether to apply normalization
    Mean      []float32   // Mean values for each channel
    Std       []float32   // Standard deviation for each channel
}

// normalizeImage applies per-channel normalization: (pixel - mean) / std
func (il *ImageLoader) normalizeImage(fm *tensor.FeatureMap, mean, std []float32) {
    if len(mean) != fm.Channels || len(std) != fm.Channels {
        panic("Mean and std arrays must match number of channels")
    }
    
    for c := 0; c < fm.Channels; c++ {
        channelMean := mean[c]
        channelStd := std[c]
        
        for h := 0; h < fm.Height; h++ {
            for w := 0; w < fm.Width; w++ {
                pixel := fm.GetUnsafe(c, h, w)
                normalized := (pixel - channelMean) / channelStd
                fm.SetUnsafe(c, h, w, normalized)
            }
        }
    }
}

// SaveImage saves a feature map as a binary image file
func (il *ImageLoader) SaveImage(fm *tensor.FeatureMap, filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return fmt.Errorf("failed to create file %s: %w", filename, err)
    }
    defer file.Close()
    
    // Write data in HWC order
    for h := 0; h < fm.Height; h++ {
        for w := 0; w < fm.Width; w++ {
            for c := 0; c < fm.Channels; c++ {
                pixel := fm.GetUnsafe(c, h, w)
                
                switch il.imageFormat {
                case BinaryFloat32:
                    err := binary.Write(file, il.byteOrder, pixel)
                    if err != nil {
                        return fmt.Errorf("failed to write pixel: %w", err)
                    }
                case BinaryUint8:
                    // Convert float32 to uint8
                    pixel8 := uint8(pixel * 255.0)
                    err := binary.Write(file, il.byteOrder, pixel8)
                    if err != nil {
                        return fmt.Errorf("failed to write pixel: %w", err)
                    }
                }
            }
        }
    }
    
    return nil
}

// GetImageFilesInfo returns information about image files in a directory
func GetImageFilesInfo(imageDir string) ([]os.FileInfo, error) {
    files, err := os.ReadDir(imageDir)
    if err != nil {
        return nil, fmt.Errorf("failed to read image directory %s: %w", imageDir, err)
    }
    
    var imageFiles []os.FileInfo
    for _, file := range files {
        if !file.IsDir() && (strings.HasSuffix(file.Name(), ".bin") || strings.HasSuffix(file.Name(), ".dat")) {
            info, err := file.Info()
            if err != nil {
                continue
            }
            imageFiles = append(imageFiles, info)
        }
    }
    
    return imageFiles, nil
}