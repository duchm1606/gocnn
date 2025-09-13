package data

import (
	"duchm1606/gocnn/internal/tensor"
	"fmt"
)

// DataManager provides a unified interface for all data loading operations
type DataManager struct {
    weightLoader *WeightLoader
    imageLoader  *ImageLoader
    labelLoader  *LabelLoader
}

// NewDataManager creates a new data manager
func NewDataManager(weightsPath string, imageFormat ImageFormat, labelFormat LabelFormat) *DataManager {
    return &DataManager{
        weightLoader: NewWeightLoader(weightsPath),
        imageLoader:  NewImageLoader(imageFormat),
        labelLoader:  NewLabelLoader(labelFormat),
    }
}

// DataBatch represents a batch of data for training or testing
type DataBatch struct {
    Images []*tensor.FeatureMap
    Labels [][]int
    Size   int
}

// ModelWeights holds all the weights for the CNN model
type ModelWeights struct {
    Kernels    []*tensor.Kernel
    Biases     [][]float32
    BatchNorms []*BatchNormParams
}

// LoadModelWeights loads all model weights from the weights directory
func (dm *DataManager) LoadModelWeights() (*ModelWeights, error) {
    weights := &ModelWeights{
        Kernels:    make([]*tensor.Kernel, 0),
        Biases:     make([][]float32, 0),
        BatchNorms: make([]*BatchNormParams, 0),
    }
    
    // Load all conv layers (adjust based on your model architecture)
	// TODO: modify
    layerConfigs := []struct {
        name     string
        size     int
        channels int
        filters  int
    }{
        {"conv1", 3, 3, 32},
        {"conv2", 3, 32, 32},
        {"conv3", 3, 32, 64},
        {"conv4", 3, 64, 64},
        {"conv5", 3, 64, 128},
        {"conv6", 3, 128, 128},
        {"conv7", 1, 128, 10},
    }
    
    for i, config := range layerConfigs {
        // Load kernel
        kernelFile := fmt.Sprintf("%s_weight.bin", config.name)
        kernel, err := dm.weightLoader.LoadKernel(kernelFile, config.size, config.channels, config.filters)
        if err != nil {
            return nil, fmt.Errorf("failed to load kernel for %s: %w", config.name, err)
        }
        weights.Kernels = append(weights.Kernels, kernel)
        
        // Load bias
        biasFile := fmt.Sprintf("%s_bias.bin", config.name)
        bias, err := dm.weightLoader.LoadBias(biasFile, config.filters)
        if err != nil {
            return nil, fmt.Errorf("failed to load bias for %s: %w", config.name, err)
        }
        weights.Biases = append(weights.Biases, bias)
        
        // Load batch normalization parameters (except for last layer)
        if i < len(layerConfigs)-1 {
            bnName := fmt.Sprintf("batchnorm%d/bn%d", i+1, i+1)
            bn, err := dm.weightLoader.LoadBatchNormParams(bnName, config.filters)
            if err != nil {
                return nil, fmt.Errorf("failed to load batch norm for %s: %w", config.name, err)
            }
            weights.BatchNorms = append(weights.BatchNorms, bn)
        }
    }
    
    return weights, nil
}

// LoadTestBatch loads a batch of test data
func (dm *DataManager) LoadTestBatch(imageDir, labelDir string, batchSize, height, width, channels, numClasses int) (*DataBatch, error) {
    // Load images
    images, err := dm.imageLoader.LoadImageBatch(imageDir, batchSize, height, width, channels)
    if err != nil {
        return nil, fmt.Errorf("failed to load images: %w", err)
    }
    
    // Load labels
    labels, err := dm.labelLoader.LoadLabelBatch(labelDir, batchSize, numClasses)
    if err != nil {
        return nil, fmt.Errorf("failed to load labels: %w", err)
    }
    
    return &DataBatch{
        Images: images,
        Labels: labels,
        Size:   batchSize,
    }, nil
}

// ValidateDataBatch checks if a data batch is valid
func (dm *DataManager) ValidateDataBatch(batch *DataBatch, expectedHeight, expectedWidth, expectedChannels, expectedClasses int) error {
    if batch == nil {
        return fmt.Errorf("data batch is nil")
    }
    
    if len(batch.Images) != len(batch.Labels) {
        return fmt.Errorf("number of images (%d) doesn't match number of labels (%d)", 
            len(batch.Images), len(batch.Labels))
    }
    
    if len(batch.Images) != batch.Size {
        return fmt.Errorf("batch size mismatch: expected %d, got %d", batch.Size, len(batch.Images))
    }
    
    // Validate each image
    for i, image := range batch.Images {
        if image.Height != expectedHeight || image.Width != expectedWidth || image.Channels != expectedChannels {
            return fmt.Errorf("image %d has wrong dimensions: (%d,%d,%d), expected (%d,%d,%d)", 
                i, image.Height, image.Width, image.Channels, expectedHeight, expectedWidth, expectedChannels)
        }
        
        err := tensor.ValidateFeatureMap(image)
        if err != nil {
            return fmt.Errorf("image %d failed validation: %w", i, err)
        }
    }
    
    // Validate each label
    for i, label := range batch.Labels {
        err := ValidateLabel(label, expectedClasses)
        if err != nil {
            return fmt.Errorf("label %d failed validation: %w", i, err)
        }
    }
    
    return nil
}

// GetDataInfo returns information about the loaded data
func (dm *DataManager) GetDataInfo(weightsPath, imageDir, labelDir string) (*DataInfo, error) {
    info := &DataInfo{}
    
    // Get weight files info
    weightFiles, err := GetWeightFilesInfo(weightsPath)
    if err != nil {
        return nil, fmt.Errorf("failed to get weight files info: %w", err)
    }
    info.WeightFiles = weightFiles
    
    // Get image files info
    imageFiles, err := GetImageFilesInfo(imageDir)
    if err != nil {
        return nil, fmt.Errorf("failed to get image files info: %w", err)
    }
    info.ImageFiles = len(imageFiles)
    
    // Calculate total data size
    var totalSize int64
    for _, size := range weightFiles {
        totalSize += size
    }
    for _, file := range imageFiles {
        totalSize += file.Size()
    }
    info.TotalDataSize = totalSize
    
    return info, nil
}

// DataInfo holds information about the dataset
type DataInfo struct {
    WeightFiles   map[string]int64
    ImageFiles    int
    TotalDataSize int64
}

// Print displays data information in a readable format
func (info *DataInfo) Print() {
    fmt.Println("Dataset Information:")
    fmt.Printf("  Weight files: %d\n", len(info.WeightFiles))
    fmt.Printf("  Image files: %d\n", info.ImageFiles)
    fmt.Printf("  Total size: %.2f MB\n", float64(info.TotalDataSize)/(1024*1024))
    
    fmt.Println("Weight files:")
    for filename, size := range info.WeightFiles {
        fmt.Printf("  %s: %.2f KB\n", filename, float64(size)/1024)
    }
}