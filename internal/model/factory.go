package model

import (
	"duchm1606/gocnn/internal/config"
	"fmt"
	"os"
	"path/filepath"
)

// ModelFactory provides methods to create and configure TinyCNN models
type ModelFactory struct {
    config *config.Config
}

// NewModelFactory creates a new model factory
func NewModelFactory(cfg *config.Config) *ModelFactory {
    return &ModelFactory{
        config: cfg,
    }
}

// CreateModel creates a new TinyCNN model with the given configuration
func (mf *ModelFactory) CreateModel(weightsPath string) (*TinyCNN, error) {
    // Validate weights directory
    err := validateWeightsDirectory(weightsPath)
    if err != nil {
        return nil, fmt.Errorf("weights directory validation failed: %w", err)
    }
    
    // Create model
    model, err := NewTinyCNN(weightsPath)
    if err != nil {
        return nil, fmt.Errorf("failed to create model: %w", err)
    }
    
    // Validate loaded model
    err = model.ValidateModel()
    if err != nil {
        return nil, fmt.Errorf("model validation failed: %w", err)
    }
    
    return model, nil
}

// CreateModelFromConfig creates a model using paths from configuration
func (mf *ModelFactory) CreateModelFromConfig() (*TinyCNN, error) {
    if mf.config == nil {
        return nil, fmt.Errorf("no configuration provided")
    }
    
    // Use weights path from config (assuming it's added to config structure)
    weightsPath := "./weights" // Default path, should come from config
    
    return mf.CreateModel(weightsPath)
}

// LoadPretrainedModel loads a model with pre-trained weights
func LoadPretrainedModel(weightsPath string) (*TinyCNN, error) {
    factory := &ModelFactory{}
    return factory.CreateModel(weightsPath)
}

// validateWeightsDirectory checks if all required weight files exist
func validateWeightsDirectory(weightsPath string) error {
    if _, err := os.Stat(weightsPath); os.IsNotExist(err) {
        return fmt.Errorf("weights directory does not exist: %s", weightsPath)
    }
    
    // Check for required files
    requiredFiles := []string{
        "conv1_weight.bin", "conv1_bias.bin",
        "conv2_weight.bin", "conv2_bias.bin",
        "conv3_weight.bin", "conv3_bias.bin",
        "conv4_weight.bin", "conv4_bias.bin",
        "conv5_weight.bin", "conv5_bias.bin",
        "conv6_weight.bin", "conv6_bias.bin",
        "conv7_weight.bin", "conv7_bias.bin",
    }
    
    for _, filename := range requiredFiles {
        fullPath := filepath.Join(weightsPath, filename)
        if _, err := os.Stat(fullPath); os.IsNotExist(err) {
            return fmt.Errorf("required weight file not found: %s", filename)
        }
    }
    
    return nil
}

// ModelConfig holds configuration for model creation
type ModelConfig struct {
    WeightsPath      string
    EnableBatchNorm  bool
    UseParallelConv  bool
    NumWorkers       int
    ValidationMode   bool
}

// CreateConfiguredModel creates a model with specific configuration
func CreateConfiguredModel(config ModelConfig) (*TinyCNN, error) {
    // Create basic model
    model, err := NewTinyCNN(config.WeightsPath)
    if err != nil {
        return nil, err
    }
    
    // Configure convolution engine
    if config.NumWorkers > 0 {
        model.convEngine.NumWorkers = config.NumWorkers
    }
    model.convEngine.UseParallel = config.UseParallelConv
    
    // Perform additional validation if requested
    if config.ValidationMode {
        err = performExtendedValidation(model)
        if err != nil {
            return nil, fmt.Errorf("extended validation failed: %w", err)
        }
    }
    
    return model, nil
}

// performExtendedValidation performs thorough model validation
func performExtendedValidation(model *TinyCNN) error {
    // Test with a dummy input
    dummyInput := make([]float32, 32*32*3)
    for i := range dummyInput {
        dummyInput[i] = 0.5 // Neutral test input
    }
    
    // Perform test inference
    _, err := model.Predict(dummyInput)
    if err != nil {
        return fmt.Errorf("test inference failed: %w", err)
    }
    
    return nil
}

// GetSupportedArchitectures returns a list of supported model architectures
func GetSupportedArchitectures() []string {
    return []string{
        "tinycnn-cifar10",
        // Add more architectures as they are implemented
    }
}

// CreateArchitectureByName creates a model with a specific architecture
func CreateArchitectureByName(archName, weightsPath string) (*TinyCNN, error) {
    switch archName {
    case "tinycnn-cifar10":
        return NewTinyCNN(weightsPath)
    default:
        return nil, fmt.Errorf("unsupported architecture: %s", archName)
    }
}