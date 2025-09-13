package main

import (
	"duchm1606/gocnn/internal/config"
	"fmt"
	"os"
	"path/filepath"
)

// InferenceConfig extends the base config with inference-specific settings
type InferenceConfig struct {
    *config.Config
    
    // Inference settings
    DefaultWeightsPath string   `yaml:"default_weights_path"`
    DefaultImagePath   string   `yaml:"default_image_path"`
    OutputFormat       string   `yaml:"output_format"`       // "text", "json", "csv"
    Precision          int      `yaml:"precision"`           // decimal places for output
    
    // Performance settings
    EnableProfiling    bool     `yaml:"enable_profiling"`
    ProfileOutputPath  string   `yaml:"profile_output_path"`
    MemoryLimit        int64    `yaml:"memory_limit"`        // bytes
    
    // Display settings
    ShowProbabilities  bool     `yaml:"show_probabilities"`
    ShowTiming         bool     `yaml:"show_timing"`
    ColorOutput        bool     `yaml:"color_output"`
}

// LoadInferenceConfig loads configuration with inference-specific defaults
func LoadInferenceConfig(configPath string) (*InferenceConfig, error) {
    // Load base config
    baseConfig, err := config.Load(configPath)
    if err != nil {
        return nil, fmt.Errorf("failed to load base config: %w", err)
    }
    
    // Create inference config with defaults
    infConfig := &InferenceConfig{
        Config:             baseConfig,
        OutputFormat:       "text",
        Precision:          4,
        EnableProfiling:    false,
        ShowProbabilities:  false,
        ShowTiming:         true,
        ColorOutput:        true,
    }
    
    // Load inference-specific settings if they exist
    // (This would extend the YAML loading to include inference settings)
    
    return infConfig, nil
}

// ValidateInferenceConfig validates the inference configuration
func ValidateInferenceConfig(cfg *InferenceConfig) error {
    // Validate base config
    if cfg.Config == nil {
        return fmt.Errorf("base configuration is nil")
    }
    
    // Validate inference-specific settings
    validFormats := map[string]bool{
        "text": true,
        "json": true,
        "csv":  true,
    }
    
    if !validFormats[cfg.OutputFormat] {
        return fmt.Errorf("invalid output format: %s", cfg.OutputFormat)
    }
    
    if cfg.Precision < 0 || cfg.Precision > 10 {
        return fmt.Errorf("precision must be between 0 and 10, got %d", cfg.Precision)
    }
    
    return nil
}

// CreateDefaultConfig creates a default configuration file
func CreateDefaultConfig(configPath string) error {
    // Create directory if it doesn't exist
    dir := filepath.Dir(configPath)
    err := os.MkdirAll(dir, 0755)
    if err != nil {
        return fmt.Errorf("failed to create config directory: %w", err)
    }
    
    // Default configuration content
    defaultConfig := `# GoCNN Inference Configuration
model_name: "TinyCNN-CIFAR10"
input_size: 32
input_channels: 3
num_classes: 10
class_names:
  - "Airplane"
  - "Automobile"
  - "Bird" 
  - "Cat"
  - "Deer"
  - "Dog"
  - "Frog"
  - "Horse"
  - "Ship"
  - "Truck"

# Inference settings
default_weights_path: "./weights"
default_image_path: "./test_images"
output_format: "text"
precision: 4

# Performance settings
enable_profiling: false
profile_output_path: "./profile.out"

# Display settings
show_probabilities: false
show_timing: true
color_output: true
`

    // Write configuration file
    err = os.WriteFile(configPath, []byte(defaultConfig), 0644)
    if err != nil {
        return fmt.Errorf("failed to write config file: %w", err)
    }
    
    return nil
}

// PrintConfigInfo displays information about the current configuration
func PrintConfigInfo(cfg *InferenceConfig) {
    fmt.Println("Configuration Information:")
    fmt.Printf("  Model: %s\n", cfg.Model.Name)
    fmt.Printf("  Input size: %d×%d×%d\n", cfg.Model.InputHeight, cfg.Model.InputWidth, cfg.Model.InputChannels)
    fmt.Printf("  Output classes: %d\n", cfg.Model.NumClasses)
    fmt.Printf("  Output format: %s\n", cfg.OutputFormat)
    fmt.Printf("  Precision: %d decimal places\n", cfg.Precision)
    fmt.Printf("  Profiling: %v\n", cfg.EnableProfiling)
    fmt.Printf("  Show probabilities: %v\n", cfg.ShowProbabilities)
    fmt.Printf("  Show timing: %v\n", cfg.ShowTiming)
    fmt.Printf("  Color output: %v\n", cfg.ColorOutput)
}