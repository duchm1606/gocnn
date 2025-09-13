package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// Config holds the model configuration
// Config holds the complete model configuration
type Config struct {
    Model     ModelConfig     `yaml:"model"`
    Data      DataConfig      `yaml:"data"`
    Inference InferenceConfig `yaml:"inference"`
    Benchmark BenchmarkConfig `yaml:"benchmark"`
}

// ModelConfig defines model-specific settings
type ModelConfig struct {
    Name          string        `yaml:"name"`
    Architecture  string        `yaml:"architecture"`
    WeightsPath   string        `yaml:"weights_path"`
    InputHeight   int           `yaml:"input_height"`
    InputWidth    int           `yaml:"input_width"`
    InputChannels int           `yaml:"input_channels"`
    NumClasses    int           `yaml:"num_classes"`
    ClassNames    []string      `yaml:"class_names"`
    Layers        []LayerConfig `yaml:"layers"`
}

// LayerConfig defines configuration for individual layers
type LayerConfig struct {
    Name            string `yaml:"name"`
    Type            string `yaml:"type"`
    KernelSize      int    `yaml:"kernel_size,omitempty"`
    Filters         int    `yaml:"filters,omitempty"`
    Stride          int    `yaml:"stride,omitempty"`
    Padding         int    `yaml:"padding,omitempty"`
    PoolSize        int    `yaml:"pool_size,omitempty"`
    PoolStride      int    `yaml:"pool_stride,omitempty"`
    ApplyBatchNorm  bool   `yaml:"apply_batch_norm,omitempty"`
    ApplyActivation bool   `yaml:"apply_activation,omitempty"`
}

// DataConfig defines data loading settings
type DataConfig struct {
    Format      string `yaml:"format"`
    Precision   string `yaml:"precision"`
    Normalize   bool   `yaml:"normalize"`
    MeanValues  []float32 `yaml:"mean_values,omitempty"`
    StdValues   []float32 `yaml:"std_values,omitempty"`
}

// InferenceConfig defines inference-specific settings
type InferenceConfig struct {
    BatchSize     int    `yaml:"batch_size"`
    UseParallel   bool   `yaml:"use_parallel"`
    NumWorkers    int    `yaml:"num_workers"`
    OutputFormat  string `yaml:"output_format"`
    SaveResults   bool   `yaml:"save_results"`
    OutputPath    string `yaml:"output_path"`
}

// BenchmarkConfig defines benchmarking settings
type BenchmarkConfig struct {
    TestDataPath    string  `yaml:"test_data_path"`
    TestLabelPath   string  `yaml:"test_label_path"`
    NumSamples      int     `yaml:"num_samples"`
    ReportTopK      int     `yaml:"report_top_k"`
    SaveConfusion   bool    `yaml:"save_confusion"`
    ProfileEnabled  bool    `yaml:"profile_enabled"`
    Tolerance       float32 `yaml:"tolerance"`
}

// Load reads and parses a YAML configuration file
func Load(configPath string) (*Config, error) {
    // Check if file exists
    if _, err := os.Stat(configPath); os.IsNotExist(err) {
        return nil, fmt.Errorf("configuration file does not exist: %s", configPath)
    }
    
    // Read file contents
    data, err := os.ReadFile(configPath)
    if err != nil {
        return nil, fmt.Errorf("failed to read config file: %w", err)
    }
    
    // Parse YAML
    var config Config
    err = yaml.Unmarshal(data, &config)
    if err != nil {
        return nil, fmt.Errorf("failed to parse YAML config: %w", err)
    }
    
    // Validate configuration
    err = config.Validate()
    if err != nil {
        return nil, fmt.Errorf("configuration validation failed: %w", err)
    }
    
    // Apply defaults
    config.ApplyDefaults()
    
    return &config, nil
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
    // Validate model config
    if c.Model.Name == "" {
        return fmt.Errorf("model name is required")
    }
    
    if c.Model.InputHeight <= 0 || c.Model.InputWidth <= 0 || c.Model.InputChannels <= 0 {
        return fmt.Errorf("invalid input dimensions: %dx%dx%d", 
            c.Model.InputHeight, c.Model.InputWidth, c.Model.InputChannels)
    }
    
    if c.Model.NumClasses <= 0 {
        return fmt.Errorf("number of classes must be positive")
    }
    
    if len(c.Model.ClassNames) != c.Model.NumClasses {
        return fmt.Errorf("class names count (%d) doesn't match num_classes (%d)", 
            len(c.Model.ClassNames), c.Model.NumClasses)
    }
    
    // Validate weights path
    if c.Model.WeightsPath == "" {
        return fmt.Errorf("weights path is required")
    }
    
    // Validate inference config
    if c.Inference.BatchSize <= 0 {
        c.Inference.BatchSize = 1 // Default
    }
    
    if c.Inference.NumWorkers <= 0 {
        c.Inference.NumWorkers = 1 // Default
    }
    
    // Validate data config
    if c.Data.Format == "" {
        c.Data.Format = "binary" // Default
    }
    
    if c.Data.Precision == "" {
        c.Data.Precision = "float32" // Default
    }
    
    return nil
}

// ApplyDefaults sets default values for optional configuration fields
func (c *Config) ApplyDefaults() {
    // Model defaults
    if c.Model.Architecture == "" {
        c.Model.Architecture = "tinycnn"
    }
    
    // Data defaults
    if c.Data.Format == "" {
        c.Data.Format = "binary"
    }
    
    if c.Data.Precision == "" {
        c.Data.Precision = "float32"
    }
    
    // Inference defaults
    if c.Inference.BatchSize <= 0 {
        c.Inference.BatchSize = 1
    }
    
    if c.Inference.NumWorkers <= 0 {
        c.Inference.NumWorkers = 1
    }
    
    if c.Inference.OutputFormat == "" {
        c.Inference.OutputFormat = "json"
    }
    
    // Benchmark defaults
    if c.Benchmark.ReportTopK <= 0 {
        c.Benchmark.ReportTopK = 5
    }
    
    if c.Benchmark.Tolerance <= 0 {
        c.Benchmark.Tolerance = 1e-6
    }
}

// GetWeightsPath returns the absolute path to the weights directory
func (c *Config) GetWeightsPath() (string, error) {
    if c.Model.WeightsPath == "" {
        return "", fmt.Errorf("weights path not configured")
    }
    
    // If path is relative, make it relative to config file directory
    if !filepath.IsAbs(c.Model.WeightsPath) {
        return filepath.Abs(c.Model.WeightsPath)
    }
    
    return c.Model.WeightsPath, nil
}

// Save writes the configuration to a YAML file
func (c *Config) Save(configPath string) error {
    data, err := yaml.Marshal(c)
    if err != nil {
        return fmt.Errorf("failed to marshal config to YAML: %w", err)
    }
    
    err = os.WriteFile(configPath, data, 0644)
    if err != nil {
        return fmt.Errorf("failed to write config file: %w", err)
    }
    
    return nil
}

// LoadWithEnvironmentOverrides loads config and applies environment variable overrides
func LoadWithEnvironmentOverrides(configPath string) (*Config, error) {
    config, err := Load(configPath)
    if err != nil {
        return nil, err
    }
    
    // Apply environment overrides
    if weightsPath := os.Getenv("GOCNN_WEIGHTS_PATH"); weightsPath != "" {
        config.Model.WeightsPath = weightsPath
    }
    
    if numWorkers := os.Getenv("GOCNN_NUM_WORKERS"); numWorkers != "" {
        // Parse and set (simplified - would need proper parsing)
        config.Inference.NumWorkers = 1 // Default fallback
    }
    
    return config, nil
}