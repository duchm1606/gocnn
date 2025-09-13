package main

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func createTestConfig(t *testing.T, dir string) string {
    configPath := filepath.Join(dir, "test_config.yaml")
    
    configContent := `model_name: "TinyCNN-CIFAR10"
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
`
    
    err := os.WriteFile(configPath, []byte(configContent), 0644)
    if err != nil {
        t.Fatalf("Failed to create test config: %v", err)
    }
    
    return configPath
}

func createTestImage(t *testing.T, dir string) string {
    imagePath := filepath.Join(dir, "test_image.bin")
    
    file, err := os.Create(imagePath)
    if err != nil {
        t.Fatalf("Failed to create test image: %v", err)
    }
    defer file.Close()
    
    // Write 32×32×3 float32 values
    for i := 0; i < 32*32*3; i++ {
        value := float32(0.5) // Neutral value
        binary.Write(file, binary.LittleEndian, value)
    }
    
    return imagePath
}

func TestValidateArgs(t *testing.T) {
    // Save original flags
    origWeights := *weightsPath
    origImage := *imagePath
    origConfig := *configPath
    
    defer func() {
        *weightsPath = origWeights
        *imagePath = origImage
        *configPath = origConfig
    }()
    
    // Test missing weights
    *weightsPath = ""
    *imagePath = "test.bin"
    *configPath = "config.yaml"
    
    err := validateArgs()
    if err == nil {
        t.Error("Expected error for missing weights path")
    }
    
    // Test missing image
    *weightsPath = "./weights"
    *imagePath = ""
    
    err = validateArgs()
    if err == nil {
        t.Error("Expected error for missing image path")
    }
}

func TestGetClassName(t *testing.T) {
    classNames := []string{"Class0", "Class1", "Class2"}
    
    // Test valid indices
    if getClassName(0, classNames) != "Class0" {
        t.Error("Failed to get correct class name for index 0")
    }
    
    if getClassName(2, classNames) != "Class2" {
        t.Error("Failed to get correct class name for index 2")
    }
    
    // Test invalid index
    result := getClassName(5, classNames)
    if result != "Unknown_5" {
        t.Errorf("Expected 'Unknown_5', got '%s'", result)
    }
    
    // Test negative index
    result = getClassName(-1, classNames)
    if result != "Unknown_-1" {
        t.Errorf("Expected 'Unknown_-1', got '%s'", result)
    }
}

func TestGetLogLevel(t *testing.T) {
    // Save original flags
    origQuiet := *quiet
    origVerbose := *verbose
    
    defer func() {
        *quiet = origQuiet
        *verbose = origVerbose
    }()
    
    // Test normal level
    *quiet = false
    *verbose = false
    if getLogLevel() != LogNormal {
        t.Error("Expected LogNormal")
    }
    
    // Test quiet level
    *quiet = true
    *verbose = false
    if getLogLevel() != LogQuiet {
        t.Error("Expected LogQuiet")
    }
    
    // Test verbose level
    *quiet = false
    *verbose = true
    if getLogLevel() != LogVerbose {
        t.Error("Expected LogVerbose")
    }
    
    // Test quiet takes precedence
    *quiet = true
    *verbose = true
    if getLogLevel() != LogQuiet {
        t.Error("Expected LogQuiet when both flags are set")
    }
}

func TestValidateImageFile(t *testing.T) {
    tempDir := t.TempDir()
    
    // Create a test file with correct size
    correctSize := int64(32 * 32 * 3 * 4) // 4 bytes per float32
    correctFile := filepath.Join(tempDir, "correct.bin")
    
    file, err := os.Create(correctFile)
    if err != nil {
        t.Fatalf("Failed to create test file: %v", err)
    }
    
    // Write correct amount of data
    data := make([]byte, correctSize)
    file.Write(data)
    file.Close()
    
    // Test with correct size
    err = ValidateImageFile(correctFile, correctSize)
    if err != nil {
        t.Errorf("Validation failed for correct file: %v", err)
    }
    
    // Test with wrong size
    err = ValidateImageFile(correctFile, correctSize+1)
    if err == nil {
        t.Error("Expected error for wrong file size")
    }
    
    // Test with non-existent file
    err = ValidateImageFile("nonexistent.bin", correctSize)
    if err == nil {
        t.Error("Expected error for non-existent file")
    }
}

func TestGetExpectedImageSize(t *testing.T) {
    size := GetExpectedImageSize(32, 32, 3)
    expected := int64(32 * 32 * 3 * 4) // 4 bytes per float32
    
    if size != expected {
        t.Errorf("Expected size %d, got %d", expected, size)
    }
    
    // Test different dimensions
    size = GetExpectedImageSize(64, 64, 1)
    expected = int64(64 * 64 * 1 * 4)
    
    if size != expected {
        t.Errorf("Expected size %d, got %d", expected, size)
    }
}

// Integration test (requires setting up test model)
func TestInferenceIntegration(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test in short mode")
    }
    
    // This test would require actual model weights and test data
    // For now, we just test that the application can be instantiated
    t.Log("Integration test placeholder - requires full model setup")
}