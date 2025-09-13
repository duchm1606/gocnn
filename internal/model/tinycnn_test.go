package model

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// Helper function to create test weight files
func createTestWeights(t testing.TB, weightsDir string) {
    // Create directories
    os.MkdirAll(weightsDir, 0755)
    
    // Create test weight files for all layers
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
        // Create weight file
        weightFile := filepath.Join(weightsDir, config.name+"_weight.bin")
        createWeightFile(t, weightFile, config.size, config.channels, config.filters)
        
        // Create bias file
        biasFile := filepath.Join(weightsDir, config.name+"_bias.bin")
        createBiasFile(t, biasFile, config.filters)
        
        // Create batch normalization files for all layers except the last one (conv7)
        if i < len(layerConfigs)-1 {
            createBatchNormFiles(t, weightsDir, i+1, config.filters)
        }
    }
}

// Helper function to create batch normalization parameter files
func createBatchNormFiles(t testing.TB, weightsDir string, layerIndex int, filters int) {
    // Create batch norm directory
    bnDir := filepath.Join(weightsDir, fmt.Sprintf("batchnorm%d", layerIndex))
    err := os.MkdirAll(bnDir, 0755)
    if err != nil {
        t.Fatalf("Failed to create batch norm directory: %v", err)
    }
    
    bnName := fmt.Sprintf("bn%d", layerIndex)
    
    // Create moving mean file
    meanFile := filepath.Join(bnDir, bnName+"_moving_mean.bin")
    createFloatArrayFile(t, meanFile, filters, 0.0)
    
    // Create moving variance file
    varianceFile := filepath.Join(bnDir, bnName+"_moving_variance.bin")
    createFloatArrayFile(t, varianceFile, filters, 1.0)
    
    // Create gamma (scale) file
    gammaFile := filepath.Join(bnDir, bnName+"_gamma.bin")
    createFloatArrayFile(t, gammaFile, filters, 1.0)
    
    // Create beta (shift) file
    betaFile := filepath.Join(bnDir, bnName+"_beta.bin")
    createFloatArrayFile(t, betaFile, filters, 0.0)
}

// Helper function to create a file with float32 array
func createFloatArrayFile(t testing.TB, filename string, size int, value float32) {
    file, err := os.Create(filename)
    if err != nil {
        t.Fatalf("Failed to create file %s: %v", filename, err)
    }
    defer file.Close()
    
    // Write array values
    for i := 0; i < size; i++ {
        err := binary.Write(file, binary.LittleEndian, value)
        if err != nil {
            t.Fatalf("Failed to write to file %s: %v", filename, err)
        }
    }
}

func createWeightFile(t testing.TB, filename string, size, channels, filters int) {
    file, err := os.Create(filename)
    if err != nil {
        t.Fatalf("Failed to create weight file: %v", err)
    }
    defer file.Close()
    
    // Write test weights
    for h := 0; h < size; h++ {
        for w := 0; w < size; w++ {
            for c := 0; c < channels; c++ {
                for f := 0; f < filters; f++ {
                    weight := float32(0.1) // Small positive weight
                    binary.Write(file, binary.LittleEndian, weight)
                }
            }
        }
    }
}

func createBiasFile(t testing.TB, filename string, filters int) {
    file, err := os.Create(filename)
    if err != nil {
        t.Fatalf("Failed to create bias file: %v", err)
    }
    defer file.Close()
    
    // Write test biases
    for f := 0; f < filters; f++ {
        bias := float32(0.01) // Small bias
        binary.Write(file, binary.LittleEndian, bias)
    }
}

func TestTinyCNNCreation(t *testing.T) {
    // Create temporary weights directory
    tempDir := t.TempDir()
    createTestWeights(t, tempDir)
    
    // Create model
    model, err := NewTinyCNN(tempDir)
    if err != nil {
        t.Fatalf("Failed to create TinyCNN: %v", err)
    }
    
    // Check model is not nil
    if model == nil {
        t.Fatal("Model is nil")
    }
    
    // Check architecture
    if model.architecture == nil {
        t.Fatal("Architecture is nil")
    }
    
    // Check weights loaded
    if len(model.weights.Kernels) == 0 {
        t.Fatal("No kernels loaded")
    }
    
    if len(model.weights.Biases) == 0 {
        t.Fatal("No biases loaded")
    }
}

func TestTinyCNNPredict(t *testing.T) {
    // Create temporary weights directory
    tempDir := t.TempDir()
    createTestWeights(t, tempDir)
    
    // Create model
    model, err := NewTinyCNN(tempDir)
    if err != nil {
        t.Fatalf("Failed to create TinyCNN: %v", err)
    }
    
    // Create test input (32×32×3 = 3072 floats)
    inputSize := 32 * 32 * 3
    imageData := make([]float32, inputSize)
    for i := range imageData {
        imageData[i] = 0.5 // Neutral input
    }
    
    // Perform prediction
    result, err := model.Predict(imageData)
    if err != nil {
        t.Fatalf("Prediction failed: %v", err)
    }
    
    // Verify result
    if result == nil {
        t.Fatal("Prediction result is nil")
    }
    
    if len(result.Probabilities) != 10 {
        t.Errorf("Expected 10 probabilities, got %d", len(result.Probabilities))
    }
    
    // Check probabilities sum to approximately 1
    sum := float32(0)
    for _, prob := range result.Probabilities {
        sum += prob
    }
    
    if sum < 0.99 || sum > 1.01 {
        t.Errorf("Probabilities don't sum to 1: %f", sum)
    }
    
    // Check predicted class is valid
    if result.PredictedClass < 0 || result.PredictedClass >= 10 {
        t.Errorf("Invalid predicted class: %d", result.PredictedClass)
    }
    
    // Check confidence is reasonable
    if result.Confidence <= 0 || result.Confidence > 1 {
        t.Errorf("Invalid confidence: %f", result.Confidence)
    }
}

func TestTinyCNNPredictBatch(t *testing.T) {
    // Create temporary weights directory
    tempDir := t.TempDir()
    createTestWeights(t, tempDir)
    
    // Create model
    model, err := NewTinyCNN(tempDir)
    if err != nil {
        t.Fatalf("Failed to create TinyCNN: %v", err)
    }
    
    // Create batch of test inputs
    batchSize := 3
    inputSize := 32 * 32 * 3
    images := make([][]float32, batchSize)
    
    for i := 0; i < batchSize; i++ {
        images[i] = make([]float32, inputSize)
        for j := range images[i] {
            images[i][j] = 0.5 + float32(i)*0.1 // Slightly different inputs
        }
    }
    
    // Perform batch prediction
    results, err := model.PredictBatch(images)
    if err != nil {
        t.Fatalf("Batch prediction failed: %v", err)
    }
    
    // Verify results
    if len(results) != batchSize {
        t.Errorf("Expected %d results, got %d", batchSize, len(results))
    }
    
    for i, result := range results {
        if result == nil {
            t.Errorf("Result %d is nil", i)
            continue
        }
        
        if len(result.Probabilities) != 10 {
            t.Errorf("Result %d has wrong number of probabilities: %d", i, len(result.Probabilities))
        }
    }
}

func TestGetTinyCNNArchitecture(t *testing.T) {
    arch := GetTinyCNNArchitecture()
    
    // Check basic properties
    if arch.InputHeight != 32 || arch.InputWidth != 32 || arch.InputChannels != 3 {
        t.Errorf("Wrong input dimensions: (%d,%d,%d)", arch.InputHeight, arch.InputWidth, arch.InputChannels)
    }
    
    if arch.NumClasses != 10 {
        t.Errorf("Wrong number of classes: %d", arch.NumClasses)
    }
    
    // Check number of layers
    expectedLayers := 12 // 7 conv + 3 maxpool + 1 global maxpool + 1 softmax
    if len(arch.Layers) != expectedLayers {
        t.Errorf("Wrong number of layers: %d, expected %d", len(arch.Layers), expectedLayers)
    }
    
    // Validate architecture
    err := arch.ValidateArchitecture()
    if err != nil {
        t.Errorf("Architecture validation failed: %v", err)
    }
}

func TestArchitectureOutputDimensions(t *testing.T) {
    arch := GetTinyCNNArchitecture()
    
    dimensions, err := arch.GetOutputDimensions()
    if err != nil {
        t.Fatalf("Failed to get output dimensions: %v", err)
    }
    
    // Check input dimensions
    if dimensions[0][0] != 32 || dimensions[0][1] != 32 || dimensions[0][2] != 3 {
        t.Errorf("Wrong input dimensions: %v", dimensions[0])
    }
    
    // Check final dimensions (should be 1×1×10 before softmax)
    finalIdx := len(dimensions) - 2 // Before softmax layer
    if dimensions[finalIdx][0] != 1 || dimensions[finalIdx][1] != 1 || dimensions[finalIdx][2] != 10 {
        t.Errorf("Wrong final dimensions: %v", dimensions[finalIdx])
    }
}

func TestModelValidation(t *testing.T) {
    // Create temporary weights directory
    tempDir := t.TempDir()
    createTestWeights(t, tempDir)
    
    // Create model
    model, err := NewTinyCNN(tempDir)
    if err != nil {
        t.Fatalf("Failed to create TinyCNN: %v", err)
    }
    
    // Validate model
    err = model.ValidateModel()
    if err != nil {
        t.Errorf("Model validation failed: %v", err)
    }
}

func TestInvalidInput(t *testing.T) {
    // Create temporary weights directory
    tempDir := t.TempDir()
    createTestWeights(t, tempDir)
    
    // Create model
    model, err := NewTinyCNN(tempDir)
    if err != nil {
        t.Fatalf("Failed to create TinyCNN: %v", err)
    }
    
    // Test with wrong input size
    wrongSizeInput := make([]float32, 100) // Wrong size
    _, err = model.Predict(wrongSizeInput)
    if err == nil {
        t.Error("Expected error for wrong input size, but got none")
    }
    
    // Test with nil input
    _, err = model.Predict(nil)
    if err == nil {
        t.Error("Expected error for nil input, but got none")
    }
}

// Benchmark tests
func BenchmarkTinyCNNPredict(b *testing.B) {
    // Setup
    tempDir := b.TempDir()
    createTestWeights(b, tempDir)
    
    model, err := NewTinyCNN(tempDir)
    if err != nil {
        b.Fatalf("Failed to create TinyCNN: %v", err)
    }
    
    inputSize := 32 * 32 * 3
    imageData := make([]float32, inputSize)
    for i := range imageData {
        imageData[i] = 0.5
    }
    
    // Benchmark
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := model.Predict(imageData)
        if err != nil {
            b.Fatalf("Prediction failed: %v", err)
        }
    }
}

func BenchmarkTinyCNNPredictBatch(b *testing.B) {
    // Setup
    tempDir := b.TempDir()
    createTestWeights(b, tempDir)
    
    model, err := NewTinyCNN(tempDir)
    if err != nil {
        b.Fatalf("Failed to create TinyCNN: %v", err)
    }
    
    batchSize := 10
    inputSize := 32 * 32 * 3
    images := make([][]float32, batchSize)
    
    for i := 0; i < batchSize; i++ {
        images[i] = make([]float32, inputSize)
        for j := range images[i] {
            images[i][j] = 0.5
        }
    }
    
    // Benchmark
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := model.PredictBatch(images)
        if err != nil {
            b.Fatalf("Batch prediction failed: %v", err)
        }
    }
}