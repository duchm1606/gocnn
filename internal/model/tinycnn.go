package model

import (
	"duchm1606/gocnn/internal/data"
	"duchm1606/gocnn/internal/ops"
	"duchm1606/gocnn/internal/tensor"
	"fmt"
	"time"
)

// TinyCNN represents the complete CNN model
type TinyCNN struct {
    architecture  *TinyCNNArchitecture
    weights       *data.ModelWeights
    convEngine    *ops.ConvolutionEngine
    
    // Performance tracking
    layerTimes    map[string]time.Duration
    totalInferences int64
}

// PredictionResult holds the result of a single inference
type PredictionResult struct {
    Probabilities    []float32         // Softmax probabilities for each class
    PredictedClass   int               // Index of most likely class
    Confidence       float32           // Confidence score (max probability)
    LayerTimes       map[string]time.Duration // Time spent in each layer type
    TotalTime        time.Duration     // Total inference time
}

// NewTinyCNN creates a new TinyCNN model
func NewTinyCNN(weightsPath string) (*TinyCNN, error) {
    // Load architecture
    arch := GetTinyCNNArchitecture()
    err := arch.ValidateArchitecture()
    if err != nil {
        return nil, fmt.Errorf("invalid architecture: %w", err)
    }
    
    // Load model weights
    dataManager := data.NewDataManager(weightsPath, data.BinaryFloat32, data.OneHotText)
    weights, err := dataManager.LoadModelWeights()
    if err != nil {
        return nil, fmt.Errorf("failed to load model weights: %w", err)
    }
    
    // Create convolution engine
    convEngine := ops.NewConvolutionEngine()
    
    model := &TinyCNN{
        architecture:    arch,
        weights:         weights,
        convEngine:      convEngine,
        layerTimes:      make(map[string]time.Duration),
        totalInferences: 0,
    }
    
    return model, nil
}

// Predict performs inference on a single image
func (cnn *TinyCNN) Predict(imageData []float32) (*PredictionResult, error) {
    startTime := time.Now()
    layerTimes := make(map[string]time.Duration)
    
    // Validate input
    expectedSize := cnn.architecture.InputHeight * cnn.architecture.InputWidth * cnn.architecture.InputChannels
    if len(imageData) != expectedSize {
        return nil, fmt.Errorf("input size mismatch: expected %d, got %d", expectedSize, len(imageData))
    }
    
    // Convert input to feature map
    input, err := tensor.NewFeatureMapFromData(imageData, 
        cnn.architecture.InputHeight, 
        cnn.architecture.InputWidth, 
        cnn.architecture.InputChannels)
    if err != nil {
        return nil, fmt.Errorf("failed to create input feature map: %w", err)
    }
    
    // Process through all layers
    current := input
    convLayerIdx := 0
    
    for i, layerConfig := range cnn.architecture.Layers {
        layerStart := time.Now()
        
        switch layerConfig.Type {
        case ConvolutionLayer:
            current, err = cnn.processConvolutionLayer(current, layerConfig, convLayerIdx)
            if err != nil {
                return nil, fmt.Errorf("failed at layer %d (%s): %w", i, layerConfig.Name, err)
            }
            convLayerIdx++
            
        case MaxPoolingLayer:
            current, err = cnn.processMaxPoolingLayer(current, layerConfig)
            if err != nil {
                return nil, fmt.Errorf("failed at layer %d (%s): %w", i, layerConfig.Name, err)
            }
            
        case GlobalMaxPoolingLayer:
            result, err := cnn.processGlobalMaxPoolingLayer(current)
            if err != nil {
                return nil, fmt.Errorf("failed at layer %d (%s): %w", i, layerConfig.Name, err)
            }
            
            // Apply softmax and return result
            return cnn.finalizePrediction(result, layerTimes, startTime)
            
        default:
            return nil, fmt.Errorf("unsupported layer type: %d", layerConfig.Type)
        }
        
        layerTimes[layerConfig.Name] = time.Since(layerStart)
    }
    
    return nil, fmt.Errorf("model did not reach final layer")
}

// processConvolutionLayer handles convolution + batch norm + activation
func (cnn *TinyCNN) processConvolutionLayer(input *tensor.FeatureMap, config LayerConfig, layerIdx int) (*tensor.FeatureMap, error) {
    if layerIdx >= len(cnn.weights.Kernels) {
        return nil, fmt.Errorf("kernel index %d out of range", layerIdx)
    }
    
    kernel := cnn.weights.Kernels[layerIdx]
    bias := cnn.weights.Biases[layerIdx]
    
    // Perform convolution
    convConfig := ops.Conv2DConfig{
        Padding: config.Padding,
        Stride:  config.Stride,
    }
    
    output := cnn.convEngine.Conv2DOptimized(input, kernel, bias, convConfig)
    
    // Apply batch normalization (if enabled and available)
    if config.ApplyBatchNorm && layerIdx < len(cnn.weights.BatchNorms) {
        batchNorm := cnn.weights.BatchNorms[layerIdx]
        ops.BatchNormalizeInPlace(output, &ops.BatchNormParams{
            Mean:     batchNorm.Mean,
            Variance: batchNorm.Variance,
            Scale:    batchNorm.Scale,
            Shift:    batchNorm.Shift,
            Epsilon:  batchNorm.Epsilon,
        })
    } else if config.ApplyActivation {
        // Apply ReLU activation if no batch norm
        ops.ReLUInPlace(output.Data)
    }
    
    return output, nil
}

// processMaxPoolingLayer handles max pooling operations
func (cnn *TinyCNN) processMaxPoolingLayer(input *tensor.FeatureMap, config LayerConfig) (*tensor.FeatureMap, error) {
    output := ops.MaxPooling2D(input, config.PoolSize, config.PoolStride)
    return output, nil
}

// processGlobalMaxPoolingLayer handles global max pooling
func (cnn *TinyCNN) processGlobalMaxPoolingLayer(input *tensor.FeatureMap) ([]float32, error) {
    result := ops.GlobalMaxPooling(input)
    return result, nil
}

// finalizePrediction applies softmax and creates the final result
func (cnn *TinyCNN) finalizePrediction(logits []float32, layerTimes map[string]time.Duration, startTime time.Time) (*PredictionResult, error) {
    // Apply softmax
    softmaxStart := time.Now()
    probabilities := ops.Softmax(logits)
    layerTimes["softmax"] = time.Since(softmaxStart)
    
    // Find predicted class and confidence
    predictedClass := ops.Argmax(probabilities)
    confidence := probabilities[predictedClass]
    
    // Update performance tracking
    totalTime := time.Since(startTime)
    cnn.totalInferences++
    
    // Accumulate layer times for performance analysis
    for layerName, layerTime := range layerTimes {
        cnn.layerTimes[layerName] += layerTime
    }
    
    return &PredictionResult{
        Probabilities:  probabilities,
        PredictedClass: predictedClass,
        Confidence:     confidence,
        LayerTimes:     layerTimes,
        TotalTime:      totalTime,
    }, nil
}

// PredictBatch performs inference on multiple images
func (cnn *TinyCNN) PredictBatch(images [][]float32) ([]*PredictionResult, error) {
    results := make([]*PredictionResult, len(images))
    
    for i, image := range images {
        result, err := cnn.Predict(image)
        if err != nil {
            return nil, fmt.Errorf("failed to predict image %d: %w", i, err)
        }
        results[i] = result
    }
    
    return results, nil
}

// GetModelInfo returns information about the model
func (cnn *TinyCNN) GetModelInfo() *ModelInfo {
    totalParams := int64(0)
    
    // Count parameters in kernels
    for _, kernel := range cnn.weights.Kernels {
        totalParams += int64(kernel.TotalWeights())
    }
    
    // Count parameters in biases
    for _, bias := range cnn.weights.Biases {
        totalParams += int64(len(bias))
    }
    
    // Count parameters in batch norm
    for _, bn := range cnn.weights.BatchNorms {
        totalParams += int64(len(bn.Mean) + len(bn.Variance) + len(bn.Scale) + len(bn.Shift))
    }
    
    return &ModelInfo{
        Architecture:     cnn.architecture,
        TotalParameters:  totalParams,
        TotalInferences:  cnn.totalInferences,
        AverageLayerTimes: cnn.getAverageLayerTimes(),
    }
}

// getAverageLayerTimes calculates average time per layer type
func (cnn *TinyCNN) getAverageLayerTimes() map[string]time.Duration {
    if cnn.totalInferences == 0 {
        return make(map[string]time.Duration)
    }
    
    avgTimes := make(map[string]time.Duration)
    for layerName, totalTime := range cnn.layerTimes {
        avgTimes[layerName] = time.Duration(int64(totalTime) / cnn.totalInferences)
    }
    
    return avgTimes
}

// ResetPerformanceCounters resets all performance tracking
func (cnn *TinyCNN) ResetPerformanceCounters() {
    cnn.layerTimes = make(map[string]time.Duration)
    cnn.totalInferences = 0
}

// ValidateModel performs basic validation on the loaded model
func (cnn *TinyCNN) ValidateModel() error {
    // Check that we have the right number of layers
    convLayers := 0
    for _, layer := range cnn.architecture.Layers {
        if layer.Type == ConvolutionLayer {
            convLayers++
        }
    }
    
    if len(cnn.weights.Kernels) != convLayers {
        return fmt.Errorf("kernel count mismatch: expected %d, got %d", convLayers, len(cnn.weights.Kernels))
    }
    
    if len(cnn.weights.Biases) != convLayers {
        return fmt.Errorf("bias count mismatch: expected %d, got %d", convLayers, len(cnn.weights.Biases))
    }
    
    // Validate each kernel
    for i, kernel := range cnn.weights.Kernels {
        err := tensor.ValidateKernel(kernel)
        if err != nil {
            return fmt.Errorf("kernel %d validation failed: %w", i, err)
        }
    }
    
    return nil
}

// ModelInfo holds information about the model
type ModelInfo struct {
    Architecture      *TinyCNNArchitecture
    TotalParameters   int64
    TotalInferences   int64
    AverageLayerTimes map[string]time.Duration
}

// Print displays model information in a readable format
func (info *ModelInfo) Print() {
    fmt.Println("Model Information:")
    fmt.Printf("  Input Size: %d×%d×%d\n",
        info.Architecture.InputHeight, 
        info.Architecture.InputWidth, 
        info.Architecture.InputChannels)
    fmt.Printf("  Output Classes: %d\n", info.Architecture.NumClasses)
    fmt.Printf("  Total Layers: %d\n", len(info.Architecture.Layers))
    fmt.Printf("  Total Parameters: %d\n", info.TotalParameters)
    fmt.Printf("  Total Inferences: %d\n", info.TotalInferences)
    
    if len(info.AverageLayerTimes) > 0 {
        fmt.Println("  Average Layer Times:")
        for layerName, avgTime := range info.AverageLayerTimes {
            fmt.Printf("    %s: %v\n", layerName, avgTime)
        }
    }
}