package main

import (
	"bufio"
	"duchm1606/gocnn/internal/model"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"duchm1606/gocnn/internal/config"
)

// InteractiveMode provides an interactive shell for multiple predictions
func runInteractiveMode(cnn *model.TinyCNN, cfg *config.Config) error {
    fmt.Println("Entering interactive mode. Type 'help' for commands, 'quit' to exit.")
    
    scanner := bufio.NewScanner(os.Stdin)
    
    for {
        fmt.Print("gocnn> ")
        
        if !scanner.Scan() {
            break
        }
        
        line := strings.TrimSpace(scanner.Text())
        if line == "" {
            continue
        }
        
        parts := strings.Fields(line)
        command := parts[0]
        
        switch command {
        case "help", "h":
            printInteractiveHelp()
            
        case "quit", "exit", "q":
            fmt.Println("Goodbye!")
            return nil
            
        case "predict", "p":
            if len(parts) < 2 {
                fmt.Println("Usage: predict <image_path>")
                continue
            }
            err := runInteractivePrediction(cnn, parts[1], cfg)
            if err != nil {
                fmt.Printf("Prediction failed: %v\n", err)
            }
            
        case "info", "i":
            printModelInfo(cnn)
            
        case "benchmark", "b":
            if len(parts) < 2 {
                fmt.Println("Usage: benchmark <image_path> [iterations]")
                continue
            }
            iterations := 10
            if len(parts) >= 3 {
                fmt.Sscanf(parts[2], "%d", &iterations)
            }
            err := runInteractiveBenchmark(cnn, parts[1], iterations, cfg)
            if err != nil {
                fmt.Printf("Benchmark failed: %v\n", err)
            }
            
        default:
            fmt.Printf("Unknown command: %s. Type 'help' for available commands.\n", command)
        }
    }
    
    return scanner.Err()
}

// printInteractiveHelp shows help for interactive mode
func printInteractiveHelp() {
    fmt.Println("Available commands:")
    fmt.Println("  predict <image>     Run inference on an image")
    fmt.Println("  benchmark <image>   Run benchmark on an image")
    fmt.Println("  info               Show model information")
    fmt.Println("  help               Show this help")
    fmt.Println("  quit               Exit interactive mode")
}

// runInteractivePrediction runs a single prediction in interactive mode
func runInteractivePrediction(cnn *model.TinyCNN, imagePath string, cfg *config.Config) error {
    // Check if file exists
    if _, err := os.Stat(imagePath); os.IsNotExist(err) {
        return fmt.Errorf("image file does not exist: %s", imagePath)
    }
    
    // Load image
    imageData, err := loadImage(imagePath, cfg)
    if err != nil {
        return err
    }
    
    // Run prediction
    start := time.Now()
    result, err := cnn.Predict(imageData)
    if err != nil {
        return err
    }
    
    // Display results
    fmt.Printf("Image: %s\n", filepath.Base(imagePath))
    fmt.Printf("Predicted: %d (%s) - Confidence: %.4f (%.2f%%)\n",
        result.PredictedClass,
        getClassName(result.PredictedClass, cfg.Model.ClassNames),
        result.Confidence,
        result.Confidence*100)
    fmt.Printf("Inference time: %v\n\n", time.Since(start))
    
    return nil
}

// runInteractiveBenchmark runs a benchmark in interactive mode
func runInteractiveBenchmark(cnn *model.TinyCNN, imagePath string, iterations int, cfg *config.Config) error {
    // Load image
    imageData, err := loadImage(imagePath, cfg)
    if err != nil {
        return err
    }
    
    fmt.Printf("Running benchmark on %s (%d iterations)...\n", filepath.Base(imagePath), iterations)
    
    start := time.Now()
    for i := 0; i < iterations; i++ {
        _, err := cnn.Predict(imageData)
        if err != nil {
            return fmt.Errorf("iteration %d failed: %w", i+1, err)
        }
    }
    totalTime := time.Since(start)
    
    avgTime := totalTime / time.Duration(iterations)
    throughput := float64(iterations) / totalTime.Seconds()
    
    fmt.Printf("Benchmark completed:\n")
    fmt.Printf("  Total time: %v\n", totalTime)
    fmt.Printf("  Average time: %v\n", avgTime)
    fmt.Printf("  Throughput: %.2f images/sec\n\n", throughput)
    
    return nil
}

// printModelInfo displays detailed model information
func printModelInfo(cnn *model.TinyCNN) {
    info := cnn.GetModelInfo()
    
    fmt.Println("Model Information:")
    fmt.Printf("  Architecture: TinyCNN for CIFAR-10\n")
    fmt.Printf("  Input size: %d×%d×%d\n",
        info.Architecture.InputHeight,
        info.Architecture.InputWidth,
        info.Architecture.InputChannels)
    fmt.Printf("  Output classes: %d\n", info.Architecture.NumClasses)
    fmt.Printf("  Total parameters: %d\n", info.TotalParameters)
    fmt.Printf("  Total layers: %d\n", len(info.Architecture.Layers))
    fmt.Printf("  Total inferences: %d\n", info.TotalInferences)
    
    if len(info.AverageLayerTimes) > 0 {
        fmt.Println("  Average layer times:")
        for layerName, avgTime := range info.AverageLayerTimes {
            fmt.Printf("    %s: %v\n", layerName, avgTime)
        }
    }
    fmt.Println()
}

// BatchProcessor handles batch processing of multiple images
type BatchProcessor struct {
    cnn    *model.TinyCNN
    config *config.Config
}

// NewBatchProcessor creates a new batch processor
func NewBatchProcessor(cnn *model.TinyCNN, cfg *config.Config) *BatchProcessor {
    return &BatchProcessor{
        cnn:    cnn,
        config: cfg,
    }
}

// ProcessDirectory processes all images in a directory
func (bp *BatchProcessor) ProcessDirectory(dirPath, outputPath string) error {
    // Find all image files
    files, err := filepath.Glob(filepath.Join(dirPath, "*.bin"))
    if err != nil {
        return fmt.Errorf("failed to find image files: %w", err)
    }
    
    if len(files) == 0 {
        return fmt.Errorf("no .bin files found in directory: %s", dirPath)
    }
    
    fmt.Printf("Processing %d images from %s...\n", len(files), dirPath)
    
    // Create output file
    var outputFile *os.File
    if outputPath != "" {
        outputFile, err = os.Create(outputPath)
        if err != nil {
            return fmt.Errorf("failed to create output file: %w", err)
        }
        defer outputFile.Close()
        
        // Write header
        fmt.Fprintf(outputFile, "Filename,PredictedClass,ClassName,Confidence,InferenceTime\n")
    }
    
    // Process each image
    totalStart := time.Now()
    // correct := 0
    
    for i, file := range files {
        // Load and predict
        imageData, err := loadImage(file, bp.config)
        if err != nil {
            fmt.Printf("Failed to load %s: %v\n", filepath.Base(file), err)
            continue
        }
        
        start := time.Now()
        result, err := bp.cnn.Predict(imageData)
        if err != nil {
            fmt.Printf("Failed to predict %s: %v\n", filepath.Base(file), err)
            continue
        }
        inferenceTime := time.Since(start)
        
        // Display progress
        if (i+1)%10 == 0 || i == len(files)-1 {
            fmt.Printf("  Processed %d/%d images\n", i+1, len(files))
        }
        
        // Write to output file
        if outputFile != nil {
            filename := filepath.Base(file)
            className := getClassName(result.PredictedClass, bp.config.Model.ClassNames)
            fmt.Fprintf(outputFile, "%s,%d,%s,%.6f,%v\n",
                filename, result.PredictedClass, className, 
                result.Confidence, inferenceTime)
        }
    }
    
    totalTime := time.Since(totalStart)
    
    fmt.Printf("\nBatch processing completed:\n")
    fmt.Printf("  Total images: %d\n", len(files))
    fmt.Printf("  Total time: %v\n", totalTime)
    fmt.Printf("  Average time per image: %v\n", totalTime/time.Duration(len(files)))
    fmt.Printf("  Throughput: %.2f images/sec\n", float64(len(files))/totalTime.Seconds())
    
    if outputPath != "" {
        fmt.Printf("  Results saved to: %s\n", outputPath)
    }
    
    return nil
}

// ValidateImageFile checks if a file is a valid image for the model
func ValidateImageFile(imagePath string, expectedSize int64) error {
    info, err := os.Stat(imagePath)
    if err != nil {
        return fmt.Errorf("cannot access file: %w", err)
    }
    
    if info.Size() != expectedSize {
        return fmt.Errorf("invalid file size: expected %d bytes, got %d bytes", 
            expectedSize, info.Size())
    }
    
    return nil
}

// GetExpectedImageSize calculates the expected image file size
func GetExpectedImageSize(height, width, channels int) int64 {
    return int64(height * width * channels * 4) // 4 bytes per float32
}