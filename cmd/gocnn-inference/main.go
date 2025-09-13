package main

import (
	"duchm1606/gocnn/internal/data"
	"duchm1606/gocnn/internal/model"
	"flag"
	"fmt"
	"os"
	"time"

	"duchm1606/gocnn/internal/config"
)

// Version information
const (
    AppName    = "gocnn-inference"
    AppVersion = "1.0.0"
    AppDesc    = "TinyCNN inference engine for CIFAR-10 classification"
)

// Command line flags
var (
    weightsPath = flag.String("weights", "", "Path to model weights directory (required)")
    imagePath   = flag.String("image", "", "Path to input image file (required)")
    configPath  = flag.String("config", "configs/cifar10.yaml", "Path to model configuration file")
    outputPath  = flag.String("output", "", "Path to save detailed results (optional)")
    verbose     = flag.Bool("verbose", false, "Enable verbose output")
    quiet       = flag.Bool("quiet", false, "Suppress non-essential output")
    showVersion = flag.Bool("version", false, "Show version information")
    showHelp    = flag.Bool("help", false, "Show detailed help")
    benchmark   = flag.Bool("benchmark", false, "Run in benchmark mode (multiple iterations)")
    iterations  = flag.Int("iterations", 10, "Number of iterations for benchmark mode")
)

func main() {
    // Parse command line flags
    flag.Parse()

    // Handle special flags
    if *showVersion {
        printVersion()
        return
    }

    if *showHelp {
        printHelp()
        return
    }

    // Validate required arguments
    if err := validateArgs(); err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        fmt.Fprintf(os.Stderr, "Use -help for usage information\n")
        os.Exit(1)
    }

    // Set log level based on flags
    logLevel := getLogLevel()

    // Run the inference
    if err := runInference(logLevel); err != nil {
        fmt.Fprintf(os.Stderr, "Inference failed: %v\n", err)
        os.Exit(1)
    }
}

// validateArgs validates command line arguments
func validateArgs() error {
    if *weightsPath == "" {
        return fmt.Errorf("weights path is required (use -weights)")
    }

    if *imagePath == "" {
        return fmt.Errorf("image path is required (use -image)")
    }

    // Check if weights directory exists
    if _, err := os.Stat(*weightsPath); os.IsNotExist(err) {
        return fmt.Errorf("weights directory does not exist: %s", *weightsPath)
    }

    // Check if image file exists
    if _, err := os.Stat(*imagePath); os.IsNotExist(err) {
        return fmt.Errorf("image file does not exist: %s", *imagePath)
    }

    // Check if config file exists
    if _, err := os.Stat(*configPath); os.IsNotExist(err) {
        return fmt.Errorf("config file does not exist: %s", *configPath)
    }

    return nil
}

// LogLevel defines the verbosity of output
type LogLevel int

const (
    LogQuiet LogLevel = iota
    LogNormal
    LogVerbose
)

// getLogLevel determines the appropriate log level
func getLogLevel() LogLevel {
    if *quiet {
        return LogQuiet
    }
    if *verbose {
        return LogVerbose
    }
    return LogNormal
}

// runInference performs the main inference workflow
func runInference(logLevel LogLevel) error {
    // Load configuration
    if logLevel >= LogVerbose {
        fmt.Printf("Loading configuration from %s...\n", *configPath)
    }

    cfg, err := config.Load(*configPath)
    if err != nil {
        return fmt.Errorf("failed to load configuration: %w", err)
    }

    // Create and load model
    if logLevel >= LogNormal {
        fmt.Printf("Loading CNN model from %s...\n", *weightsPath)
    }

    start := time.Now()
    cnn, err := model.NewTinyCNN(*weightsPath)
    if err != nil {
        return fmt.Errorf("failed to load model: %w", err)
    }
    loadTime := time.Since(start)

    if logLevel >= LogVerbose {
        fmt.Printf("Model loaded in %v\n", loadTime)
        
        // Print model information
        modelInfo := cnn.GetModelInfo()
        fmt.Printf("Model Information:\n")
        fmt.Printf("  Total Parameters: %d\n", modelInfo.TotalParameters)
        fmt.Printf("  Input Size: %d×%d×%d\n", 
            modelInfo.Architecture.InputHeight,
            modelInfo.Architecture.InputWidth, 
            modelInfo.Architecture.InputChannels)
        fmt.Printf("  Output Classes: %d\n", modelInfo.Architecture.NumClasses)
    }

    // Load and preprocess image
    if logLevel >= LogVerbose {
        fmt.Printf("Loading image from %s...\n", *imagePath)
    }

    imageData, err := loadImage(*imagePath, cfg)
    if err != nil {
        return fmt.Errorf("failed to load image: %w", err)
    }

    // Run inference
    if *benchmark {
        return runBenchmark(cnn, imageData, cfg, logLevel)
    } else {
        return runSingleInference(cnn, imageData, cfg, logLevel)
    }
}

// loadImage loads and preprocesses an image file
func loadImage(imagePath string, cfg *config.Config) ([]float32, error) {
    imageLoader := data.NewImageLoader(data.BinaryFloat32)
    
    // Load image
    fm, err := imageLoader.LoadImage(imagePath, cfg.Model.InputHeight, cfg.Model.InputWidth, cfg.Model.InputChannels)
    if err != nil {
        return nil, fmt.Errorf("failed to load image: %w", err)
    }

    // Validate image dimensions
    if fm.Height != cfg.Model.InputHeight || fm.Width != cfg.Model.InputWidth || fm.Channels != cfg.Model.InputChannels {
        return nil, fmt.Errorf("image dimensions (%d×%d×%d) don't match expected (%d×%d×%d)",
            fm.Height, fm.Width, fm.Channels,
            cfg.Model.InputHeight, cfg.Model.InputWidth, cfg.Model.InputChannels)
    }

    return fm.Data, nil
}

// runSingleInference performs a single inference
func runSingleInference(cnn *model.TinyCNN, imageData []float32, cfg *config.Config, logLevel LogLevel) error {
    if logLevel >= LogNormal {
        fmt.Println("Running inference...")
    }

    start := time.Now()
    result, err := cnn.Predict(imageData)
    if err != nil {
        return fmt.Errorf("inference failed: %w", err)
    }
    totalTime := time.Since(start)

    // Display results
    if logLevel >= LogQuiet {
        fmt.Println("\nPrediction Results:")
        fmt.Printf("  Predicted Class: %d (%s)\n", 
            result.PredictedClass, 
            getClassName(result.PredictedClass, cfg.Model.ClassNames))
        fmt.Printf("  Confidence: %.4f (%.2f%%)\n", 
            result.Confidence, 
            result.Confidence*100)
    }

    if logLevel >= LogVerbose {
        fmt.Println("\nAll Class Probabilities:")
        for i, prob := range result.Probabilities {
            className := getClassName(i, cfg.Model.ClassNames)
            fmt.Printf("    %d (%s): %.6f\n", i, className, prob)
        }

        fmt.Printf("\nTiming Information:\n")
        fmt.Printf("  Total Inference Time: %v\n", totalTime)
        
        for layerName, layerTime := range result.LayerTimes {
            fmt.Printf("  %s: %v\n", layerName, layerTime)
        }
    }

    // Save detailed results if output path is specified
    if *outputPath != "" {
        err := saveDetailedResults(result, *outputPath, cfg)
        if err != nil {
            return fmt.Errorf("failed to save results: %w", err)
        }
        
        if logLevel >= LogNormal {
            fmt.Printf("\nDetailed results saved to: %s\n", *outputPath)
        }
    }

    return nil
}

// runBenchmark performs multiple inference iterations for benchmarking
func runBenchmark(cnn *model.TinyCNN, imageData []float32, cfg *config.Config, logLevel LogLevel) error {
    if logLevel >= LogNormal {
        fmt.Printf("Running benchmark with %d iterations...\n", *iterations)
    }

    var totalTime time.Duration
    var results []*model.PredictionResult

    for i := 0; i < *iterations; i++ {
        start := time.Now()
        result, err := cnn.Predict(imageData)
        if err != nil {
            return fmt.Errorf("benchmark iteration %d failed: %w", i+1, err)
        }
        
        iterTime := time.Since(start)
        totalTime += iterTime
        results = append(results, result)

        if logLevel >= LogVerbose {
            fmt.Printf("  Iteration %d: %v (class: %d, confidence: %.4f)\n", 
                i+1, iterTime, result.PredictedClass, result.Confidence)
        }
    }

    // Calculate statistics
    avgTime := totalTime / time.Duration(*iterations)
    minTime := results[0].TotalTime
    maxTime := results[0].TotalTime

    for _, result := range results[1:] {
        if result.TotalTime < minTime {
            minTime = result.TotalTime
        }
        if result.TotalTime > maxTime {
            maxTime = result.TotalTime
        }
    }

    // Check consistency
    firstPrediction := results[0].PredictedClass
    consistent := true
    for _, result := range results[1:] {
        if result.PredictedClass != firstPrediction {
            consistent = false
            break
        }
    }

    // Display benchmark results
    fmt.Println("\nBenchmark Results:")
    fmt.Printf("  Iterations: %d\n", *iterations)
    fmt.Printf("  Total Time: %v\n", totalTime)
    fmt.Printf("  Average Time: %v\n", avgTime)
    fmt.Printf("  Min Time: %v\n", minTime)
    fmt.Printf("  Max Time: %v\n", maxTime)
    fmt.Printf("  Throughput: %.2f images/sec\n", float64(*iterations)/totalTime.Seconds())
    fmt.Printf("  Predictions Consistent: %v\n", consistent)

    if !consistent {
        fmt.Println("  Warning: Predictions were not consistent across iterations")
    }

    return nil
}

// saveDetailedResults saves comprehensive results to a file
func saveDetailedResults(result *model.PredictionResult, outputPath string, cfg *config.Config) error {
    file, err := os.Create(outputPath)
    if err != nil {
        return fmt.Errorf("failed to create output file: %w", err)
    }
    defer file.Close()

    // Write detailed results
    fmt.Fprintf(file, "GoCNN Inference Results\n")
    fmt.Fprintf(file, "=======================\n\n")
    fmt.Fprintf(file, "Predicted Class: %d (%s)\n", 
        result.PredictedClass, 
        getClassName(result.PredictedClass, cfg.Model.ClassNames))
    fmt.Fprintf(file, "Confidence: %.6f\n", result.Confidence)
    fmt.Fprintf(file, "Total Inference Time: %v\n\n", result.TotalTime)

    fmt.Fprintf(file, "All Class Probabilities:\n")
    for i, prob := range result.Probabilities {
        className := getClassName(i, cfg.Model.ClassNames)
        fmt.Fprintf(file, "  %d (%s): %.8f\n", i, className, prob)
    }

    fmt.Fprintf(file, "\nLayer Timing Breakdown:\n")
    for layerName, layerTime := range result.LayerTimes {
        fmt.Fprintf(file, "  %s: %v\n", layerName, layerTime)
    }

    return nil
}

// getClassName returns the human-readable class name
func getClassName(classIndex int, classNames []string) string {
    if classIndex >= 0 && classIndex < len(classNames) {
        return classNames[classIndex]
    }
    return fmt.Sprintf("Unknown_%d", classIndex)
}

// printVersion displays version information
func printVersion() {
    fmt.Printf("%s version %s\n", AppName, AppVersion)
    fmt.Printf("%s\n", AppDesc)
}

// printHelp displays detailed help information
func printHelp() {
    fmt.Printf("%s - %s\n\n", AppName, AppDesc)
    
    fmt.Println("USAGE:")
    fmt.Printf("  %s -weights <path> -image <path> [options]\n\n", AppName)
    
    fmt.Println("REQUIRED:")
    fmt.Println("  -weights <path>    Path to directory containing model weights")
    fmt.Println("  -image <path>      Path to input image file (32x32x3 binary format)")
    
    fmt.Println("\nOPTIONS:")
    fmt.Println("  -config <path>     Path to model configuration file (default: configs/cifar10.yaml)")
    fmt.Println("  -output <path>     Save detailed results to file")
    fmt.Println("  -verbose           Enable verbose output")
    fmt.Println("  -quiet             Suppress non-essential output")
    fmt.Println("  -benchmark         Run in benchmark mode")
    fmt.Println("  -iterations <n>    Number of iterations for benchmark (default: 10)")
    fmt.Println("  -version           Show version information")
    fmt.Println("  -help              Show this help message")
    
    fmt.Println("\nEXAMPLES:")
    fmt.Printf("  # Basic inference\n")
    fmt.Printf("  %s -weights ./weights -image ./test.bin\n\n", AppName)
    
    fmt.Printf("  # Verbose inference with output file\n")
    fmt.Printf("  %s -weights ./weights -image ./test.bin -verbose -output results.txt\n\n", AppName)
    
    fmt.Printf("  # Benchmark mode\n")
    fmt.Printf("  %s -weights ./weights -image ./test.bin -benchmark -iterations 100\n\n", AppName)
    
    fmt.Println("SUPPORTED IMAGE FORMAT:")
    fmt.Println("  Binary files containing 32×32×3 float32 values (12,288 bytes)")
    fmt.Println("  Data order: Height × Width × Channels (HWC)")
    fmt.Println("  Value range: [0.0, 1.0] (normalized pixel values)")
    
    fmt.Println("\nCIFAR-10 CLASSES:")
    fmt.Println("  0: Airplane   5: Dog")
    fmt.Println("  1: Automobile 6: Frog")
    fmt.Println("  2: Bird       7: Horse")
    fmt.Println("  3: Cat        8: Ship")
    fmt.Println("  4: Deer       9: Truck")
}