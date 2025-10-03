package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"duchm1606/gocnn/internal/config"
	"duchm1606/gocnn/internal/data"
	"duchm1606/gocnn/internal/metrics"
	"duchm1606/gocnn/internal/model"
)

// Version information
const (
    AppName    = "gocnn-benchmark"
    AppVersion = "1.0.0"
    AppDesc    = "TinyCNN benchmarking and evaluation tool for CIFAR-10"
)

// Command line flags
var (
    weightsPath = flag.String("weights", "", "Path to model weights directory (required)")
    imagesPath  = flag.String("images", "", "Path to test images directory (required)")
    labelsPath  = flag.String("labels", "", "Path to test labels directory (required)")
    configPath  = flag.String("config", "configs/cifar10.yaml", "Path to model configuration file")
    outputPath  = flag.String("output", "", "Path to save detailed results (optional)")
    
    numSamples  = flag.Int("samples", 100, "Number of test samples to evaluate")
    numWorkers  = flag.Int("workers", 4, "Number of parallel workers")
    batchSize   = flag.Int("batch", 1, "Batch size for evaluation")
    
    reportFormat = flag.String("format", "text", "Output format: text, csv, json")
    verbose      = flag.Bool("verbose", false, "Enable verbose output")
    quiet        = flag.Bool("quiet", false, "Suppress non-essential output")
    showMatrix   = flag.Bool("matrix", false, "Show confusion matrix")
    showTiming   = flag.Bool("timing", true, "Show detailed timing information")
    
    profileCPU = flag.String("cpuprofile", "", "Write CPU profile to file")
    profileMem = flag.String("memprofile", "", "Write memory profile to file")
    
    showVersion = flag.Bool("version", false, "Show version information")
    showHelp    = flag.Bool("help", false, "Show detailed help")
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

    // Set up profiling if requested
    if *profileCPU != "" {
        if err := startCPUProfile(*profileCPU); err != nil {
            fmt.Fprintf(os.Stderr, "Failed to start CPU profiling: %v\n", err)
            os.Exit(1)
        }
        defer stopCPUProfile()
    }

    // Run benchmark
    if err := runBenchmark(); err != nil {
        fmt.Fprintf(os.Stderr, "Benchmark failed: %v\n", err)
        os.Exit(1)
    }

    // Write memory profile if requested
    if *profileMem != "" {
        if err := writeMemProfile(*profileMem); err != nil {
            fmt.Fprintf(os.Stderr, "Failed to write memory profile: %v\n", err)
        }
    }
}

// validateArgs validates command line arguments
func validateArgs() error {
    if *weightsPath == "" {
        return fmt.Errorf("weights path is required (use -weights)")
    }

    if *imagesPath == "" {
        return fmt.Errorf("images path is required (use -images)")
    }

    if *labelsPath == "" {
        return fmt.Errorf("labels path is required (use -labels)")
    }

    // Check if directories/files exist
    paths := map[string]string{
        "weights directory": *weightsPath,
        "images directory":  *imagesPath,
        "labels directory":  *labelsPath,
        "config file":       *configPath,
    }

    for desc, path := range paths {
        if _, err := os.Stat(path); os.IsNotExist(err) {
            return fmt.Errorf("%s does not exist: %s", desc, path)
        }
    }

    // Validate numeric parameters
    if *numSamples <= 0 {
        return fmt.Errorf("number of samples must be positive, got %d", *numSamples)
    }

    if *numWorkers <= 0 {
        return fmt.Errorf("number of workers must be positive, got %d", *numWorkers)
    }

    if *batchSize <= 0 {
        return fmt.Errorf("batch size must be positive, got %d", *batchSize)
    }

    // Validate report format
    validFormats := map[string]bool{
        "text": true,
        "csv":  true,
        "json": true,
    }

    if !validFormats[*reportFormat] {
        return fmt.Errorf("invalid report format: %s (valid: text, csv, json)", *reportFormat)
    }

    return nil
}

// runBenchmark executes the main benchmarking workflow
func runBenchmark() error {
    if !*quiet {
        fmt.Printf("Starting %s v%s\n", AppName, AppVersion)
        fmt.Printf("Evaluating %d samples with %d workers\n\n", *numSamples, *numWorkers)
    }

    // Load configuration
    if *verbose {
        fmt.Printf("Loading configuration from %s...\n", *configPath)
    }

    cfg, err := config.Load(*configPath)
    if err != nil {
        return fmt.Errorf("failed to load configuration: %w", err)
    }

    // Load model
    if !*quiet {
        fmt.Printf("Loading CNN model from %s...\n", *weightsPath)
    }

    start := time.Now()
    cnn, err := model.NewTinyCNN(*weightsPath)
    if err != nil {
        return fmt.Errorf("failed to load model: %w", err)
    }
    loadTime := time.Since(start)

    if *verbose {
        fmt.Printf("Model loaded in %v\n", loadTime)
        printModelInfo(cnn)
    }

    // Load test data
    if !*quiet {
        fmt.Printf("Loading test data (%d samples)...\n", *numSamples)
    }

    start = time.Now()
    testData, err := loadTestData(cfg)
    if err != nil {
        return fmt.Errorf("failed to load test data: %w", err)
    }
    dataLoadTime := time.Since(start)

    if *verbose {
        fmt.Printf("Test data loaded in %v\n", dataLoadTime)
        fmt.Printf("Images: %d, Labels: %d\n", len(testData.Images), len(testData.Labels))
    }

    // Run evaluation
    if !*quiet {
        fmt.Printf("\nRunning evaluation...\n")
    }

    evaluator := metrics.NewEvaluator(*numWorkers, *verbose)
    start = time.Now()
    results, err := evaluator.EvaluateModel(cnn, testData.Images, testData.Labels)
    if err != nil {
        return fmt.Errorf("evaluation failed: %w", err)
    }
    evalTime := time.Since(start)

    if !*quiet {
        fmt.Printf("Evaluation completed in %v\n\n", evalTime)
    }

    // Generate and display report
    reporter := NewReporter(*reportFormat, cfg.Model.ClassNames)
    return reporter.GenerateReport(results, evalTime, *outputPath)
}

// loadTestData loads test images and labels
func loadTestData(cfg *config.Config) (*data.DataBatch, error) {
    dataManager := data.NewDataManager("", data.BinaryFloat32, data.OneHotText)
    
    return dataManager.LoadTestBatch(
        *imagesPath,
        *labelsPath,
        *numSamples,
        cfg.Model.InputHeight,
        cfg.Model.InputWidth,
        cfg.Model.InputChannels,
        cfg.Model.NumClasses,
    )
}

// printModelInfo displays model information
func printModelInfo(cnn *model.TinyCNN) {
    info := cnn.GetModelInfo()
    
    fmt.Printf("Model Information:\n")
    fmt.Printf("  Input Size: %d×%d×%d\n",
        info.Architecture.InputHeight,
        info.Architecture.InputWidth,
        info.Architecture.InputChannels)
    fmt.Printf("  Output Classes: %d\n", info.Architecture.NumClasses)
    fmt.Printf("  Total Parameters: %d\n", info.TotalParameters)
    fmt.Printf("  Total Layers: %d\n", len(info.Architecture.Layers))
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
    fmt.Printf("  %s -weights <path> -images <path> -labels <path> [options]\n\n", AppName)
    
    fmt.Println("REQUIRED:")
    fmt.Println("  -weights <path>    Path to directory containing model weights")
    fmt.Println("  -images <path>     Path to directory containing test images")
    fmt.Println("  -labels <path>     Path to directory containing test labels")
    
    fmt.Println("\nOPTIONS:")
    fmt.Println("  -config <path>     Path to model configuration file (default: configs/cifar10.yaml)")
    fmt.Println("  -output <path>     Save detailed results to file")
    fmt.Println("  -samples <n>       Number of test samples to evaluate (default: 100)")
    fmt.Println("  -workers <n>       Number of parallel workers (default: 4)")
    fmt.Println("  -batch <n>         Batch size for evaluation (default: 1)")
    fmt.Println("  -format <fmt>      Output format: text, csv, json (default: text)")
    fmt.Println("  -verbose           Enable verbose output")
    fmt.Println("  -quiet             Suppress non-essential output")
    fmt.Println("  -matrix            Show confusion matrix")
    fmt.Println("  -timing            Show detailed timing information (default: true)")
    fmt.Println("  -cpuprofile <file> Write CPU profile to file")
    fmt.Println("  -memprofile <file> Write memory profile to file")
    fmt.Println("  -version           Show version information")
    fmt.Println("  -help              Show this help message")
    
    fmt.Println("\nEXAMPLES:")
    fmt.Printf("  # Basic evaluation\n")
    fmt.Printf("  %s -weights ./weights -images ./test_images -labels ./test_labels\n\n", AppName)
    
    fmt.Printf("  # Comprehensive evaluation with detailed output\n")
    fmt.Printf("  %s -weights ./weights -images ./test_images -labels ./test_labels \\\n", AppName)
    fmt.Printf("    -samples 1000 -workers 8 -verbose -matrix -output results.json\n\n")
    
    fmt.Printf("  # Performance profiling\n")
    fmt.Printf("  %s -weights ./weights -images ./test_images -labels ./test_labels \\\n", AppName)
    fmt.Printf("    -cpuprofile cpu.prof -memprofile mem.prof\n\n")
    
    fmt.Println("METRICS COMPUTED:")
    fmt.Println("  - Top-1 Accuracy (primary metric)")
    fmt.Println("  - Top-5 Accuracy")
    fmt.Println("  - Per-class Accuracy")
    fmt.Println("  - Confusion Matrix")
    fmt.Println("  - Inference Timing Statistics")
    fmt.Println("  - Throughput (samples/second)")
    
    fmt.Println("\nOUTPUT FORMATS:")
    fmt.Println("  text - Human-readable console output")
    fmt.Println("  csv  - Comma-separated values for analysis")
    fmt.Println("  json - Structured JSON for programmatic use")
}