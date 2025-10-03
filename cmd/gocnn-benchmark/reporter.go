package main

import (
	"duchm1606/gocnn/internal/metrics"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// Reporter generates evaluation reports in different formats
type Reporter struct {
    format     string
    classNames []string
}

// NewReporter creates a new reporter
func NewReporter(format string, classNames []string) *Reporter {
    return &Reporter{
        format:     format,
        classNames: classNames,
    }
}

// GenerateReport generates and outputs the evaluation report
func (r *Reporter) GenerateReport(result *metrics.EvaluationResult, evalTime time.Duration, outputPath string) error {
    switch r.format {
    case "text":
        return r.generateTextReport(result, evalTime, outputPath)
    case "csv":
        return r.generateCSVReport(result, outputPath)
    case "json":
        return r.generateJSONReport(result, outputPath)
    default:
        return fmt.Errorf("unsupported format: %s", r.format)
    }
}

// generateTextReport generates a human-readable text report
func (r *Reporter) generateTextReport(result *metrics.EvaluationResult, evalTime time.Duration, outputPath string) error {
    var output *os.File
    var err error
    
    if outputPath != "" {
        output, err = os.Create(outputPath)
        if err != nil {
            return fmt.Errorf("failed to create output file: %w", err)
        }
        defer output.Close()
    } else {
        output = os.Stdout
    }
    
    // Write header
    fmt.Fprintf(output, "TinyCNN Evaluation Report\n")
    fmt.Fprintf(output, "=========================\n\n")
    fmt.Fprintf(output, "Generated: %s\n", time.Now().Format("2006-01-02 15:04:05"))
    fmt.Fprintf(output, "Evaluation Time: %v\n\n", evalTime)
    
    // Overall metrics
    fmt.Fprintf(output, "Overall Performance:\n")
    fmt.Fprintf(output, "  Total Samples: %d\n", result.TotalSamples)
    fmt.Fprintf(output, "  Correct Predictions: %d\n", result.CorrectPredictions)
    fmt.Fprintf(output, "  Top-1 Accuracy: %.4f (%.2f%%)\n", result.Top1Accuracy, result.Top1Accuracy*100)
    fmt.Fprintf(output, "  Top-5 Accuracy: %.4f (%.2f%%)\n", result.Top5Accuracy, result.Top5Accuracy*100)
    fmt.Fprintf(output, "\n")
    
    // Timing metrics
    if *showTiming {
        fmt.Fprintf(output, "Timing Performance:\n")
        fmt.Fprintf(output, "  Total Inference Time: %v\n", result.TotalInferenceTime)
        fmt.Fprintf(output, "  Average Inference Time: %v\n", result.AverageInferenceTime)
        fmt.Fprintf(output, "  Min Inference Time: %v\n", result.MinInferenceTime)
        fmt.Fprintf(output, "  Max Inference Time: %v\n", result.MaxInferenceTime)
        fmt.Fprintf(output, "  Throughput: %.2f samples/second\n", result.Throughput)
        fmt.Fprintf(output, "\n")
    }
    
    // Per-class metrics
    fmt.Fprintf(output, "Per-Class Performance:\n")
    fmt.Fprintf(output, "  Class           Accuracy  Precision   Recall     F1-Score\n")
    fmt.Fprintf(output, "  --------------------------------------------------------\n")
    
    for i, className := range r.classNames {
        if i < len(result.ClassAccuracies) {
            fmt.Fprintf(output, "  %-13s   %.4f     %.4f     %.4f     %.4f\n",
                className,
                result.ClassAccuracies[i],
                result.ClassPrecisions[i],
                result.ClassRecalls[i],
                result.ClassF1Scores[i])
        }
    }
    fmt.Fprintf(output, "\n")
    
    // Confusion matrix
    if *showMatrix {
        fmt.Fprintf(output, "Confusion Matrix:\n")
        fmt.Fprintf(output, "  Predicted →\n")
        fmt.Fprintf(output, "T ↓   ")
        
        // Header
        for i := 0; i < len(r.classNames) && i < len(result.ConfusionMatrix); i++ {
            fmt.Fprintf(output, "%8s", fmt.Sprintf("C%d", i))
        }
        fmt.Fprintf(output, "\n")
        
        // Matrix rows
        for i, row := range result.ConfusionMatrix {
            if i < len(r.classNames) {
                fmt.Fprintf(output, "C%-2d ", i)
                for j, count := range row {
                    if j < len(r.classNames) {
                        fmt.Fprintf(output, "%8d", count)
                    }
                }
                fmt.Fprintf(output, "  (%s)\n", r.classNames[i])
            }
        }
        fmt.Fprintf(output, "\n")
    }
    
    // Summary statistics
    avgAccuracy := 0.0
    for _, acc := range result.ClassAccuracies {
        avgAccuracy += acc
    }
    avgAccuracy /= float64(len(result.ClassAccuracies))
    
    fmt.Fprintf(output, "Summary Statistics:\n")
    fmt.Fprintf(output, "  Average Class Accuracy: %.4f (%.2f%%)\n", avgAccuracy, avgAccuracy*100)
    fmt.Fprintf(output, "  Standard Deviation: %.4f\n", r.computeStdDev(result.ClassAccuracies, avgAccuracy))
    
    if outputPath != "" {
        fmt.Printf("Text report saved to: %s\n", outputPath)
    }
    
    return nil
}

// generateCSVReport generates a CSV report for analysis
func (r *Reporter) generateCSVReport(result *metrics.EvaluationResult, outputPath string) error {
    if outputPath == "" {
        outputPath = "evaluation_results.csv"
    }
    
    file, err := os.Create(outputPath)
    if err != nil {
        return fmt.Errorf("failed to create CSV file: %w", err)
    }
    defer file.Close()
    
    writer := csv.NewWriter(file)
    defer writer.Flush()
    
    // Write overall metrics
    writer.Write([]string{"Metric", "Value"})
    writer.Write([]string{"Total Samples", fmt.Sprintf("%d", result.TotalSamples)})
    writer.Write([]string{"Correct Predictions", fmt.Sprintf("%d", result.CorrectPredictions)})
    writer.Write([]string{"Top-1 Accuracy", fmt.Sprintf("%.6f", result.Top1Accuracy)})
    writer.Write([]string{"Top-5 Accuracy", fmt.Sprintf("%.6f", result.Top5Accuracy)})
    writer.Write([]string{"Throughput", fmt.Sprintf("%.6f", result.Throughput)})
    writer.Write([]string{""}) // Empty row
    
    // Write per-class metrics
    writer.Write([]string{"Class", "Accuracy", "Precision", "Recall", "F1-Score"})
    for i, className := range r.classNames {
        if i < len(result.ClassAccuracies) {
            writer.Write([]string{
                className,
                fmt.Sprintf("%.6f", result.ClassAccuracies[i]),
                fmt.Sprintf("%.6f", result.ClassPrecisions[i]),
                fmt.Sprintf("%.6f", result.ClassRecalls[i]),
                fmt.Sprintf("%.6f", result.ClassF1Scores[i]),
            })
        }
    }
    
    fmt.Printf("CSV report saved to: %s\n", outputPath)
    return nil
}

// generateJSONReport generates a JSON report for programmatic use
func (r *Reporter) generateJSONReport(result *metrics.EvaluationResult, outputPath string) error {
    if outputPath == "" {
        outputPath = "evaluation_results.json"
    }
    
    // Create enhanced result with metadata
    enhancedResult := struct {
        *metrics.EvaluationResult
        Metadata struct {
            GeneratedAt time.Time `json:"generated_at"`
            ClassNames  []string  `json:"class_names"`
            Format      string    `json:"format"`
        } `json:"metadata"`
    }{
        EvaluationResult: result,
    }
    
    enhancedResult.Metadata.GeneratedAt = time.Now()
    enhancedResult.Metadata.ClassNames = r.classNames
    enhancedResult.Metadata.Format = "TinyCNN Evaluation v1.0"
    
    file, err := os.Create(outputPath)
    if err != nil {
        return fmt.Errorf("failed to create JSON file: %w", err)
    }
    defer file.Close()
    
    encoder := json.NewEncoder(file)
    encoder.SetIndent("", "  ")
    
    err = encoder.Encode(enhancedResult)
    if err != nil {
        return fmt.Errorf("failed to encode JSON: %w", err)
    }
    
    fmt.Printf("JSON report saved to: %s\n", outputPath)
    return nil
}

// computeStdDev computes standard deviation
func (r *Reporter) computeStdDev(values []float64, mean float64) float64 {
    if len(values) <= 1 {
        return 0.0
    }
    
    var sumSquaredDiff float64
    for _, val := range values {
        diff := val - mean
        sumSquaredDiff += diff * diff
    }
    
    variance := sumSquaredDiff / float64(len(values)-1)
    return variance
}
