package metrics

import (
	"duchm1606/gocnn/internal/model"
	"duchm1606/gocnn/internal/tensor"
	"fmt"
	"sync"
	"time"
)

// Evaluator performs comprehensive model evaluation
type Evaluator struct {
    numWorkers int
    verbose    bool
}

// NewEvaluator creates a new evaluator
func NewEvaluator(numWorkers int, verbose bool) *Evaluator {
    return &Evaluator{
        numWorkers: numWorkers,
        verbose:    verbose,
    }
}

// EvaluationResult holds comprehensive evaluation results
type EvaluationResult struct {
    // Basic metrics
    TotalSamples       int     `json:"total_samples"`
    CorrectPredictions int     `json:"correct_predictions"`
    Top1Accuracy       float64 `json:"top1_accuracy"`
    Top5Accuracy       float64 `json:"top5_accuracy"`
    
    // Per-class metrics
    ClassAccuracies    []float64 `json:"class_accuracies"`
    ClassPrecisions    []float64 `json:"class_precisions"`
    ClassRecalls       []float64 `json:"class_recalls"`
    ClassF1Scores      []float64 `json:"class_f1_scores"`
    
    // Confusion matrix
    ConfusionMatrix    [][]int   `json:"confusion_matrix"`
    
    // Timing metrics
    TotalInferenceTime time.Duration            `json:"total_inference_time"`
    AverageInferenceTime time.Duration          `json:"average_inference_time"`
    MinInferenceTime   time.Duration            `json:"min_inference_time"`
    MaxInferenceTime   time.Duration            `json:"max_inference_time"`
    LayerTimings       map[string]time.Duration `json:"layer_timings"`
    
    // Throughput metrics
    Throughput         float64 `json:"throughput"` // samples per second
    
    // Individual predictions (for detailed analysis)
    Predictions        []PredictionDetail `json:"predictions,omitempty"`
}

// PredictionDetail holds information about a single prediction
type PredictionDetail struct {
    SampleIndex    int           `json:"sample_index"`
    TrueClass      int           `json:"true_class"`
    PredictedClass int           `json:"predicted_class"`
    Confidence     float32       `json:"confidence"`
    Probabilities  []float32     `json:"probabilities"`
    InferenceTime  time.Duration `json:"inference_time"`
    Correct        bool          `json:"correct"`
}

// EvaluateModel performs comprehensive evaluation of the model
func (e *Evaluator) EvaluateModel(cnn *model.TinyCNN, images []*tensor.FeatureMap, labels [][]int) (*EvaluationResult, error) {
    numSamples := len(images)
    if numSamples != len(labels) {
        return nil, fmt.Errorf("number of images (%d) doesn't match number of labels (%d)", numSamples, len(labels))
    }

    // Initialize result
    result := &EvaluationResult{
        TotalSamples:    numSamples,
        ConfusionMatrix: make([][]int, 10),
        LayerTimings:    make(map[string]time.Duration),
        Predictions:     make([]PredictionDetail, numSamples),
    }

    for i := range result.ConfusionMatrix {
        result.ConfusionMatrix[i] = make([]int, 10)
    }

    // Create work channels
    jobs := make(chan int, numSamples)
    results := make(chan PredictionDetail, numSamples)

    // Start workers
    var wg sync.WaitGroup
    for w := 0; w < e.numWorkers; w++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for sampleIdx := range jobs {
                detail := e.evaluateSample(cnn, images[sampleIdx], labels[sampleIdx], sampleIdx)
                results <- detail
            }
        }()
    }

    // Send jobs
    go func() {
        for i := 0; i < numSamples; i++ {
            jobs <- i
        }
        close(jobs)
    }()

    // Wait for workers to complete
    go func() {
        wg.Wait()
        close(results)
    }()

    // Collect results
    for detail := range results {
        result.Predictions[detail.SampleIndex] = detail
        
        if e.verbose && detail.SampleIndex%10 == 0 {
            fmt.Printf("  Processed %d/%d samples\n", detail.SampleIndex+1, numSamples)
        }
    }

    // Compute aggregate metrics
    e.computeAggregateMetrics(result)

    return result, nil
}

// evaluateSample evaluates a single sample
func (e *Evaluator) evaluateSample(cnn *model.TinyCNN, image *tensor.FeatureMap, label []int, sampleIdx int) PredictionDetail {
    // Convert feature map to flat array
    imageData := image.Data

    // Run inference
    start := time.Now()
    prediction, err := cnn.Predict(imageData)
    inferenceTime := time.Since(start)

    if err != nil {
        // Handle error case
        return PredictionDetail{
            SampleIndex:   sampleIdx,
            TrueClass:     argmaxInt(label),
            PredictedClass: -1,
            Confidence:    0,
            InferenceTime: inferenceTime,
            Correct:       false,
        }
    }

    trueClass := argmaxInt(label)
    correct := prediction.PredictedClass == trueClass

    return PredictionDetail{
        SampleIndex:    sampleIdx,
        TrueClass:      trueClass,
        PredictedClass: prediction.PredictedClass,
        Confidence:     prediction.Confidence,
        Probabilities:  prediction.Probabilities,
        InferenceTime:  inferenceTime,
        Correct:        correct,
    }
}

// computeAggregateMetrics computes all aggregate metrics from individual predictions
func (e *Evaluator) computeAggregateMetrics(result *EvaluationResult) {
    numClasses := len(result.ConfusionMatrix)
    
    // Initialize timing metrics
    result.MinInferenceTime = result.Predictions[0].InferenceTime
    result.MaxInferenceTime = result.Predictions[0].InferenceTime
    var totalTime time.Duration

    // Count correct predictions and build confusion matrix
    for _, pred := range result.Predictions {
        if pred.Correct {
            result.CorrectPredictions++
        }

        // Update confusion matrix
        if pred.TrueClass >= 0 && pred.TrueClass < numClasses &&
           pred.PredictedClass >= 0 && pred.PredictedClass < numClasses {
            result.ConfusionMatrix[pred.TrueClass][pred.PredictedClass]++
        }

        // Update timing metrics
        totalTime += pred.InferenceTime
        if pred.InferenceTime < result.MinInferenceTime {
            result.MinInferenceTime = pred.InferenceTime
        }
        if pred.InferenceTime > result.MaxInferenceTime {
            result.MaxInferenceTime = pred.InferenceTime
        }
    }

    // Compute accuracy metrics
    result.Top1Accuracy = float64(result.CorrectPredictions) / float64(result.TotalSamples)
    result.Top5Accuracy = e.computeTop5Accuracy(result.Predictions)

    // Compute timing metrics
    result.TotalInferenceTime = totalTime
    result.AverageInferenceTime = totalTime / time.Duration(result.TotalSamples)
    result.Throughput = float64(result.TotalSamples) / totalTime.Seconds()

    // Compute per-class metrics
    result.ClassAccuracies = e.computeClassAccuracies(result.ConfusionMatrix)
    result.ClassPrecisions = e.computeClassPrecisions(result.ConfusionMatrix)
    result.ClassRecalls = e.computeClassRecalls(result.ConfusionMatrix)
    result.ClassF1Scores = e.computeClassF1Scores(result.ClassPrecisions, result.ClassRecalls)
}

// computeTop5Accuracy computes top-5 accuracy
func (e *Evaluator) computeTop5Accuracy(predictions []PredictionDetail) float64 {
    correct := 0
    
    for _, pred := range predictions {
        if len(pred.Probabilities) >= 5 {
            top5 := getTop5Indices(pred.Probabilities)
            for _, idx := range top5 {
                if idx == pred.TrueClass {
                    correct++
                    break
                }
            }
        }
    }
    
    return float64(correct) / float64(len(predictions))
}

// computeClassAccuracies computes per-class accuracy
func (e *Evaluator) computeClassAccuracies(confusionMatrix [][]int) []float64 {
    numClasses := len(confusionMatrix)
    accuracies := make([]float64, numClasses)
    
    for i := 0; i < numClasses; i++ {
        totalForClass := 0
        for j := 0; j < numClasses; j++ {
            totalForClass += confusionMatrix[i][j]
        }
        
        if totalForClass > 0 {
            accuracies[i] = float64(confusionMatrix[i][i]) / float64(totalForClass)
        }
    }
    
    return accuracies
}

// computeClassPrecisions computes per-class precision
func (e *Evaluator) computeClassPrecisions(confusionMatrix [][]int) []float64 {
    numClasses := len(confusionMatrix)
    precisions := make([]float64, numClasses)
    
    for j := 0; j < numClasses; j++ {
        totalPredictedAsClass := 0
        for i := 0; i < numClasses; i++ {
            totalPredictedAsClass += confusionMatrix[i][j]
        }
        
        if totalPredictedAsClass > 0 {
            precisions[j] = float64(confusionMatrix[j][j]) / float64(totalPredictedAsClass)
        }
    }
    
    return precisions
}

// computeClassRecalls computes per-class recall (same as accuracy for multiclass)
func (e *Evaluator) computeClassRecalls(confusionMatrix [][]int) []float64 {
    return e.computeClassAccuracies(confusionMatrix)
}

// computeClassF1Scores computes per-class F1 scores
func (e *Evaluator) computeClassF1Scores(precisions, recalls []float64) []float64 {
    numClasses := len(precisions)
    f1Scores := make([]float64, numClasses)
    
    for i := 0; i < numClasses; i++ {
        if precisions[i]+recalls[i] > 0 {
            f1Scores[i] = 2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i])
        }
    }
    
    return f1Scores
}

// Helper functions
func argmaxInt(slice []int) int {
    maxIdx := 0
    maxVal := slice[0]
    
    for i, val := range slice[1:] {
        if val > maxVal {
            maxVal = val
            maxIdx = i + 1
        }
    }
    
    return maxIdx
}

func getTop5Indices(probabilities []float32) []int {
    type IndexValue struct {
        Index int
        Value float32
    }
    
    // Create index-value pairs
    pairs := make([]IndexValue, len(probabilities))
    for i, val := range probabilities {
        pairs[i] = IndexValue{i, val}
    }
    
    // Sort by value (descending)
    for i := 0; i < len(pairs); i++ {
        for j := i + 1; j < len(pairs); j++ {
            if pairs[j].Value > pairs[i].Value {
                pairs[i], pairs[j] = pairs[j], pairs[i]
            }
        }
    }
    
    // Return top 5 indices
    top5 := make([]int, 5)
    for i := 0; i < 5 && i < len(pairs); i++ {
        top5[i] = pairs[i].Index
    }
    
    return top5
}