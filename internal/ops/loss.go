package ops

import (
	"math"
)

// Implement Loss Functions

// CrossEntropyLoss computes cross-entropy loss between predictions and labels
// predictions: softmax probabilities [0, 1]
// labels: one-hot encoded ground truth
func CrossEntropyLoss(predictions, labels []float32) float32 {
    if len(predictions) != len(labels) {
        panic("predictions and labels must have same length")
    }
    
    var loss float32
    for i, pred := range predictions {
        if labels[i] > 0 {
            // Avoid log(0) by adding small epsilon
            if pred < 1e-15 {
                pred = 1e-15
            }
            loss += -labels[i] * float32(math.Log(float64(pred)))
        }
    }
    
    return loss
}


// CrossEntropyLossFromLogits computes cross-entropy loss from logits
// This is more numerically stable than computing softmax first
func CrossEntropyLossFromLogits(logits, labels []float32) float32 {
    if len(logits) != len(labels) {
        panic("logits and labels must have same length")
    }
    
    // Apply log-softmax for numerical stability
    logSoftmax := LogSoftmax(logits)
    
    var loss float32
    for i, logProb := range logSoftmax {
        if labels[i] > 0 {
            loss += -labels[i] * logProb
        }
    }
    
    return loss
}

// SparseCrossEntropyLoss computes cross-entropy loss with sparse labels
// predictions: softmax probabilities
// trueClassIndex: index of the true class (not one-hot encoded)
func SparseCrossEntropyLoss(predictions []float32, trueClassIndex int) float32 {
    if trueClassIndex < 0 || trueClassIndex >= len(predictions) {
        panic("true class index out of bounds")
    }
    
    pred := predictions[trueClassIndex]
    if pred < 1e-15 {
        pred = 1e-15
    }
    
    return -float32(math.Log(float64(pred)))
}

// MeanSquaredError computes MSE loss between predictions and targets
func MeanSquaredError(predictions, targets []float32) float32 {
    if len(predictions) != len(targets) {
        panic("predictions and targets must have same length")
    }
    
    var sumSquaredError float32
    for i, pred := range predictions {
        diff := pred - targets[i]
        sumSquaredError += diff * diff
    }
    
    return sumSquaredError / float32(len(predictions))
}

// Accuracy computes classification accuracy
func Accuracy(predictions, labels []float32) float32 {
    if len(predictions) != len(labels) {
        panic("predictions and labels must have same length")
    }
    
    predictedClass := Argmax(predictions)
    trueClass := Argmax(labels)
    
    if predictedClass == trueClass {
        return 1.0
    }
    return 0.0
}

// Top5Accuracy computes top-5 accuracy
func Top5Accuracy(predictions, labels []float32) float32 {
    if len(predictions) != len(labels) {
        panic("predictions and labels must have same length")
    }
    
    top5Indices := ArgmaxTop5(predictions)
    trueClass := Argmax(labels)
    
    for _, predictedClass := range top5Indices {
        if predictedClass == trueClass {
            return 1.0
        }
    }
    
    return 0.0
}