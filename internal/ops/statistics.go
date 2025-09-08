package ops

import (
	"math"
	"sort"
)

// Argmax returns the index of the maximum value
// TODO: Duplicate with tensor.Argmax. Consider merging.
func Argmax(slice []float32) int {
    if len(slice) == 0 {
        return -1
    }
    
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

// ArgmaxTop5 returns indices of top 5 maximum values
func ArgmaxTop5(slice []float32) []int {
    if len(slice) == 0 {
        return []int{}
    }
    
    // Create pairs of (value, index)
    type valueIndex struct {
        value float32
        index int
    }
    
    pairs := make([]valueIndex, len(slice))
    for i, val := range slice {
        pairs[i] = valueIndex{val, i}
    }
    
    // Sort by value (descending)
    sort.Slice(pairs, func(i, j int) bool {
        return pairs[i].value > pairs[j].value
    })
    
    // Return top 5 indices
    topK := 5
    if len(pairs) < topK {
        topK = len(pairs)
    }
    
    result := make([]int, topK)
    for i := 0; i < topK; i++ {
        result[i] = pairs[i].index
    }
    
    return result
}

// Max returns the maximum value in a slice
func Max(slice []float32) float32 {
    if len(slice) == 0 {
        return 0
    }
    
    maxVal := slice[0]
    for _, val := range slice[1:] {
        if val > maxVal {
            maxVal = val
        }
    }
    
    return maxVal
}

// Min returns the minimum value in a slice
func Min(slice []float32) float32 {
    if len(slice) == 0 {
        return 0
    }
    
    minVal := slice[0]
    for _, val := range slice[1:] {
        if val < minVal {
            minVal = val
        }
    }
    
    return minVal
}

// Sum returns the sum of all values
func Sum(slice []float32) float32 {
    var sum float32
    for _, val := range slice {
        sum += val
    }
    return sum
}

// Mean returns the average value
func Mean(slice []float32) float32 {
    if len(slice) == 0 {
        return 0
    }
    return Sum(slice) / float32(len(slice))
}

// Variance returns the variance of the slice
func Variance(slice []float32) float32 {
    if len(slice) <= 1 {
        return 0
    }
    
    mean := Mean(slice)
    var sumSquaredDiff float32
    
    for _, val := range slice {
        diff := val - mean
        sumSquaredDiff += diff * diff
    }
    
    return sumSquaredDiff / float32(len(slice)-1)
}

// StandardDeviation returns the standard deviation
func StandardDeviation(slice []float32) float32 {
    return float32(math.Sqrt(float64(Variance(slice))))
}

// Normalize normalizes a slice to have mean=0 and std=1
func Normalize(slice []float32) []float32 {
    if len(slice) == 0 {
        return []float32{}
    }
    
    mean := Mean(slice)
    std := StandardDeviation(slice)
    
    if std == 0 {
        // If std is 0, all values are the same, return zeros
        result := make([]float32, len(slice))
        return result
    }
    
    result := make([]float32, len(slice))
    for i, val := range slice {
        result[i] = (val - mean) / std
    }
    
    return result
}

// NormalizeInPlace normalizes a slice in-place
func NormalizeInPlace(slice []float32) {
    if len(slice) == 0 {
        return
    }
    
    mean := Mean(slice)
    std := StandardDeviation(slice)
    
    if std == 0 {
        // If std is 0, set all values to 0
        for i := range slice {
            slice[i] = 0
        }
        return
    }
    
    for i, val := range slice {
        slice[i] = (val - mean) / std
    }
}

// Clip constrains values to be within [min, max]
func Clip(slice []float32, minVal, maxVal float32) []float32 {
    result := make([]float32, len(slice))
    for i, val := range slice {
        if val < minVal {
            result[i] = minVal
        } else if val > maxVal {
            result[i] = maxVal
        } else {
            result[i] = val
        }
    }
    return result
}

// ClipInPlace constrains values in-place
func ClipInPlace(slice []float32, minVal, maxVal float32) {
    for i, val := range slice {
        if val < minVal {
            slice[i] = minVal
        } else if val > maxVal {
            slice[i] = maxVal
        }
    }
}