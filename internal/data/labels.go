package data

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// LabelLoader handles loading of label data
type LabelLoader struct {
    labelFormat LabelFormat
}

// LabelFormat specifies the format of label files
type LabelFormat int

const (
    OneHotText   LabelFormat = iota // Text files with one-hot encoded labels
    ClassIndex                      // Text files with class indices
    BinaryOneHot                    // Binary files with one-hot encoded labels
)

// NewLabelLoader creates a new label loader
func NewLabelLoader(format LabelFormat) *LabelLoader {
    return &LabelLoader{
        labelFormat: format,
    }
}

// LoadLabel loads a single label from a file
func (ll *LabelLoader) LoadLabel(filename string, numClasses int) ([]int, error) {
    switch ll.labelFormat {
    case OneHotText:
        return ll.loadOneHotText(filename, numClasses)
    case ClassIndex:
        return ll.loadClassIndex(filename, numClasses)
    case BinaryOneHot:
        return ll.loadBinaryOneHot(filename, numClasses)
    default:
        return nil, fmt.Errorf("unsupported label format: %d", ll.labelFormat)
    }
}

// loadOneHotText loads one-hot encoded labels from text file
func (ll *LabelLoader) loadOneHotText(filename string, numClasses int) ([]int, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to open label file %s: %w", filename, err)
    }
    defer file.Close()
    
    scanner := bufio.NewScanner(file)
    if !scanner.Scan() {
        return nil, fmt.Errorf("label file %s is empty", filename)
    }
    
    line := strings.TrimSpace(scanner.Text())
    fields := strings.Fields(line)
    
    if len(fields) != numClasses {
        return nil, fmt.Errorf("label file %s has %d values, expected %d", filename, len(fields), numClasses)
    }
    
    label := make([]int, numClasses)
    for i, field := range fields {
        value, err := strconv.Atoi(field)
        if err != nil {
            return nil, fmt.Errorf("invalid label value '%s' in %s: %w", field, filename, err)
        }
        
        if value != 0 && value != 1 {
            return nil, fmt.Errorf("label value must be 0 or 1, got %d in %s", value, filename)
        }
        
        label[i] = value
    }
    
    // Validate one-hot encoding (exactly one 1)
    sum := 0
    for _, val := range label {
        sum += val
    }
    
    if sum != 1 {
        return nil, fmt.Errorf("invalid one-hot encoding in %s: sum = %d, expected 1", filename, sum)
    }
    
    return label, nil
}

// loadClassIndex loads class index and converts to one-hot
func (ll *LabelLoader) loadClassIndex(filename string, numClasses int) ([]int, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to open label file %s: %w", filename, err)
    }
    defer file.Close()
    
    scanner := bufio.NewScanner(file)
    if !scanner.Scan() {
        return nil, fmt.Errorf("label file %s is empty", filename)
    }
    
    line := strings.TrimSpace(scanner.Text())
    classIndex, err := strconv.Atoi(line)
    if err != nil {
        return nil, fmt.Errorf("invalid class index '%s' in %s: %w", line, filename, err)
    }
    
    if classIndex < 0 || classIndex >= numClasses {
        return nil, fmt.Errorf("class index %d out of range [0, %d) in %s", classIndex, numClasses, filename)
    }
    
    // Convert to one-hot encoding
    label := make([]int, numClasses)
    label[classIndex] = 1
    
    return label, nil
}

// loadBinaryOneHot loads one-hot encoded labels from binary file
func (ll *LabelLoader) loadBinaryOneHot(filename string, numClasses int) ([]int, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to open label file %s: %w", filename, err)
    }
    defer file.Close()
    
    // Read as int32 values
    label := make([]int32, numClasses)
    err = binary.Read(file, binary.LittleEndian, label)
    if err != nil {
        return nil, fmt.Errorf("failed to read binary label from %s: %w", filename, err)
    }
    
    // Convert to int and validate
    result := make([]int, numClasses)
    sum := 0
    for i, val := range label {
        if val != 0 && val != 1 {
            return nil, fmt.Errorf("invalid label value %d at position %d in %s", val, i, filename)
        }
        result[i] = int(val)
        sum += result[i]
    }
    
    if sum != 1 {
        return nil, fmt.Errorf("invalid one-hot encoding in %s: sum = %d, expected 1", filename, sum)
    }
    
    return result, nil
}

// LoadLabelBatch loads multiple labels from a directory
func (ll *LabelLoader) LoadLabelBatch(labelDir string, numLabels, numClasses int) ([][]int, error) {
    labels := make([][]int, numLabels)
    
    for i := 0; i < numLabels; i++ {
        filename := filepath.Join(labelDir, fmt.Sprintf("label_test_%d.txt", i))
        
        label, err := ll.LoadLabel(filename, numClasses)
        if err != nil {
            return nil, fmt.Errorf("failed to load label %d: %w", i, err)
        }
        
        labels[i] = label
    }
    
    return labels, nil
}

// ConvertOneHotToClassIndex converts one-hot encoded label to class index
func ConvertOneHotToClassIndex(oneHot []int) int {
    for i, val := range oneHot {
        if val == 1 {
            return i
        }
    }
    return -1 // Invalid one-hot encoding
}

// ConvertClassIndexToOneHot converts class index to one-hot encoded label
func ConvertClassIndexToOneHot(classIndex, numClasses int) []int {
    if classIndex < 0 || classIndex >= numClasses {
        return nil
    }
    
    oneHot := make([]int, numClasses)
    oneHot[classIndex] = 1
    return oneHot
}

// ValidateLabel checks if a label is valid one-hot encoding
func ValidateLabel(label []int, numClasses int) error {
    if len(label) != numClasses {
        return fmt.Errorf("label length %d doesn't match expected %d", len(label), numClasses)
    }
    
    sum := 0
    for i, val := range label {
        if val != 0 && val != 1 {
            return fmt.Errorf("invalid label value %d at position %d", val, i)
        }
        sum += val
    }
    
    if sum != 1 {
        return fmt.Errorf("invalid one-hot encoding: sum = %d, expected 1", sum)
    }
    
    return nil
}

// GetClassDistribution computes the distribution of classes in a batch of labels
func GetClassDistribution(labels [][]int) map[int]int {
    distribution := make(map[int]int)
    
    for _, label := range labels {
        classIndex := ConvertOneHotToClassIndex(label)
        if classIndex >= 0 {
            distribution[classIndex]++
        }
    }
    
    return distribution
}

// PrintClassDistribution prints the class distribution in a readable format
func PrintClassDistribution(distribution map[int]int, classNames []string) {
    fmt.Println("Class Distribution:")
    for classIndex, count := range distribution {
        className := fmt.Sprintf("Class %d", classIndex)
        if classIndex < len(classNames) {
            className = classNames[classIndex]
        }
        fmt.Printf("  %s: %d\n", className, count)
    }
}