package data

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func createTestWeightFile(t testing.TB, filename string, size, channels, filters int) {
    file, err := os.Create(filename)
    if err != nil {
        t.Fatalf("Failed to create test file: %v", err)
    }
    defer file.Close()
    
    // Write test data in the correct order
    for h := 0; h < size; h++ {
        for w := 0; w < size; w++ {
            for c := 0; c < channels; c++ {
                for f := 0; f < filters; f++ {
                    value := float32(h*1000 + w*100 + c*10 + f)
                    binary.Write(file, binary.LittleEndian, value)
                }
            }
        }
    }
}

func createTestImageFile(t testing.TB, filename string, height, width, channels int) {
    file, err := os.Create(filename)
    if err != nil {
        t.Fatalf("Failed to create test file: %v", err)
    }
    defer file.Close()
    
    // Write test data in HWC order
    for h := 0; h < height; h++ {
        for w := 0; w < width; w++ {
            for c := 0; c < channels; c++ {
                value := float32(h*100 + w*10 + c)
                binary.Write(file, binary.LittleEndian, value)
            }
        }
    }
}

func createTestLabelFile(t *testing.T, filename string, classIndex, numClasses int) {
    file, err := os.Create(filename)
    if err != nil {
        t.Fatalf("Failed to create test file: %v", err)
    }
    defer file.Close()
    
    // Write one-hot encoded label
    for i := 0; i < numClasses; i++ {
        if i == classIndex {
            file.WriteString("1")
        } else {
            file.WriteString("0")
        }
        if i < numClasses-1 {
            file.WriteString(" ")
        }
    }
    file.WriteString("\n")
}

func TestWeightLoader(t *testing.T) {
    // Create temporary directory
    tempDir := t.TempDir()
    
    // Create test weight file
    weightFile := filepath.Join(tempDir, "test_weight.bin")
    createTestWeightFile(t, weightFile, 3, 2, 4)
    
    // Create weight loader
    loader := NewWeightLoader(tempDir)
    
    // Load kernel
    kernel, err := loader.LoadKernel("test_weight.bin", 3, 2, 4)
    if err != nil {
        t.Fatalf("Failed to load kernel: %v", err)
    }
    
    // Verify dimensions
    if kernel.Size != 3 || kernel.Channels != 2 || kernel.Filters != 4 {
        t.Errorf("Wrong kernel dimensions: got (%d,%d,%d), expected (3,2,4)", 
            kernel.Size, kernel.Channels, kernel.Filters)
    }
    
    // Verify a few values
    expected := float32(0*1000 + 0*100 + 0*10 + 0) // h=0, w=0, c=0, f=0
    actual := kernel.GetWeight(0, 0, 0, 0)
    if actual != expected {
        t.Errorf("Wrong weight value: got %f, expected %f", actual, expected)
    }
}

func TestImageLoader(t *testing.T) {
    // Create temporary directory
    tempDir := t.TempDir()
    
    // Create test image file
    imageFile := filepath.Join(tempDir, "test_image.bin")
    createTestImageFile(t, imageFile, 4, 4, 3)
    
    // Create image loader
    loader := NewImageLoader(BinaryFloat32)
    
    // Load image
    image, err := loader.LoadImage(imageFile, 4, 4, 3)
    if err != nil {
        t.Fatalf("Failed to load image: %v", err)
    }
    
    // Verify dimensions
    if image.Height != 4 || image.Width != 4 || image.Channels != 3 {
        t.Errorf("Wrong image dimensions: got (%d,%d,%d), expected (4,4,3)", 
            image.Height, image.Width, image.Channels)
    }
    
    // Verify a value
    expected := float32(0*100 + 0*10 + 0) // h=0, w=0, c=0
    actual := image.Get(0, 0, 0)
    if actual != expected {
        t.Errorf("Wrong pixel value: got %f, expected %f", actual, expected)
    }
}

func TestLabelLoader(t *testing.T) {
    // Create temporary directory
    tempDir := t.TempDir()
    
    // Create test label file
    labelFile := filepath.Join(tempDir, "test_label.txt")
    createTestLabelFile(t, labelFile, 2, 5) // Class 2 out of 5 classes
    
    // Create label loader
    loader := NewLabelLoader(OneHotText)
    
    // Load label
    label, err := loader.LoadLabel(labelFile, 5)
    if err != nil {
        t.Fatalf("Failed to load label: %v", err)
    }
    
    // Verify label
    expected := []int{0, 0, 1, 0, 0}
    if len(label) != len(expected) {
        t.Errorf("Wrong label length: got %d, expected %d", len(label), len(expected))
    }
    
    for i, val := range label {
        if val != expected[i] {
            t.Errorf("Wrong label value at position %d: got %d, expected %d", i, val, expected[i])
        }
    }
}

func TestDataManager(t *testing.T) {
    // Create temporary directories
    tempDir := t.TempDir()
    weightsDir := filepath.Join(tempDir, "weights")
    imagesDir := filepath.Join(tempDir, "images")
    labelsDir := filepath.Join(tempDir, "labels")
    
    os.MkdirAll(weightsDir, 0755)
    os.MkdirAll(imagesDir, 0755)
    os.MkdirAll(labelsDir, 0755)
    
    // Create test files
    createTestWeightFile(t, filepath.Join(weightsDir, "conv1_weight.bin"), 3, 3, 32)
    createTestImageFile(t, filepath.Join(imagesDir, "test_img_0.bin"), 32, 32, 3)
    createTestLabelFile(t, filepath.Join(labelsDir, "label_test_0.txt"), 1, 10)
    
    // Create data manager
    dm := NewDataManager(weightsDir, BinaryFloat32, OneHotText)
    
    // Test loading single image and label
    image, err := dm.imageLoader.LoadImage(filepath.Join(imagesDir, "test_img_0.bin"), 32, 32, 3)
    if err != nil {
        t.Fatalf("Failed to load image: %v", err)
    }
    
    label, err := dm.labelLoader.LoadLabel(filepath.Join(labelsDir, "label_test_0.txt"), 10)
    if err != nil {
        t.Fatalf("Failed to load label: %v", err)
    }
    
    // Verify loaded data
    if image == nil {
        t.Error("Loaded image is nil")
    }
    
    if label == nil {
        t.Error("Loaded label is nil")
    }
    
    // Verify label is valid one-hot
    err = ValidateLabel(label, 10)
    if err != nil {
        t.Errorf("Loaded label failed validation: %v", err)
    }
}

// Benchmark tests
func BenchmarkLoadKernel(b *testing.B) {
    tempDir := b.TempDir()
    weightFile := filepath.Join(tempDir, "bench_weight.bin")
    createTestWeightFile(b, weightFile, 3, 64, 128)
    
    loader := NewWeightLoader(tempDir)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := loader.LoadKernel("bench_weight.bin", 3, 64, 128)
        if err != nil {
            b.Fatalf("Failed to load kernel: %v", err)
        }
    }
}

func BenchmarkLoadImage(b *testing.B) {
    tempDir := b.TempDir()
    imageFile := filepath.Join(tempDir, "bench_image.bin")
    createTestImageFile(b, imageFile, 32, 32, 3)
    
    loader := NewImageLoader(BinaryFloat32)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := loader.LoadImage(imageFile, 32, 32, 3)
        if err != nil {
            b.Fatalf("Failed to load image: %v", err)
        }
    }
}