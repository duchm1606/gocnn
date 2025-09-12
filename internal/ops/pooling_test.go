package ops

import (
	"duchm1606/gocnn/internal/tensor"
	"math"
	"testing"
)

func TestMaxPooling2D(t *testing.T) {
    // Create 4x4 input with known values
    input := tensor.NewFeatureMap(4, 4, 1)
    
    // Fill with values 1-16
    for i := 0; i < 4; i++ {
        for j := 0; j < 4; j++ {
            input.Set(0, i, j, float32(i*4+j+1))
        }
    }
    
    // Perform 2x2 max pooling with stride 2
    output := MaxPooling2D(input, 2, 2)
    
    // Check output dimensions
    if output.Height != 2 || output.Width != 2 || output.Channels != 1 {
        t.Errorf("Expected output dimensions (2,2,1), got (%d,%d,%d)", 
            output.Height, output.Width, output.Channels)
    }
    
    // Check output values
    expected := [][]float32{
        {6, 8},   // max of (1,2,5,6) and (3,4,7,8)
        {14, 16}, // max of (9,10,13,14) and (11,12,15,16)
    }
    
    for i := 0; i < 2; i++ {
        for j := 0; j < 2; j++ {
            result := output.Get(0, i, j)
            if result != expected[i][j] {
                t.Errorf("At (%d,%d): expected %f, got %f", i, j, expected[i][j], result)
            }
        }
    }
}

func TestAvgPooling2D(t *testing.T) {
    // Create 2x2 input
    input := tensor.NewFeatureMap(2, 2, 1)
    input.Set(0, 0, 0, 1.0)
    input.Set(0, 0, 1, 2.0)
    input.Set(0, 1, 0, 3.0)
    input.Set(0, 1, 1, 4.0)
    
    // Perform 2x2 average pooling
    output := AvgPooling2D(input, 2, 2)
    
    // Should produce 1x1 output with average value
    if output.Height != 1 || output.Width != 1 {
        t.Errorf("Expected output size (1,1), got (%d,%d)", output.Height, output.Width)
    }
    
    expected := float32(2.5) // (1+2+3+4)/4 = 2.5
    result := output.Get(0, 0, 0)
    
    if math.Abs(float64(result-expected)) > 1e-6 {
        t.Errorf("Expected %f, got %f", expected, result)
    }
}

func TestMaxPoolingMultiChannel(t *testing.T) {
    // Create 4x4x2 input (2 channels)
    input := tensor.NewFeatureMap(4, 4, 2)
    
    // Fill channels with different patterns
    for c := 0; c < 2; c++ {
        for i := 0; i < 4; i++ {
            for j := 0; j < 4; j++ {
                value := float32((c+1) * (i*4 + j + 1))
                input.Set(c, i, j, value)
            }
        }
    }
    
    output := MaxPooling2D(input, 2, 2)
    
    // Check that channels are preserved
    if output.Channels != 2 {
        t.Errorf("Expected 2 channels, got %d", output.Channels)
    }
    
    // Check that each channel is pooled independently
    for c := 0; c < 2; c++ {
        val := output.Get(c, 0, 0)
        expected := float32((c + 1) * 6) // max of first 2x2 region for each channel
        
        if val != expected {
            t.Errorf("Channel %d: expected %f, got %f", c, expected, val)
        }
    }
}

func TestGlobalMaxPooling(t *testing.T) {
    // Create 3x3x2 input
    input := tensor.NewFeatureMap(3, 3, 2)
    
    // Set known max values for each channel
    input.Set(0, 1, 1, 10.0) // Channel 0 max
    input.Set(1, 2, 0, 20.0) // Channel 1 max
    
    // Fill other positions with smaller values
    for c := 0; c < 2; c++ {
        for i := 0; i < 3; i++ {
            for j := 0; j < 3; j++ {
                if !(c == 0 && i == 1 && j == 1) && !(c == 1 && i == 2 && j == 0) {
                    input.Set(c, i, j, float32(c+1))
                }
            }
        }
    }
    
    result := GlobalMaxPooling(input)
    
    // Check output length
    if len(result) != 2 {
        t.Errorf("Expected 2 values, got %d", len(result))
    }
    
    // Check max values
    if result[0] != 10.0 {
        t.Errorf("Channel 0: expected 10.0, got %f", result[0])
    }
    if result[1] != 20.0 {
        t.Errorf("Channel 1: expected 20.0, got %f", result[1])
    }
}

func TestGlobalAvgPooling(t *testing.T) {
    // Create 2x2x1 input with known values
    input := tensor.NewFeatureMap(2, 2, 1)
    input.Set(0, 0, 0, 1.0)
    input.Set(0, 0, 1, 2.0)
    input.Set(0, 1, 0, 3.0)
    input.Set(0, 1, 1, 4.0)
    
    result := GlobalAvgPooling(input)
    
    if len(result) != 1 {
        t.Errorf("Expected 1 value, got %d", len(result))
    }
    
    expected := float32(2.5) // (1+2+3+4)/4
    if math.Abs(float64(result[0]-expected)) > 1e-6 {
        t.Errorf("Expected %f, got %f", expected, result[0])
    }
}

func TestPoolingWithStride1(t *testing.T) {
    // Create 3x3 input
    input := tensor.NewFeatureMap(3, 3, 1)
    for i := 0; i < 3; i++ {
        for j := 0; j < 3; j++ {
            input.Set(0, i, j, float32(i*3+j+1))
        }
    }
    
    // Perform 2x2 pooling with stride 1 (overlapping windows)
    output := MaxPooling2D(input, 2, 1)
    
    // Should produce 2x2 output
    if output.Height != 2 || output.Width != 2 {
        t.Errorf("Expected output size (2,2), got (%d,%d)", output.Height, output.Width)
    }
    
    // Check specific values
    expected := [][]float32{
        {5, 6}, // max of (1,2,4,5) and (2,3,5,6)
        {8, 9}, // max of (4,5,7,8) and (5,6,8,9)
    }
    
    for i := 0; i < 2; i++ {
        for j := 0; j < 2; j++ {
            result := output.Get(0, i, j)
            if result != expected[i][j] {
                t.Errorf("At (%d,%d): expected %f, got %f", i, j, expected[i][j], result)
            }
        }
    }
}

func TestAdaptiveMaxPooling(t *testing.T) {
    // Create 6x6 input
    input := tensor.NewFeatureMap(6, 6, 1)
    input.RandomFill()
    
    // Adaptive pooling to 2x2 output
    output := AdaptiveMaxPooling(input, 2, 2)
    
    // Check output dimensions
    if output.Height != 2 || output.Width != 2 {
        t.Errorf("Expected output size (2,2), got (%d,%d)", output.Height, output.Width)
    }
    
    // Check that channels are preserved
    if output.Channels != input.Channels {
        t.Errorf("Expected %d channels, got %d", input.Channels, output.Channels)
    }
}

func TestGetPoolingOutputDims(t *testing.T) {
    testCases := []struct {
        inputH, inputW, kernelSize, stride int
        expectedH, expectedW               int
    }{
        {32, 32, 2, 2, 16, 16}, // Standard 2x2 pooling
        {28, 28, 2, 2, 14, 14}, // Even dimensions
        {29, 29, 2, 2, 14, 14}, // Odd dimensions
        {16, 16, 4, 4, 4, 4},   // 4x4 pooling
    }
    
    for _, tc := range testCases {
        outH, outW := GetPoolingOutputDims(tc.inputH, tc.inputW, tc.kernelSize, tc.stride)
        if outH != tc.expectedH || outW != tc.expectedW {
            t.Errorf("Input (%d,%d), kernel %d, stride %d: expected (%d,%d), got (%d,%d)",
                tc.inputH, tc.inputW, tc.kernelSize, tc.stride,
                tc.expectedH, tc.expectedW, outH, outW)
        }
    }
}

// Benchmark tests
func BenchmarkMaxPooling2D(b *testing.B) {
    input := tensor.NewFeatureMap(32, 32, 64)
    input.RandomFill()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        output := MaxPooling2D(input, 2, 2)
        _ = output
    }
}

func BenchmarkGlobalMaxPooling(b *testing.B) {
    input := tensor.NewFeatureMap(32, 32, 128)
    input.RandomFill()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        result := GlobalMaxPooling(input)
        _ = result
    }
}

func BenchmarkAdaptiveMaxPooling(b *testing.B) {
    input := tensor.NewFeatureMap(32, 32, 64)
    input.RandomFill()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        output := AdaptiveMaxPooling(input, 7, 7)
        _ = output
    }
}