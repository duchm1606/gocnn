package ops

import (
	"duchm1606/gocnn/internal/tensor"
	"math"
	"testing"
)

func TestConv2DBasic(t *testing.T) {
    // Create a simple 3x3 input with 1 channel
    input := tensor.NewFeatureMap(3, 3, 1)
    input.Fill(1.0) // All ones
    
    // Create a 3x3 kernel with 1 filter
    kernel := tensor.NewKernel(3, 1, 1)
    kernel.Weights[0] = 1.0 // Set all weights to 1
    for i := range kernel.Weights {
        kernel.Weights[i] = 1.0
    }
    
    bias := []float32{0.0}
    config := Conv2DConfig{Padding: 0, Stride: 1}
    
    // Perform convolution
    output := Conv2D(input, kernel, bias, config)
    
    // With all 1s input and all 1s kernel, output should be 9.0
    expected := float32(9.0)
    result := output.Get(0, 0, 0)
    
    if math.Abs(float64(result-expected)) > 1e-6 {
        t.Errorf("Expected %f, got %f", expected, result)
    }
    
    // Check output dimensions
    if output.Height != 1 || output.Width != 1 || output.Channels != 1 {
        t.Errorf("Expected output dimensions (1,1,1), got (%d,%d,%d)", 
            output.Height, output.Width, output.Channels)
    }
}

func TestConv2DWithPadding(t *testing.T) {
    // Create 2x2 input
    input := tensor.NewFeatureMap(2, 2, 1)
    input.Set(0, 0, 0, 1.0)
    input.Set(0, 0, 1, 2.0)
    input.Set(0, 1, 0, 3.0)
    input.Set(0, 1, 1, 4.0)
    
    // Create 3x3 identity kernel
    kernel := tensor.NewKernel(3, 1, 1)
    kernel.SetWeight(0, 0, 1, 1, 1.0) // Center weight = 1, others = 0
    
    bias := []float32{0.0}
    config := Conv2DConfig{Padding: 1, Stride: 1}
    
    output := Conv2D(input, kernel, bias, config)
    
    // With padding=1, output should be same size as input
    if output.Height != 2 || output.Width != 2 {
        t.Errorf("Expected output size (2,2), got (%d,%d)", output.Height, output.Width)
    }
    
    // Check that values are preserved (identity convolution)
    tolerance := float32(1e-6)
    testCases := []struct {
        h, w     int
        expected float32
    }{
        {0, 0, 1.0},
        {0, 1, 2.0},
        {1, 0, 3.0},
        {1, 1, 4.0},
    }
    
    for _, tc := range testCases {
        result := output.Get(0, tc.h, tc.w)
        if math.Abs(float64(result-tc.expected)) > float64(tolerance) {
            t.Errorf("At (%d,%d): expected %f, got %f", tc.h, tc.w, tc.expected, result)
        }
    }
}

func TestConv2DWithStride(t *testing.T) {
    // Create 4x4 input
    input := tensor.NewFeatureMap(4, 4, 1)
    for i := 0; i < 4; i++ {
        for j := 0; j < 4; j++ {
            input.Set(0, i, j, float32(i*4+j+1))
        }
    }
    
    // Create 2x2 average pooling kernel
    kernel := tensor.NewKernel(2, 1, 1)
    for i := range kernel.Weights {
        kernel.Weights[i] = 0.25 // Average of 2x2 region
    }
    
    bias := []float32{0.0}
    config := Conv2DConfig{Padding: 0, Stride: 2}
    
    output := Conv2D(input, kernel, bias, config)
    
    // With stride=2, output should be 2x2
    if output.Height != 2 || output.Width != 2 {
        t.Errorf("Expected output size (2,2), got (%d,%d)", output.Height, output.Width)
    }
}

func TestConv2DMultiChannel(t *testing.T) {
    // Create 3x3x2 input (2 channels)
    input := tensor.NewFeatureMap(3, 3, 2)
    
    // Fill each channel with different values
    for h := 0; h < 3; h++ {
        for w := 0; w < 3; w++ {
            input.Set(0, h, w, 1.0) // Channel 0: all 1s
            input.Set(1, h, w, 2.0) // Channel 1: all 2s
        }
    }
    
    // Create 3x3x2x1 kernel (2 input channels, 1 output filter)
    kernel := tensor.NewKernel(3, 2, 1)
    
    // Set weights: channel 0 weights = 1, channel 1 weights = 0.5
    for h := 0; h < 3; h++ {
        for w := 0; w < 3; w++ {
            kernel.SetWeight(0, 0, h, w, 1.0)   // Filter 0, Channel 0
            kernel.SetWeight(0, 1, h, w, 0.5)   // Filter 0, Channel 1
        }
    }
    
    bias := []float32{0.0}
    config := Conv2DConfig{Padding: 0, Stride: 1}
    
    output := Conv2D(input, kernel, bias, config)
    
    // Expected result: 1*9*1 + 2*9*0.5 = 9 + 9 = 18
    expected := float32(18.0)
    result := output.Get(0, 0, 0)
    
    if math.Abs(float64(result-expected)) > 1e-6 {
        t.Errorf("Multi-channel convolution: expected %f, got %f", expected, result)
    }
}

func TestConv2DParallelVsSerial(t *testing.T) {
    // Create larger input for meaningful parallel test
    input := tensor.NewFeatureMap(32, 32, 3)
    input.RandomFill()
    
    // Create kernel with multiple filters
    kernel := tensor.NewKernel(3, 3, 8)
    kernel.RandomFill()
    
    bias := make([]float32, 8)
    config := Conv2DConfig{Padding: 1, Stride: 1}
    
    // Compute both serial and parallel
    outputSerial := Conv2D(input, kernel, bias, config)
    outputParallel := Conv2DParallel(input, kernel, bias, config)
    
    // Results should be identical
    if outputSerial.Height != outputParallel.Height ||
       outputSerial.Width != outputParallel.Width ||
       outputSerial.Channels != outputParallel.Channels {
        t.Error("Parallel and serial outputs have different dimensions")
    }
    
    // Check all values are equal (within floating point precision)
    tolerance := float32(1e-6)
    for c := 0; c < outputSerial.Channels; c++ {
        for h := 0; h < outputSerial.Height; h++ {
            for w := 0; w < outputSerial.Width; w++ {
                serial := outputSerial.Get(c, h, w)
                parallel := outputParallel.Get(c, h, w)
                
                if math.Abs(float64(serial-parallel)) > float64(tolerance) {
                    t.Errorf("Parallel/serial mismatch at (%d,%d,%d): %f vs %f", 
                        c, h, w, serial, parallel)
                }
            }
        }
    }
}

func TestGetConvOutputDims(t *testing.T) {
    testCases := []struct {
        inputH, inputW, kernelSize, padding, stride int
        expectedH, expectedW                        int
    }{
        {32, 32, 3, 1, 1, 32, 32}, // Same padding
        {32, 32, 3, 0, 1, 30, 30}, // Valid padding
        {32, 32, 3, 0, 2, 15, 15}, // Stride 2
        {28, 28, 5, 2, 1, 28, 28}, // 5x5 kernel with padding 2
    }
    
    for _, tc := range testCases {
        outH, outW := GetConvOutputDims(tc.inputH, tc.inputW, tc.kernelSize, tc.padding, tc.stride)
        if outH != tc.expectedH || outW != tc.expectedW {
            t.Errorf("Input (%d,%d), kernel %d, padding %d, stride %d: expected (%d,%d), got (%d,%d)",
                tc.inputH, tc.inputW, tc.kernelSize, tc.padding, tc.stride,
                tc.expectedH, tc.expectedW, outH, outW)
        }
    }
}

// Benchmark tests
func BenchmarkConv2DSmall(b *testing.B) {
    input := tensor.NewFeatureMap(32, 32, 3)
    input.RandomFill()
    
    kernel := tensor.NewKernel(3, 3, 32)
    kernel.RandomFill()
    
    bias := make([]float32, 32)
    config := Conv2DConfig{Padding: 1, Stride: 1}
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        output := Conv2D(input, kernel, bias, config)
        _ = output
    }
}

func BenchmarkConv2DParallel(b *testing.B) {
    input := tensor.NewFeatureMap(32, 32, 3)
    input.RandomFill()
    
    kernel := tensor.NewKernel(3, 3, 32)
    kernel.RandomFill()
    
    bias := make([]float32, 32)
    config := Conv2DConfig{Padding: 1, Stride: 1}
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        output := Conv2DParallel(input, kernel, bias, config)
        _ = output
    }
}

func BenchmarkConv2DLarge(b *testing.B) {
    input := tensor.NewFeatureMap(64, 64, 64)
    input.RandomFill()
    
    kernel := tensor.NewKernel(3, 64, 128)
    kernel.RandomFill()
    
    bias := make([]float32, 128)
    config := Conv2DConfig{Padding: 1, Stride: 1}
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        output := Conv2DParallel(input, kernel, bias, config)
        _ = output
    }
}