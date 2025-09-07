package tensor

import (
	"testing"
)

func TestNewKernel(t *testing.T) {
    k := NewKernel(3, 32, 64)
    
    if k.Size != 3 {
        t.Errorf("Expected size 3, got %d", k.Size)
    }
    if k.Channels != 32 {
        t.Errorf("Expected channels 32, got %d", k.Channels)
    }
    if k.Filters != 64 {
        t.Errorf("Expected filters 64, got %d", k.Filters)
    }
    
    expectedSize := 3 * 3 * 32 * 64
    if len(k.Weights) != expectedSize {
        t.Errorf("Expected weights size %d, got %d", expectedSize, len(k.Weights))
    }
}

func TestKernelGetSetWeight(t *testing.T) {
    k := NewKernel(3, 2, 2)
    
    testCases := []struct {
        f, c, h, w int
        value      float32
    }{
        {0, 0, 0, 0, 1.5},
        {0, 1, 2, 1, 2.5},
        {1, 0, 1, 2, 3.5},
        {1, 1, 2, 2, 4.5},
    }
    
    for _, tc := range testCases {
        k.SetWeight(tc.f, tc.c, tc.h, tc.w, tc.value)
        got := k.GetWeight(tc.f, tc.c, tc.h, tc.w)
        if got != tc.value {
            t.Errorf("SetWeight/GetWeight mismatch at (%d,%d,%d,%d): expected %f, got %f", 
                tc.f, tc.c, tc.h, tc.w, tc.value, got)
        }
    }
}

func TestKernelValidation(t *testing.T) {
    // Valid kernel
    k := NewKernel(3, 3, 32)
    err := ValidateKernel(k)
    if err != nil {
        t.Errorf("Valid kernel failed validation: %v", err)
    }
    
    // Nil kernel
    err = ValidateKernel(nil)
    if err == nil {
        t.Error("Nil kernel should fail validation")
    }
    
    // Invalid dimensions
    invalidK := &Kernel{Size: 0, Channels: 3, Filters: 32}
    err = ValidateKernel(invalidK)
    if err == nil {
        t.Error("Invalid dimensions should fail validation")
    }
}

func BenchmarkKernelGetWeight(b *testing.B) {
    k := NewKernel(3, 32, 64)
    k.RandomFill()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = k.GetWeight(i%64, (i/64)%32, (i/2048)%3, (i/6144)%3)
    }
}