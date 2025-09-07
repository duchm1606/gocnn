package tensor

import (
	"testing"
)

func TestNewFeatureMap(t *testing.T) {
    fm := NewFeatureMap(32, 32, 3)
    
    if fm.Height != 32 {
        t.Errorf("Expected height 32, got %d", fm.Height)
    }
    if fm.Width != 32 {
        t.Errorf("Expected width 32, got %d", fm.Width)
    }
    if fm.Channels != 3 {
        t.Errorf("Expected channels 3, got %d", fm.Channels)
    }
    
    expectedSize := 32 * 32 * 3
    if len(fm.Data) != expectedSize {
        t.Errorf("Expected data size %d, got %d", expectedSize, len(fm.Data))
    }
    
    // Check that data is zero-initialized
    for i, val := range fm.Data {
        if val != 0 {
            t.Errorf("Expected zero initialization, got %f at index %d", val, i)
        }
    }
}

func TestFeatureMapGetSet(t *testing.T) {
    fm := NewFeatureMap(3, 3, 2)
    
    // Test setting and getting values
    testCases := []struct {
        c, h, w int
        value   float32
    }{
        {0, 0, 0, 1.5},
        {0, 1, 2, 2.5},
        {1, 2, 1, 3.5},
    }
    
    for _, tc := range testCases {
        fm.Set(tc.c, tc.h, tc.w, tc.value)
        got := fm.Get(tc.c, tc.h, tc.w)
        if got != tc.value {
            t.Errorf("Set/Get mismatch at (%d,%d,%d): expected %f, got %f", 
                tc.c, tc.h, tc.w, tc.value, got)
        }
    }
}

func TestFeatureMapBoundsChecking(t *testing.T) {
    fm := NewFeatureMap(2, 2, 1)
    
    // Test valid bounds
    fm.Set(0, 1, 1, 1.0)
    val := fm.Get(0, 1, 1)
    if val != 1.0 {
        t.Errorf("Expected 1.0, got %f", val)
    }
    
    // Test invalid bounds (should panic)
    invalidCases := []struct {
        c, h, w int
        desc    string
    }{
        {-1, 0, 0, "negative channel"},
        {1, 0, 0, "channel out of bounds"},
        {0, -1, 0, "negative height"},
        {0, 2, 0, "height out of bounds"},
        {0, 0, -1, "negative width"},
        {0, 0, 2, "width out of bounds"},
    }
    
    for _, ic := range invalidCases {
        t.Run(ic.desc, func(t *testing.T) {
            defer func() {
                if r := recover(); r == nil {
                    t.Errorf("Expected panic for %s", ic.desc)
                }
            }()
            fm.Get(ic.c, ic.h, ic.w)
        })
    }
}

func TestFeatureMapClone(t *testing.T) {
    original := NewFeatureMap(2, 2, 1)
    original.Set(0, 0, 0, 5.0)
    original.Set(0, 1, 1, 10.0)
    
    clone := original.Clone()
    
    // Check that dimensions match
    if clone.Height != original.Height || clone.Width != original.Width || clone.Channels != original.Channels {
        t.Error("Clone dimensions don't match original")
    }
    
    // Check that data matches
    if clone.Get(0, 0, 0) != 5.0 || clone.Get(0, 1, 1) != 10.0 {
        t.Error("Clone data doesn't match original")
    }
    
    // Check that it's a deep copy
    clone.Set(0, 0, 0, 15.0)
    if original.Get(0, 0, 0) != 5.0 {
        t.Error("Clone is not independent of original")
    }
}

func TestPadFeatureMap(t *testing.T) {
    original := NewFeatureMap(2, 2, 1)
    original.Fill(1.0)
    
    padded := PadFeatureMap(original, 1)
    
    // Check new dimensions
    if padded.Height != 4 || padded.Width != 4 || padded.Channels != 1 {
        t.Errorf("Expected padded dimensions (4,4,1), got (%d,%d,%d)", 
            padded.Height, padded.Width, padded.Channels)
    }
    
    // Check that padding is zero
    if padded.Get(0, 0, 0) != 0 || padded.Get(0, 3, 3) != 0 {
        t.Error("Padding should be zero")
    }
    
    // Check that center contains original data
    if padded.Get(0, 1, 1) != 1.0 || padded.Get(0, 2, 2) != 1.0 {
        t.Error("Center should contain original data")
    }
}

// Benchmark tests
func BenchmarkFeatureMapGet(b *testing.B) {
    fm := NewFeatureMap(32, 32, 3)
    fm.RandomFill()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = fm.Get(i%3, (i/3)%32, (i/96)%32)
    }
}

func BenchmarkFeatureMapGetUnsafe(b *testing.B) {
    fm := NewFeatureMap(32, 32, 3)
    fm.RandomFill()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = fm.GetUnsafe(i%3, (i/3)%32, (i/96)%32)
    }
}