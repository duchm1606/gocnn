package tensor

// FeatureMap represents a multi-dimensional tensor
type FeatureMap struct {
    Height   int
    Width    int
    Channels int
    Data     []float32
}

// Kernel represents convolution weights
type Kernel struct {
    Size     int
    Channels int
    Filters  int
    Weights  []float32
}