package config

// Config holds the model configuration
type Config struct {
	ModelName     string   `yaml:"model_name"`
	InputSize     int      `yaml:"input_size"`
	InputChannels int      `yaml:"input_channels"`
	NumClasses    int      `yaml:"num_classes"`
	ClassNames    []string `yaml:"class_names"`
}
