PROJECT_NAME = gocnn
BINARY_DIR = bin
GO_FILES = $(shell find . -name '*.go' -not -path './vendor/*')

# Binary names
INFERENCE_BINARY = $(BINARY_DIR)/$(PROJECT_NAME)-inference
BENCHMARK_BINARY = $(BINARY_DIR)/$(PROJECT_NAME)-benchmark

# Build flags
BUILD_FLAGS = -ldflags="-w -s"
TEST_FLAGS = -v -race -coverprofile=coverage.out

.PHONY: all build test clean install help

# Default target
all: build

# Build all binaries
build: $(INFERENCE_BINARY) $(BENCHMARK_BINARY)

$(INFERENCE_BINARY): $(GO_FILES)
	@mkdir -p $(BINARY_DIR)
	go build $(BUILD_FLAGS) -o $@ ./cmd/gocnn-inference

$(BENCHMARK_BINARY): $(GO_FILES)
	@mkdir -p $(BINARY_DIR)
	go build $(BUILD_FLAGS) -o $@ ./cmd/gocnn-benchmark

# Run tests
test:
	go test $(TEST_FLAGS) ./...

# Run tests with coverage report
test-coverage: test
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Clean build artifacts
clean:
	rm -rf $(BINARY_DIR)
	rm -f coverage.out coverage.html

# Install binaries to GOPATH/bin
install: build
	go install ./cmd/gocnn-inference
	go install ./cmd/gocnn-benchmark

# Format code
fmt:
	go fmt ./...

# Lint code (requires golangci-lint)
lint:
	golangci-lint run

# Download dependencies
deps:
	go mod download
	go mod tidy

# Generate documentation
docs:
	godoc -http=:6060

# Show help
help:
	@echo "Available targets:"
	@echo "  build          Build all binaries"
	@echo "  test           Run unit tests"
	@echo "  test-coverage  Run tests with coverage report"
	@echo "  clean          Clean build artifacts"
	@echo "  install        Install binaries"
	@echo "  fmt            Format Go code"
	@echo "  lint           Lint code (requires golangci-lint)"
	@echo "  deps           Download and tidy dependencies"
	@echo "  docs           Start local documentation server"
	@echo "  help           Show this help message"