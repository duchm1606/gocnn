#!/bin/bash
set -e

echo "Building GoCNN..."

# Format code
echo "Formatting code..."
go fmt ./...

# Run tests
echo "Running tests..."
go test ./...

# Build binaries
echo "Building binaries..."
make build

echo "Build complete! Binaries are in bin/ directory"