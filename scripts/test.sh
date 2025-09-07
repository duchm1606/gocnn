#!/bin/bash
set -e

echo "Running comprehensive tests..."

# Unit tests
echo "Running unit tests..."
go test -v -race -coverprofile=coverage.out ./...

# Generate coverage report
echo "Generating coverage report..."
go tool cover -html=coverage.out -o coverage.html

# Display coverage summary
echo "Coverage summary:"
go tool cover -func=coverage.out | tail -1

echo "Tests complete! Coverage report: coverage.html"