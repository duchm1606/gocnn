package main

import (
	"flag"
	"fmt"
	"log"
	"os"
)

func main() {
	var (
		weightsPath = flag.String("weights", "", "Path to weights directory")
		imagePath   = flag.String("image", "", "Path to input image")
		configPath  = flag.String("config", "configs/cifar10.yaml", "Path to config file")
		verbose     = flag.Bool("verbose", false, "Verbose output")
	)
	flag.Parse()

	if *weightsPath == "" || *imagePath == "" {
		fmt.Println("Usage: gocnn-inference -weights <path> -image <path>")
		flag.PrintDefaults()
		os.Exit(1)
	}

	log.Printf("Loading model from: %s", *weightsPath)
	log.Printf("Processing image: %s", *imagePath)
	log.Printf("Config: %s", *configPath)
	log.Printf("Verbose: %v", *verbose)

	fmt.Println("Inference functionality coming in Task 11!")
}