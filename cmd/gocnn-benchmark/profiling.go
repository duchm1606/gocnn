package main

import (
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
)

var cpuProfile *os.File

// startCPUProfile starts CPU profiling
func startCPUProfile(filename string) error {
    var err error
    cpuProfile, err = os.Create(filename)
    if err != nil {
        return fmt.Errorf("failed to create CPU profile file: %w", err)
    }
    
    err = pprof.StartCPUProfile(cpuProfile)
    if err != nil {
        cpuProfile.Close()
        return fmt.Errorf("failed to start CPU profiling: %w", err)
    }
    
    return nil
}

// stopCPUProfile stops CPU profiling
func stopCPUProfile() {
    if cpuProfile != nil {
        pprof.StopCPUProfile()
        cpuProfile.Close()
        cpuProfile = nil
    }
}

// writeMemProfile writes memory profile to file
func writeMemProfile(filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return fmt.Errorf("failed to create memory profile file: %w", err)
    }
    defer file.Close()
    
    runtime.GC() // Force garbage collection before profiling
    
    err = pprof.WriteHeapProfile(file)
    if err != nil {
        return fmt.Errorf("failed to write memory profile: %w", err)
    }
    
    return nil
}