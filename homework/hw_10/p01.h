#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <math.h>

bool readBinaryFiles(const std::string& inputFileName, const std::string& outputFileName,
                     float*& inputArray, float*& outputArray, size_t &num ){
    // Open both input files for reading in binary mode
    std::ifstream inFile(inputFileName, std::ios::binary);
    std::ifstream inFile1(outputFileName, std::ios::binary);

    if (!inFile || !inFile1) {
        std::cerr << "Error opening input files!" << std::endl;
        return false;
    }

    // Get file sizes
    inFile.seekg(0, std::ios::end);
    std::streamsize inputSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    inFile1.seekg(0, std::ios::end);
    std::streamsize outputSize = inFile1.tellg();
    inFile1.seekg(0, std::ios::beg);

    // Calculate number of floats in each file
    size_t numFloats = inputSize / sizeof(float);
    if (outputSize / sizeof(float) != numFloats) {
        std::cerr << "Input and output files do not contain the same number of floats!" << std::endl;
        return false;
    }

    // Allocate memory for the float arrays
    inputArray = new float[numFloats];
    outputArray = new float[numFloats];

    // Read the data from input.bin
    inFile.read(reinterpret_cast<char*>(inputArray), inputSize);

    // Read the data from output.bin
    inFile1.read(reinterpret_cast<char*>(outputArray), outputSize);

    // Close the files
    inFile.close();
    inFile1.close();


    num=numFloats;
    
    return true;
}
/**/
