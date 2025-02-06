#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "p01.h"
#include <time.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>

std::vector<std::string> loadWordList() {
    return {
        "apple", "ball", "cat", "dog", "elephant", "fish", "giraffe", "hat", "igloo", "juice",
        "kite", "lemon", "monkey", "nut", "orange", "pear", "queen", "rabbit", "snake", "tiger",
        "to", "a", "" // Add more words as needed
    };
}

int main(int argc, char** argv) {
    const int targetCharacterCount = 10000;
    
    // Seed the random number generator
    std::srand(std::time(nullptr));

    float** l1tol2w;
    float** l2tol3w;

    // Allocate memory for the arrays
    l1tol2w = new float*[size];
    l2tol3w = new float*[size];

    for (int i = 0; i < size; i++) {
        l1tol2w[i] = new float[size];
        l2tol3w[i] = new float[size];
    }

    // Initialize l1tol2w with random values that sum to 1
    for (int i = 0; i < size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < size; j++) {
            l1tol2w[i][j] = static_cast<float>(rand()) / RAND_MAX;
            sum += l1tol2w[i][j];
        }
        for (int j = 0; j < size; j++) {
            l1tol2w[i][j] /= sum;
        }
    }

    // Initialize l2tol3w
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            l2tol3w[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Load the word list
    std::vector<std::string> wordList = loadWordList();
    size_t wordListSize = wordList.size();

    // Open files for writing
    std::ofstream outFile("input.bin", std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return 1;
    }

    std::ofstream outFile1("output.bin", std::ios::binary);
    if (!outFile1) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return 1;
    }

    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, wordListSize - 1);

    int charactersWritten = 0;
    int shift = 3; // Caesar cipher shift

    // Write words and spaces until reaching target character count
    while (charactersWritten < targetCharacterCount) {
        std::string randomWord = wordList[dis(gen)];
        int spaceLeft = targetCharacterCount - charactersWritten;

        if (spaceLeft <= randomWord.length()) {
            randomWord = randomWord.substr(0, spaceLeft);
            for (char c : randomWord) {
                // Apply Caesar shift and convert to float for shifted character
                char shifted = c + shift;
                float f = static_cast<float>(shifted);
                uint32_t* floatAsInt = reinterpret_cast<uint32_t*>(&f);
                outFile1.write(reinterpret_cast<char*>(floatAsInt), sizeof(f));

                // Convert original character to float and write to file
                f = static_cast<float>(c);
                floatAsInt = reinterpret_cast<uint32_t*>(&f);
                outFile.write(reinterpret_cast<char*>(floatAsInt), sizeof(f));
            }
            charactersWritten += randomWord.length();
            break;
        } else {
            for (char c : randomWord) {
                // Apply Caesar shift and convert to float for shifted character
                char shifted = c + shift;
                float f = static_cast<float>(shifted);
                uint32_t* floatAsInt = reinterpret_cast<uint32_t*>(&f);
                outFile1.write(reinterpret_cast<char*>(floatAsInt), sizeof(f));

                // Convert original character to float and write to file
                f = static_cast<float>(c);
                floatAsInt = reinterpret_cast<uint32_t*>(&f);
                outFile.write(reinterpret_cast<char*>(floatAsInt), sizeof(f));
            }
            
            // Append a space at the end of the randomWord
            char space = ' ';
            float spaceFloat = static_cast<float>(space);
            uint32_t* spaceAsInt = reinterpret_cast<uint32_t*>(&spaceFloat);

            // Write the raw bytes of the space to both files
            outFile.write(reinterpret_cast<char*>(spaceAsInt), sizeof(spaceFloat));
            outFile1.write(reinterpret_cast<char*>(spaceAsInt), sizeof(spaceFloat));

            charactersWritten += randomWord.length() + 1;
        }
    }

    outFile.close();
    outFile1.close();

    return 0;
}
