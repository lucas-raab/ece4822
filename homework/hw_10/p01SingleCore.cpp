#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "p01.h"





// Structure to hold the neural network
typedef struct {
  int input_size;
  int hidden_size;
  int output_size;
  
  // Weights and biases
  float** weights_input_hidden;
  float** weights_hidden_output;
  float* bias_hidden;
  float* bias_output;
  
  // Activations and outputs
  float* hidden_activation;
  float* hidden_output;
  float* output_activation;
  float* predicted_output;
} NeuralNetwork;

// Helper functions for matrix operations
float** allocate_2d_matrix(int rows, int cols) {
    float** matrix = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*)malloc(cols * sizeof(float));
    }
    return matrix;
}

void free_2d_matrix(float** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Initialize random weights between -1 and 1
float random_weight() {
    return (float)rand() / RAND_MAX * 2 - 1;
}

// Sigmoid activation function
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

// Initialize the neural network
NeuralNetwork* create_neural_network(int input_size, int hidden_size, int output_size) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;
    
    // Initialize weights
    nn->weights_input_hidden = allocate_2d_matrix(input_size, hidden_size);
    nn->weights_hidden_output = allocate_2d_matrix(hidden_size, output_size);
    
    // Initialize biases
    nn->bias_hidden = (float*)calloc(hidden_size, sizeof(float));
    nn->bias_output = (float*)calloc(output_size, sizeof(float));
    
    // Initialize temporary storage
    nn->hidden_activation = (float*)malloc(hidden_size * sizeof(float));
    nn->hidden_output = (float*)malloc(hidden_size * sizeof(float));
    nn->output_activation = (float*)malloc(output_size * sizeof(float));
    nn->predicted_output = (float*)malloc(output_size * sizeof(float));
    
    // Initialize random weights
    std::cout << "Input size " << input_size << std::endl;
    std::cout << "Hidden size " << hidden_size << std::endl;
    std::cout << "Output size " << output_size << std::endl;
    
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            nn->weights_input_hidden[i][j] = random_weight();
        }
    }
    
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < output_size; j++) {
            nn->weights_hidden_output[i][j] = random_weight();
        }
    }
    
    return nn;
}

// Free neural network memory
void free_neural_network(NeuralNetwork* nn) {
    free_2d_matrix(nn->weights_input_hidden, nn->input_size);
    free_2d_matrix(nn->weights_hidden_output, nn->hidden_size);
    free(nn->bias_hidden);
    free(nn->bias_output);
    free(nn->hidden_activation);
    free(nn->hidden_output);
    free(nn->output_activation);
    free(nn->predicted_output);
    free(nn);
}

// Feedforward pass
void feedforward(NeuralNetwork* nn, float* input) {
    // Input to hidden layer
    for (int i = 0; i < nn->hidden_size; i++) {
        nn->hidden_activation[i] = nn->bias_hidden[i];
        for (int j = 0; j < nn->input_size; j++) {
            nn->hidden_activation[i] += input[j] * nn->weights_input_hidden[j][i];
        }
        nn->hidden_output[i] = sigmoid(nn->hidden_activation[i]);
    }
    
    // Hidden to output layer
    for (int i = 0; i < nn->output_size; i++) {
        nn->output_activation[i] = nn->bias_output[i];
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->output_activation[i] += nn->hidden_output[j] * nn->weights_hidden_output[j][i];
        }
        nn->predicted_output[i] = sigmoid(nn->output_activation[i]);
    }
}

// Backward pass
void backward(NeuralNetwork* nn, float* input, float* target, float learning_rate) {
    // Output layer error
  float* output_delta = (float*)malloc(nn->output_size * sizeof(float));
    for (int i = 0; i < nn->output_size; i++) {
        output_delta[i] = (target[i] - nn->predicted_output[i]) * 
                         sigmoid_derivative(nn->predicted_output[i]);
    }
    
    // Hidden layer error
    float* hidden_delta = (float*)malloc(nn->hidden_size * sizeof(float));
    for (int i = 0; i < nn->hidden_size; i++) {
        hidden_delta[i] = 0;
        for (int j = 0; j < nn->output_size; j++) {
            hidden_delta[i] += output_delta[j] * nn->weights_hidden_output[i][j];
        }
        hidden_delta[i] *= sigmoid_derivative(nn->hidden_output[i]);
    }
    
    // Update weights and biases
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->output_size; j++) {
            nn->weights_hidden_output[i][j] += learning_rate * nn->hidden_output[i] * output_delta[j];
        }
    }
    
    for (int i = 0; i < nn->input_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->weights_input_hidden[i][j] += learning_rate * input[i] * hidden_delta[j];
        }
    }
    
    for (int i = 0; i < nn->output_size; i++) {
        nn->bias_output[i] += learning_rate * output_delta[i];
    }
    
    for (int i = 0; i < nn->hidden_size; i++) {
        nn->bias_hidden[i] += learning_rate * hidden_delta[i];
    }
    
    free(output_delta);
    free(hidden_delta);
}

// Training function
void train(NeuralNetwork* nn, float* X, float* y, int num_samples, 
	   int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
      float total_loss = 0;
        

            feedforward(nn, X);
            backward(nn, X, y, learning_rate);
            
            // Calculate loss
            for (int j = 0; j < nn->output_size; j++) {
	      float error = y[j] - nn->predicted_output[j];
                total_loss += error * error;
            }

        
        if (epoch % 1000 == 0) {
            printf("Epoch %d, Loss: %f\n", epoch, total_loss / num_samples);
        }
    }
}

int main() {
    srand(time(NULL));
    
  // Load data from files
    float* inputArray = nullptr;
    float* outputArray = nullptr;
    size_t size = 0;
    if (!readBinaryFiles("input.bin", "output.bin", inputArray, outputArray, size)) {
        fprintf(stderr, "Can't read from files\n");
        return 1;
    }

    std::cout << "size: " << size << std::endl;
    
    // Create pointers for X and y
    //    float** X_ptr = allocate_2d_matrix(size, 2);
    //    float** y_ptr = allocate_2d_matrix(size, 1);
    //    for (size_t i = 0; i < size; i++) {
    //        X_ptr[i][0] = inputArray[i * 2];
    //        X_ptr[i][1] = inputArray[i * 2 + 1];
    //        y_ptr[i][0] = outputArray[i];
    //    }

    // Create and train neural network
    NeuralNetwork* nn = create_neural_network(size, size, size); 
    train(nn, inputArray,outputArray, size, 10000, 0.1);
    
    // Test the network
    printf("\nPredictions after training:\n");

    
    for (int i = 0; i < size; i++) {
      printf("%d, %f \n",i,nn->predicted_output[i]);
    }
    
    // Clean up

    free_neural_network(nn);
    
    return 0;
}
