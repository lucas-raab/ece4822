#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main() {
    srand(time(NULL));
    // Prompt the user for the number of samples
    size_t num_samples;
    printf("Enter the number of samples: ");
    scanf("%zu", &num_samples);
    // Open the output files
    FILE* input_file = fopen("input.bin", "wb");
    FILE* output_file = fopen("output.bin", "wb");
    if (!input_file || !output_file) {
        fprintf(stderr, "Error opening output files.\n");
        return 1;
    }
    float max=-99;
    int maxIndex=0;
    // Generate and write the data
    for (int i = 0; i < num_samples; i++) {
      // Generate random input data (2 floats)
      float input1 = (float)rand() / RAND_MAX * 2.0f - 1.0f;
      if(input1>=max){
        max=input1;
        maxIndex=i-1;
      }
      float output = input1 ;
      // Write the data to the files
      fwrite(&input1, sizeof(float), 1, input_file);
    }
    for(int i=0;i<num_samples;i++){
      float output=i==maxIndex;
      fwrite(&output, sizeof(float), 1, output_file);
    }
    // Close the files
    fclose(input_file);
    fclose(output_file);
    printf("Training data generated and written to 'input.bin' and 'output.bin'.\n");
    return 0;
}
