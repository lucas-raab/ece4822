#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <output file path> <input file path>\n", argv[0]);
        return 1;
    }

    const char* output_path = argv[1];


    FILE* output_file = fopen(output_path, "rb");
    if (!output_file) {
        fprintf(stderr, "Error opening output file: %s\n", output_path);
        return 1;
    }

    float output;
    size_t index = 0;

    printf("Contents of '%s':\n", output_path);
    while (fread(&output, sizeof(float), 1, output_file) == 1) {
        printf("Output[%zu] = %f\n", index++, output);
    }

    fclose(output_file);


    return 0;
}
