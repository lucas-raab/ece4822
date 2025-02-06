# Compiler and compilation flags
CC = g++
CFLAGS = -O2

# Output executable names
OUTPUT_FIR = fir.exe
OUTPUT_IIR = iir.exe

# Source files
SRCS_FIR = fir.cpp
SRCS_IIR = iir.cpp

# Default target
all: $(OUTPUT_FIR) $(OUTPUT_IIR)

# Rule to build fir executable
$(OUTPUT_FIR): $(SRCS_FIR)
	$(CC) $(CFLAGS) -o $(OUTPUT_FIR) $(SRCS_FIR)

# Rule to build iir executable
$(OUTPUT_IIR): $(SRCS_IIR)
	$(CC) $(CFLAGS) -o $(OUTPUT_IIR) $(SRCS_IIR)

# Clean target to remove generated files
clean:
	rm -f $(OUTPUT_FIR) $(OUTPUT_IIR)
