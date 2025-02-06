# Makefile for building p01.exe from p01.cpp

# Compiler and compilation flags
CC = g++
CFLAGS = -O2

# Output executable name
OUTPUT = p01.exe

# Source files
SRCS = p01.cpp

# Default target
all: $(OUTPUT)

# Rule to build the output executable
$(OUTPUT): $(SRCS)
	$(CC) $(CFLAGS) -o $(OUTPUT) $(SRCS)

# Clean target to remove generated files
clean:
	rm -f $(OUTPUT)
