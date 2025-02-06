# file: hw_03/Makefile
#
# The source code should be in a file hw_03.cc.
#

# define compilation flags
#
CFLAGS += -O2 -Wall
#CFLAGS += -g -Wall

# define source and object files
#
SRC = hw_03.cc
OBJ = hw_03.o

# define dependencies
#
DEPS = $(NEDC_NFC)/lib/libdsp.a

# define include files
#
INCLUDES = $(ISIP_INCLUDE)

# define a target for the application
#
all: hw_03.exe

# define a target to link the application
#
hw_03.exe: $(OBJ) $(DEPS)
	g++  $(CFLAGS) -o hw_03.exe \
	hw_03.o \
	$(ISIP_LIBS)

# define a target to compile the application
#
hw_03.o: $(SRC) $(DEPS)
	g++ $(CFLAGS) -c $(SRC) $(INCLUDES) -o $(OBJ)

# define a target to clean the directory
#
clean:
	rm -f hw_03.exe hw_03.o

#
# end of file
