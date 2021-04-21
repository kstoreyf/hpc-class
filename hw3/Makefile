CXX = g++
CXXFLAGS = -std=c++11 -g -O3 -fopenmp -march=native

RM = rm -f
MKDIRS = mkdir -p

TARGETS = $(basename $(wildcard *.c)) $(basename $(wildcard *.cpp))

all : $(TARGETS)

%:%.c
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	$(RM) $(TARGETS)
