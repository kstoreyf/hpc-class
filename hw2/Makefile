CC = g++

CXXFLAGS = -g -std=c++11 -O3 -march=native

TARGETS: MMult1 val_test01_solved val_test02_solved fast-sin

all: $(TARGETS) 

# There must be a way to do this in a loop, 
# but not sure what best practice is!

fast_sin: fast_sin.cpp
	$(CC) $(CXXFLAGS) -o fast_sin fast_sin.cpp

MMult1: MMult1.cpp
	$(CC) $(CXXFLAGS) -o MMult1 MMult1.cpp

val_test01_solved: val_test01_solved.cpp
	$(CC) $(CXXFLAGS) -o val_test01_solved val_test01_solved.cpp

val_test02_solved: val_test02_solved.cpp
	$(CC) $(CXXFLAGS) -o val_test02_solved val_test02_solved.cpp

clean:
	$(RM) $(TARGETS)
