CXX      := g++
NVCC     := nvcc
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -Iinclude
NVFLAGS  := -std=c++17 -O2 -Iinclude --expt-relaxed-constexpr
LDFLAGS  :=

SRC      := src/pack.cpp src/quantize.cpp
TEST_SRC := tests/main.cpp tests/test_tile.cpp tests/test_pack.cpp \
            tests/test_fp16.cpp tests/test_roundtrip.cpp

CUDA_TEST_SRC := tests/test_write_kernel.cu

OBJ      := $(SRC:.cpp=.o)
TEST_OBJ := $(TEST_SRC:.cpp=.o)

.PHONY: all test test-cuda clean

all: test

build/turboquant_tests: $(OBJ) $(TEST_OBJ) | build
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

build/test_write_kernel: $(OBJ) $(CUDA_TEST_SRC) | build
	$(NVCC) $(NVFLAGS) -o $@ $(OBJ) $(CUDA_TEST_SRC) -lcudart

build:
	mkdir -p build

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

test: build/turboquant_tests
	./build/turboquant_tests

test-cuda: build/test_write_kernel
	./build/test_write_kernel

clean:
	rm -f $(OBJ) $(TEST_OBJ) build/turboquant_tests build/test_write_kernel
	rmdir build 2>/dev/null || true
