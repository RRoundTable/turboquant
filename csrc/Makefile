CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -Iinclude
LDFLAGS  :=

SRC      := src/pack.cpp src/quantize.cpp
TEST_SRC := tests/main.cpp tests/test_tile.cpp tests/test_pack.cpp \
            tests/test_fp16.cpp tests/test_roundtrip.cpp

OBJ      := $(SRC:.cpp=.o)
TEST_OBJ := $(TEST_SRC:.cpp=.o)

.PHONY: all test clean

all: test

build/turboquant_tests: $(OBJ) $(TEST_OBJ) | build
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

build:
	mkdir -p build

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

test: build/turboquant_tests
	./build/turboquant_tests

clean:
	rm -f $(OBJ) $(TEST_OBJ) build/turboquant_tests
	rmdir build 2>/dev/null || true
