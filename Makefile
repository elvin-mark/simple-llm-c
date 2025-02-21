CC=gcc
INCLUDE=./include
CFLAGS= -fPIC
LDFLAGS= -shared

BUILD_DIR=./build
SRC_DIR=./src/llm

SRCS=$(wildcard $(SRC_DIR)/**/*.c)
OBJS=$(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))

SHARED_LIB=libtinyllm.so

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCLUDE) -c $< -o $@

test-core: $(OBJS) | $(BUILD_DIR)
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/test-core.o -c ./tests/core.c
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/test-core $(BUILD_DIR)/test-core.o $(OBJS) -lm

test-nn: $(OBJS) | $(BUILD_DIR)
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/test-nn.o -c ./tests/nn.c
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/test-nn $(BUILD_DIR)/test-nn.o $(OBJS) -lm

test-gpt2: $(OBJS) | $(BUILD_DIR)
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/test-gpt2.o -c ./tests/gpt2.c
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/test-gpt2 $(BUILD_DIR)/test-gpt2.o $(OBJS) -lm

clean:
	rm -r $(BUILD_DIR)

$(SHARED_LIB): $(OBJS) | $(BUILD_DIR)
	$(CC) $(OBJS) $(LDFLAGS) -o $(BUILD_DIR)/$(SHARED_LIB) 

