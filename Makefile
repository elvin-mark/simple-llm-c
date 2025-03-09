CC=gcc
INCLUDE=./include
CFLAGS+= -Ofast -fopenmp -O3 -ffast-math -lm
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
	$(CC) $(CFLAGS) -fPIC -I$(INCLUDE) -c $< -o $@

test-core: $(OBJS) | $(BUILD_DIR)
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/test-core.o -c ./tests/core.c
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/test-core $(BUILD_DIR)/test-core.o $(OBJS) $(CFLAGS)

test-nn: $(OBJS) | $(BUILD_DIR)
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/test-nn.o -c ./tests/nn.c
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/test-nn $(BUILD_DIR)/test-nn.o $(OBJS) $(CFLAGS)

test-gpt2: $(OBJS) | $(BUILD_DIR)
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/test-gpt2.o -c ./tests/gpt2.c
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/test-gpt2 $(BUILD_DIR)/test-gpt2.o $(OBJS) $(CFLAGS)

llm-server: $(OBJS) | $(BUILD_DIR)
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/llm-server.o -c ./server/main.c
	$(CC) -I$(INCLUDE) -o $(BUILD_DIR)/llm-server $(BUILD_DIR)/llm-server.o $(OBJS) $(CFLAGS)

clean:
	rm -r $(BUILD_DIR)

$(SHARED_LIB): $(OBJS) | $(BUILD_DIR)
	$(CC) $(OBJS) $(LDFLAGS) -o $(BUILD_DIR)/$(SHARED_LIB) 

