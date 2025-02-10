CC=gcc
INCLUDE=../include

clean:
	cd build && rm *.o || echo "No objects" 
objs: clean 
	cd build && $(CC) -I$(INCLUDE) -c ../src/llm/**/*.c
test-core: objs
	cd build && $(CC) -I$(INCLUDE) -c ../tests/core.c
	cd build && $(CC) *.o -o test-core
