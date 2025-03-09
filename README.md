# Simple LLM inference written in C

This is just a personal project for me to practice C and explore how LLM works.

# Build me

You can build the library by running the following command

```sh
make libtinyllm.so
```

# Run GPT2

```sh
make test-gpt2
```

```sh
CFLAGS="-DORIGINAL" make clean test-gpt2 
```

```sh
CFLAGS="-DBLAS -lblas" make clean test-gpt2 
```

```sh
./build/test-gpt2
```
