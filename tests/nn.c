#include "llm/core/tensor.h"
#include "llm/nn/layers.h"
#include "llm/nn/blocks.h"
#include <stdlib.h>

void test_ffn(){
    Tensor *x = create_tensor2("2 5");
    Tensor *w1 = create_tensor2("5 3");
    Tensor *b1 = create_tensor2("1 3");
    Tensor *w2 = create_tensor2("3 4");
    Tensor *b2 = NULL;

    randomize_tensor(x);
    randomize_tensor(w1);
    randomize_tensor(b1);
    randomize_tensor(w2);
    randomize_tensor(b2);


    Tensor *o = ffn(x, w1, b1, w2, b2, sigmoid_layer);

    print_tensor(o);

    free_tensor(x);
    free_tensor(w1);
    free_tensor(b1);
    free_tensor(w2);
    free_tensor(b2);
}

int main(){
    test_ffn();
}
