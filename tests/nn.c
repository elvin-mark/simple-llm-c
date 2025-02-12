#include "llm/core/tensor.h"
#include "llm/nn/layers.h"
#include "llm/nn/blocks.h"
#include <stdlib.h>
#include <stdio.h>

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

void test_softmax_layer(){
    Tensor *o = create_tensor2("2 2");
    randomize_tensor(o);
    print_tensor(o);

    Tensor *m = softmax_layer(o, 1);
    print_tensor(m);

    free_tensor(o);
    free_tensor(m);
}

void test_rms_norm_layer(){
    Tensor *o = create_tensor2("2 2");
    randomize_tensor(o);
    print_tensor(o);

    Tensor *m = rms_norm_layer(o, 1);
    print_tensor(m);

    free_tensor(o);
    free_tensor(m);
}

void test_layer_norm_layer(){
    Tensor *o = create_tensor2("2 2");
    Tensor *g = create_tensor2("1 2");
    Tensor *b = create_tensor2("1 2");

    randomize_tensor(o);
    randomize_tensor(g);
    randomize_tensor(b);

    print_tensor(o);

    Tensor *m = layer_norm_layer(o, 1, g, b);
    print_tensor(m);

    free_tensor(o);
    free_tensor(m);
    free_tensor(g);
    free_tensor(b);
}
int main(){
    //test_ffn();
    //test_softmax_layer();
    //test_rms_norm_layer();
    test_layer_norm_layer();
}
