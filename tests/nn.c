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

void test_attention(){
    Tensor *q = create_tensor2("5 16 12");
    Tensor *k = create_tensor2("5 16 12");
    Tensor *v = create_tensor2("5 16 12");

    randomize_tensor(q);
    randomize_tensor(k);
    randomize_tensor(v);

    print_tensor(q);
    print_tensor(k);
    print_tensor(v);

    Tensor *o = attention(q, k ,v, NULL);

    print_tensor(o);

    free_tensor(q);
    free_tensor(k);
    free_tensor(v);
    free_tensor(o);
}

void test_mha(){

    Tensor *q = create_tensor2("192 192");
    Tensor *k = create_tensor2("192 192");
    Tensor *v = create_tensor2("192 192");
    Tensor *p = create_tensor2("192 192");

    Tensor *x = create_tensor2("5 192");
    
    randomize_tensor(q);
    randomize_tensor(k);
    randomize_tensor(v);
    randomize_tensor(p);
    randomize_tensor(x);

    Tensor *o = mha(x,q, NULL, k, NULL, v, NULL, p, NULL, 12, 0);
    
    print_tensor(o);

    free_tensor(q);
    free_tensor(k);
    free_tensor(v);
    free_tensor(p);
    free_tensor(x);
    free_tensor(o);
}

int main(){
    //test_ffn();
    //test_softmax_layer();
    //test_rms_norm_layer();
    //test_layer_norm_layer();
    //test_attention();
    test_mha();
}
