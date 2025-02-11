#include "llm/core/tensor.h"
#include "llm/nn/blocks.h"
#include "llm/nn/layers.h"

Tensor * ffn(Tensor *m, Tensor *w1, Tensor *b1, Tensor *w2, Tensor *b2, Tensor* (*act_fn)(Tensor*)){
    Tensor *o1 = linear_layer(m, w1, b1);
    Tensor *o2 = act_fn(o1);
    Tensor *o3 = linear_layer(o2, w2, b2);

    free_tensor(o1);
    free_tensor(o2);

    return o3;
}

