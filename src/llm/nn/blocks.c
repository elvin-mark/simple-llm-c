#include "llm/core/tensor.h"
#include "llm/nn/blocks.h"
#include "llm/nn/layers.h"
#include "llm/utils/errors.h"
#include <math.h>

Tensor *ffn(Tensor *m, Tensor *w1, Tensor *b1, Tensor *w2, Tensor *b2, Tensor* (*act_fn)(Tensor*)){
    Tensor *o1 = linear_layer(m, w1, b1);
    Tensor *o2 = act_fn(o1);
    Tensor *o3 = linear_layer(o2, w2, b2);

    free_tensor(o1);
    free_tensor(o2);

    return o3;
}

Tensor *attention(Tensor *q, Tensor *k, Tensor *v, Tensor *mask){
    // q, k and v are of dimmension N, dim/n_head, n_head
    assert(q->dim == 3 && k->dim == 3 && v->dim == 3, "q, k and v must be of 3-dimension tensors");
    int dim = q->shape[1] * q->shape[2];
    int scale = 1 / sqrtf(1.0 * dim);

    Tensor *qk_ = einsum2("ijk ljk ilk", q, k); 
    for(int i = 0; i<qk_->size; i++) qk_->data[i] *= scale;
    
    if(mask){
        Tensor *m_ = qk_;
        m_ = add_tensors(qk_, mask);
        free_tensor(qk_);
        qk_ = m_;
    }
    
    Tensor *s_ = softmax_layer(qk_,1);
    Tensor *o_ = einsum2("ijk jlk ilk", s_, v);

    free_tensor(qk_);
    free_tensor(s_);

    return o_;
}
