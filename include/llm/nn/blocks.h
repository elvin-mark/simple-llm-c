#ifndef NN_BLOCKS_H
#define NN_BLOCKS_H
#include "llm/core/tensor.h"

Tensor *ffn(Tensor *m, Tensor *w1, Tensor *b1, Tensor *w2, Tensor *b2,
            Tensor *(*act_fn)(Tensor *));
Tensor *attention(Tensor *q, Tensor *k, Tensor *v, Tensor *mask);
Tensor *mha(Tensor *x, Tensor *q_w, Tensor *q_b, Tensor *k_w, Tensor *k_b,
            Tensor *v_w, Tensor *v_b, Tensor *proj_w, Tensor *proj_b,
            int n_head, int mask_enabled);

#endif
