#ifndef NN_BLOCKS_H
#define NN_BLOCKS_H

Tensor *ffn(Tensor *m, Tensor *w1, Tensor *b1, Tensor *w2, Tensor *b2, Tensor* (*act_fn)(Tensor*));
Tensor *attention(Tensor *q, Tensor *k, Tensor *v, Tensor *mask);

#endif
