#ifndef NN_LAYERS_H
#define NN_LAYERS_H

#include "llm/core/tensor.h"

Tensor* linear_layer(Tensor *m, Tensor *w, Tensor *b);
Tensor* sigmoid_layer(Tensor *m);
Tensor* tanh_layer(Tensor *m);
Tensor* relu_layer(Tensor *m);

#endif
