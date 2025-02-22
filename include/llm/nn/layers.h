#ifndef NN_LAYERS_H
#define NN_LAYERS_H

#include "llm/core/tensor.h"

Tensor *linear_layer(Tensor *m, Tensor *w, Tensor *b);
Tensor *sigmoid_layer(Tensor *m);
Tensor *tanh_layer(Tensor *m);
Tensor *relu_layer(Tensor *m);
Tensor *gelu_layer(Tensor *m);
Tensor *silu_layer(Tensor *m);
Tensor *softmax_layer(Tensor *m, int index);
Tensor *rms_norm_layer(Tensor *m, int index);
Tensor *layer_norm_layer(Tensor *m, int index, Tensor *g, Tensor *b);
#endif
