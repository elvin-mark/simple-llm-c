#include "llm/core/tensor.h"
#include "llm/nn/act_fun.h"

Tensor* linear_layer(Tensor *m, Tensor *w, Tensor *b){
    Tensor *mw = matmul(m, w);
    if(b){
        Tensor *o = add_tensors(mw, b);
        free_tensor(mw);
        return o;
    }
    return mw;
}

Tensor* sigmoid_layer(Tensor *m){
    return apply_fn_to_tensor(m, sigmoid);
}

Tensor* tanh_layer(Tensor *m){
    return apply_fn_to_tensor(m, tanh2);
}

Tensor* relu_layer(Tensor *m){
    return apply_fn_to_tensor(m, relu);
}

