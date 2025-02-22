#include "llm/core/tensor.h"
#include "llm/nn/act_fun.h"
#include <math.h>

float square(float x) { return x * x; }

Tensor *linear_layer(Tensor *m, Tensor *w, Tensor *b) {
  Tensor *mw = matmul(m, w);
  if (b) {
    Tensor *o = add_tensors(mw, b);
    free_tensor(mw);
    return o;
  }
  return mw;
}

Tensor *sigmoid_layer(Tensor *m) { return apply_fn_to_tensor(m, sigmoid); }

Tensor *tanh_layer(Tensor *m) { return apply_fn_to_tensor(m, tanh2); }

Tensor *relu_layer(Tensor *m) { return apply_fn_to_tensor(m, relu); }

Tensor *silu_layer(Tensor *m) { return apply_fn_to_tensor(m, silu); }

Tensor *gelu_layer(Tensor *m) { return apply_fn_to_tensor(m, gelu); }

Tensor *softmax_layer(Tensor *m, int index) {
  Tensor *max_ = tensor_max(m, index);
  Tensor *o1_ = sub_tensors(m, max_);
  Tensor *expx_ = apply_fn_to_tensor(o1_, expf);
  Tensor *sumex_ = tensor_sum(expx_, index);
  Tensor *o = div_tensors(expx_, sumex_);

  free_tensor(max_);
  free_tensor(o1_);
  free_tensor(expx_);
  free_tensor(sumex_);

  return o;
}

Tensor *rms_norm_layer(Tensor *m, int index) {
  Tensor *m2_ = apply_fn_to_tensor(m, square);
  Tensor *mean_ = tensor_mean(m2_, index);
  for (int i = 0; i < mean_->size; i++)
    mean_->data[i] += 1e-6;
  Tensor *sqrt_ = apply_fn_to_tensor(mean_, sqrtf);
  Tensor *o = div_tensors(m, sqrt_);

  free_tensor(m2_);
  free_tensor(mean_);
  free_tensor(sqrt_);

  return o;
}

Tensor *layer_norm_layer(Tensor *m, int index, Tensor *g, Tensor *b) {
  Tensor *mean_ = tensor_mean(m, index);
  Tensor *var_ = tensor_var(m, index);
  for (int i = 0; i < var_->size; i++)
    var_->data[i] += 1e-12;
  Tensor *o_ = sub_tensors(m, mean_);
  Tensor *sqrt_ = apply_fn_to_tensor(var_, sqrtf);
  Tensor *o1_ = mul_tensors(g, o_);
  Tensor *o2_ = div_tensors(o1_, sqrt_);

  Tensor *o3_ = o2_;
  if (b)
    o3_ = add_tensors(o2_, b);

  free_tensor(mean_);
  free_tensor(var_);
  free_tensor(o_);
  free_tensor(sqrt_);
  free_tensor(o1_);
  free_tensor(o2_);

  return o3_;
}
