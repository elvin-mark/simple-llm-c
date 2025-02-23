#include "llm/nn/blocks.h"
#include "llm/core/tensor.h"
#include "llm/nn/layers.h"
#include "llm/utils/errors.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Tensor *ffn(Tensor *m, Tensor *w1, Tensor *b1, Tensor *w2, Tensor *b2,
            Tensor *(*act_fn)(Tensor *)) {
  Tensor *o1 = linear_layer(m, w1, b1);
  Tensor *o2 = act_fn(o1);
  Tensor *o3 = linear_layer(o2, w2, b2);

  free_tensor(o1);
  free_tensor(o2);

  return o3;
}

Tensor *attention(Tensor *q, Tensor *k, Tensor *v, Tensor *mask) {
  // q, k and v are of dimmension N, dim/n_head, n_head
  assert(q->dim == 3 && k->dim == 3 && v->dim == 3,
         "q, k and v must be of 3-dimension tensors");
  int dim = q->shape[1];
  float scale = 1.0 / sqrtf(1.0 * dim);

  Tensor *qk_ = einsum2("ikj lkj ikl", q, k);
  for (int i = 0; i < qk_->size; i++)
    qk_->data[i] *= scale;

  if (mask) {
    Tensor *m_ = qk_;
    m_ = add_tensors(qk_, mask);
    free_tensor(qk_);
    qk_ = m_;
  }
  Tensor *s_ = softmax_layer(qk_, 2);
  Tensor *o_ = einsum2("ikj jkl ikl", s_, v);

  free_tensor(qk_);
  free_tensor(s_);

  return o_;
}

Tensor *mha(Tensor *x, Tensor *q_w, Tensor *q_b, Tensor *k_w, Tensor *k_b,
            Tensor *v_w, Tensor *v_b, Tensor *proj_w, Tensor *proj_b,
            int n_head, int mask_enabled) {

  Tensor *q = linear_layer(x, q_w, q_b);
  Tensor *k = linear_layer(x, k_w, k_b);
  Tensor *v = linear_layer(x, v_w, v_b);

  int *new_shape = malloc(sizeof(int) * 3);
  new_shape[0] = x->shape[0];
  new_shape[1] = n_head;
  new_shape[2] = x->shape[1] / n_head;

  reshape_tensor(q, 3, new_shape);
  reshape_tensor(k, 3, new_shape);
  reshape_tensor(v, 3, new_shape);

  Tensor *mask = NULL;
  if (mask_enabled) {
    mask = tri_matrix(x->shape[0]);
    new_shape[1] = 1;
    new_shape[2] = x->shape[0];
    reshape_tensor(mask, 3, new_shape);
    for (int i = 0; i < mask->size; i++)
      mask->data[i] = (1.0 - mask->data[i]) * -1e10;
  }

  Tensor *o1_ = attention(q, k, v, mask);
  new_shape[2] = n_head;
  new_shape[1] = x->shape[1];
  reshape_tensor(o1_, 2, new_shape);

  Tensor *o = linear_layer(o1_, proj_w, proj_b);

  free(new_shape);
  free_tensor(q);
  free_tensor(k);
  free_tensor(v);
  free_tensor(o1_);
  if (mask_enabled)
    free_tensor(mask);
  return o;
}
