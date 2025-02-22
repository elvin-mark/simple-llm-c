#include "llm/core/tensor.h"
#include <stdio.h>

void test_tensor_creation() {
  int shape[3] = {2, 3, 4};
  Tensor *m = create_tensor(3, shape);
  randomize_tensor(m);
  print_tensor(m);
  free_tensor(m);
}

void test_tensor_creation2() {
  Tensor *m = create_tensor2("2 3 4");
  randomize_tensor(m);
  print_tensor(m);
  free_tensor(m);
}

void test_tri_matrix() {
  Tensor *o = tri_matrix(3);
  print_tensor(o);
  free_tensor(o);
}

void test_basic_tensor_operations() {
  int shape[2] = {2, 3};
  Tensor *m1 = create_tensor(2, shape);
  Tensor *m2 = create_tensor(2, shape);

  randomize_tensor(m1);
  randomize_tensor(m2);

  print_tensor(m1);
  print_tensor(m2);

  Tensor *o = add_tensors(m1, m2);
  print_tensor(o);

  free_tensor(m1);
  free_tensor(m2);
  free_tensor(o);
}

void test_einsum() {
  int shape[2] = {2, 2};
  Tensor *m1 = create_tensor(2, shape);
  Tensor *m2 = create_tensor(2, shape);

  randomize_tensor(m1);
  randomize_tensor(m2);

  for (int i = 0; i < 4; i++)
    printf("%.2f ", m1->data[i]);
  printf("\n");

  print_tensor(m1);
  print_tensor(m2);

  int idxs1[2] = {0, 1};
  int idxs2[2] = {1, 2};
  int idxs[2] = {0, 2};
  Tensor *o = einsum(3, idxs1, m1, idxs2, m2, 2, idxs);
  print_tensor(o);

  free_tensor(m1);
  free_tensor(m2);
  free_tensor(o);
}

void test_einsum2() {
  int shape[2] = {2, 2};
  Tensor *m1 = create_tensor(2, shape);
  Tensor *m2 = create_tensor(2, shape);

  randomize_tensor(m1);
  randomize_tensor(m2);

  for (int i = 0; i < 4; i++)
    printf("%.2f ", m1->data[i]);
  printf("\n");

  print_tensor(m1);
  print_tensor(m2);

  Tensor *o = einsum2("ij jk ik\0", m1, m2);
  print_tensor(o);

  free_tensor(m1);
  free_tensor(m2);
  free_tensor(o);
}

void test_matmul() {
  int shape[2] = {2, 2};
  Tensor *m1 = create_tensor(2, shape);
  Tensor *m2 = create_tensor(2, shape);

  randomize_tensor(m1);
  randomize_tensor(m2);

  for (int i = 0; i < 4; i++)
    printf("%.2f ", m1->data[i]);
  printf("\n");

  print_tensor(m1);
  print_tensor(m2);

  Tensor *o = matmul(m1, m2);
  print_tensor(o);

  free_tensor(m1);
  free_tensor(m2);
  free_tensor(o);
}

void test_transpose() {
  int new_order[2] = {1, 0};
  Tensor *m = create_tensor2("2 2");

  randomize_tensor(m);

  Tensor *mT = clone_tensor(m);
  transpose_tensor(mT, new_order);
  Tensor *o = matmul(m, mT);

  print_tensor(m);
  print_tensor(mT);
  print_tensor(o);

  free_tensor(m);
  free_tensor(mT);
  free_tensor(o);
}

float pow2(float x) { return x * x; }
void test_apply_fn() {
  Tensor *m = create_tensor2("2 2");
  randomize_tensor(m);
  Tensor *o = apply_fn_to_tensor(m, pow2);

  print_tensor(m);
  print_tensor(o);

  free_tensor(m);
  free_tensor(o);
}

void test_tensor_sum() {
  Tensor *o = create_tensor2("2 2");
  randomize_tensor(o);
  print_tensor(o);
  Tensor *m = tensor_sum(o, 1);
  print_tensor(m);

  free_tensor(m);
  free_tensor(o);
}

void test_tensor_mean() {
  Tensor *o = create_tensor2("2 2");
  randomize_tensor(o);
  print_tensor(o);
  Tensor *m = tensor_mean(o, 1);
  print_tensor(m);

  free_tensor(m);
  free_tensor(o);
}

void test_tensor_var() {
  Tensor *o = create_tensor2("2 2");
  randomize_tensor(o);
  print_tensor(o);
  Tensor *m = tensor_var(o, 1);
  print_tensor(m);

  free_tensor(m);
  free_tensor(o);
}

void test_tensor_max() {
  Tensor *o = create_tensor2("2 4 6");
  randomize_tensor(o);

  print_tensor(o);

  Tensor *m = tensor_max(o, 2);
  print_tensor(m);

  free_tensor(o);
  free_tensor(m);
}

int main() {
  // test_tensor_creation();
  // test_tensor_creation2();
  test_tri_matrix();
  // test_basic_tensor_operations();
  // test_einsum();
  // test_einsum2();
  // test_matmul();
  // test_transpose();
  // test_apply_fn();
  // test_tensor_sum();
  // test_tensor_mean();
  // test_tensor_var();
  // test_tensor_max();
  return 0;
}
