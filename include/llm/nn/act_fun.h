#ifndef NN_ACT_FUN_H
#define NN_ACT_FUN_H

#include <math.h>

float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

float tanh2(float x) {
  // TODO: deal with edge cases
  float a = expf(x);
  float b = 1.0 / a;
  return (a - b) / (a + b);
}

float relu(float x) { return x > 0 ? x : 0.0; }

float gelu(float x) {
  return 0.5 * x * (1 + tanhf(sqrtf(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

float silu(float x) { return x / (1.0 + expf(-x)); }

#endif
