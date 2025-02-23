#ifndef MODELS_GPT2_H
#define MODELS_GPT2_H

#include "llm/core/tensor.h"

typedef struct GPT2FFN {
  Tensor *fc_w;
  Tensor *fc_b;
  Tensor *proj_w;
  Tensor *proj_b;
} GPT2FFN;

typedef struct GPT2Attn {
  Tensor *q_w;
  Tensor *q_b;
  Tensor *k_w;
  Tensor *k_b;
  Tensor *v_w;
  Tensor *v_b;
  Tensor *proj_w;
  Tensor *proj_b;
} GPT2Attn;

typedef struct GPT2Block {
  Tensor *ln1_w;
  Tensor *ln1_b;
  Tensor *ln2_w;
  Tensor *ln2_b;
  GPT2FFN mlp;
  GPT2Attn attn;
} GPT2Block;

typedef struct GPT2 {
  Tensor *wte;
  Tensor *wpe;
  int num_blocks;
  GPT2Block *blocks;
  Tensor *lnf_w;
  Tensor *lnf_b;
} GPT2;

typedef struct GPT2Config {
  int hidden_dim;
  int num_head;
  int vocab_size;
  int n_ctx;
  int fc_dim;
  int num_blocks;
} GPT2Config;

GPT2 *load_model(GPT2Config config, char *model_path);
void free_model(GPT2 *gpt2);
Tensor *gpt2_transformer_block(Tensor *x, GPT2Block block, int n_head);
Tensor *gpt2_forward(int num_inputs, int *inputs, GPT2 gpt2, int n_head);
void gpt2_generate(int *num_inputs, int *inputs, GPT2 gpt2, int n_head,
                   int n_tokens);

#endif
