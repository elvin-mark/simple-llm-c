#include "llm/models/gpt2.h"
#include "llm/core/tensor.h"
#include "llm/nn/blocks.h"
#include "llm/nn/layers.h"
#include <endian.h>
#include <stdlib.h>

void load_model(GPT2Config config, GPT2 gpt2, char *model_path) {
  // TODO
  int shape1[2] = {config.vocab_size, config.hidden_dim};
  int shape2[2] = {config.n_ctx, config.hidden_dim};
  int shape3[2] = {config.hidden_dim, config.hidden_dim};
  int shape4[2] = {1, config.hidden_dim};

  gpt2.wte = create_tensor(2, shape1);
  gpt2.wpe = create_tensor(2, shape2);

  gpt2.lnf_w = create_tensor(2, shape4);
  gpt2.lnf_b = create_tensor(2, shape4);

  gpt2.num_blocks = config.num_blocks;
  gpt2.blocks = malloc(sizeof(GPT2Block) * gpt2.num_blocks);
  for (int i = 0; i < config.num_blocks; i++) {
    gpt2.blocks[i].ln1_w = create_tensor(2, shape4);
    gpt2.blocks[i].ln1_b = create_tensor(2, shape4);
    gpt2.blocks[i].ln2_w = create_tensor(2, shape4);
    gpt2.blocks[i].ln2_b = create_tensor(2, shape4);

    gpt2.blocks[i].attn.q_w = create_tensor(2, shape3);
    gpt2.blocks[i].attn.q_b = create_tensor(2, shape4);
    gpt2.blocks[i].attn.k_w = create_tensor(2, shape3);
    gpt2.blocks[i].attn.k_b = create_tensor(2, shape4);
    gpt2.blocks[i].attn.v_w = create_tensor(2, shape3);
    gpt2.blocks[i].attn.v_b = create_tensor(2, shape4);
    gpt2.blocks[i].attn.proj_w = create_tensor(2, shape3);
    gpt2.blocks[i].attn.proj_b = create_tensor(2, shape4);

    gpt2.blocks[i].mlp.fc_w = create_tensor(2, shape3);
    gpt2.blocks[i].mlp.fc_b = create_tensor(2, shape4);
    gpt2.blocks[i].mlp.proj_w = create_tensor(2, shape3);
    gpt2.blocks[i].mlp.proj_b = create_tensor(2, shape4);
  }
}

void free_model(GPT2 gpt2) {
  free_tensor(gpt2.wte);
  free_tensor(gpt2.wpe);
  for (int i = 0; i < gpt2.num_blocks; i++) {
    free_tensor(gpt2.blocks[i].ln1_w);
    free_tensor(gpt2.blocks[i].ln1_b);
    free_tensor(gpt2.blocks[i].ln2_w);
    free_tensor(gpt2.blocks[i].ln2_b);
    free_tensor(gpt2.blocks[i].mlp.fc_w);

    free_tensor(gpt2.blocks[i].mlp.fc_w);
    free_tensor(gpt2.blocks[i].mlp.fc_b);
    free_tensor(gpt2.blocks[i].mlp.proj_w);
    free_tensor(gpt2.blocks[i].mlp.proj_b);

    free_tensor(gpt2.blocks[i].attn.q_w);
    free_tensor(gpt2.blocks[i].attn.q_b);
    free_tensor(gpt2.blocks[i].attn.k_w);
    free_tensor(gpt2.blocks[i].attn.k_b);
    free_tensor(gpt2.blocks[i].attn.v_w);
    free_tensor(gpt2.blocks[i].attn.v_b);
    free_tensor(gpt2.blocks[i].attn.proj_w);
    free_tensor(gpt2.blocks[i].attn.proj_b);
  }
  free_tensor(gpt2.lnf_w);
  free_tensor(gpt2.lnf_b);
}

Tensor *gpt2_transformer_block(Tensor *x, GPT2Block block, int n_head) {
  Tensor *o1_ = layer_norm_layer(x, 1, block.ln1_w, block.ln1_b);
  Tensor *o2_ = mha(o1_, block.attn.q_w, block.attn.q_b, block.attn.k_w,
                    block.attn.k_b, block.attn.v_w, block.attn.v_b,
                    block.attn.proj_w, block.attn.proj_b, n_head, 1);
  Tensor *o3_ = add_tensors(x, o2_);

  Tensor *o4_ = layer_norm_layer(o3_, 1, block.ln2_w, block.ln2_b);
  Tensor *o5_ = ffn(o4_, block.mlp.fc_w, block.mlp.fc_b, block.mlp.proj_w,
                    block.mlp.proj_b, gelu_layer);
  Tensor *o6_ = add_tensors(o3_, o6_);

  free_tensor(o1_);
  free_tensor(o2_);
  free_tensor(o3_);
  free_tensor(o4_);
  free_tensor(o5_);

  return o6_;
}

Tensor *gpt2_forward(int num_inputs, int *inputs, GPT2 gpt2, int n_head) {
  Tensor *o1_ = slice(gpt2.wte, num_inputs, inputs);
  int *range = malloc(sizeof(int) * num_inputs);
  for (int i = 0; i < num_inputs; i++)
    range[i] = i;
  Tensor *o2_ = slice(gpt2.wpe, num_inputs, range);
  Tensor *o_ = add_tensors(o1_, o2_);
  free_tensor(o1_);
  free_tensor(o2_);

  Tensor *tmp;
  for (int i = 0; i < gpt2.num_blocks; i++) {
    tmp = gpt2_transformer_block(o_, gpt2.blocks[i], n_head);
    free_tensor(o_);
    o_ = tmp;
  }

  Tensor *o3_ = layer_norm_layer(o_, 1, gpt2.lnf_w, gpt2.lnf_b);
  Tensor *o4_ = einsum2("ij kj ik", o3_, gpt2.wte);
  free_tensor(o_);
  free_tensor(o3_);
  free(range);

  return o4_;
}

void gpt2_generate(int *num_inputs, int *inputs, GPT2 gpt2, int n_head,
                   int n_tokens) {
  Tensor *o;
  int token;
  int score;
  int k;
  int num_inputs_ = *num_inputs;
  for (int i = 0; i < n_tokens; i++) {
    k = num_inputs_ - 1;
    o = gpt2_forward(num_inputs_, inputs, gpt2, n_head);
    token = 0;
    score = o->data[k * o->stride[0]];
    for (int j = 1; j < o->shape[1]; j++)
      if (o->data[k * o->stride[0] + j] > score)
        token = j;
    inputs = realloc(inputs, sizeof(int) * (num_inputs_ + 1));
    inputs[num_inputs_] = token;
    num_inputs_ += 1;
    free_tensor(o);
  }
  *num_inputs = num_inputs_;
}
