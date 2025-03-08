#include "llm/models/gpt2.h"
#include <stdio.h>
#include <stdlib.h>
int main() {
  GPT2Config config = {768, 12, 50257, 1024, 3072, 12};
  GPT2 *gpt2 = load_model(config, "model.bin");
  int count = 1;
  int *inputs = malloc(sizeof(int) * count);

  inputs[0] = 30000;
  gpt2_generate(&count, &inputs, *gpt2, 12, 8);

  for (int i = 0; i < count; i++)
    printf("%d ", inputs[i]);

  free_model(gpt2);
  free(inputs);
  return 0;
}
