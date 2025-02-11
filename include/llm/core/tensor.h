#ifndef CORE_TENSOR_H
#define CORE_TENSOR_H

typedef struct Tensor{
     float  *data;
     int     dim;
     int    *shape;
     int    *stride;
     int     size;
} Tensor;

Tensor* create_tensor(int dim, int *shape);
Tensor* create_tensor2(char *shape_string); // Wrapper over create_tensor for easy initialization. shape string should be like "2 3 4"
void    free_tensor(Tensor *m);
void    print_tensor(Tensor *m);
void    randomize_tensor(Tensor *m);
Tensor* clone_tensor(Tensor *m);

int* init_indices(int dim);
int  increase_indices(int dim, int* indices, int* shape); // if indices overflow then return 1 else return 0
int  get_pos(int dim, int *indices, int *stride);
int* get_max_shape(int dim, int *s1, int *s2);
int get_pos_from_running_indices(int* running_indices, int dim, int *idxs, int *stride);

void transpose_tensor(Tensor *m, int *order); 

Tensor* add_tensors(Tensor *m1, Tensor *m2);
Tensor* sub_tensors(Tensor *m1, Tensor *m2);
Tensor* mul_tensors(Tensor *m1, Tensor *m2);
Tensor* div_tensors(Tensor *m1, Tensor *m2);

Tensor *einsum(int num_idx, int *idxs1, Tensor *m1, int *idxs2, Tensor *m2, int dim, int *idxs);
Tensor *einsum2(char *idxs, Tensor *m1, Tensor *m2); // Wrapper over einsum for easy calling

Tensor *matmul(Tensor *m1, Tensor *m2);
Tensor *apply_fn_to_tensor(Tensor *m, float (*fn)(float));
#endif

