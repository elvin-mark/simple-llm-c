#ifndef CORE_MATRIX_H
#define CORE_MATRIX_H

typedef struct Matrix{
     float  *data;
     int     dim;
     int    *shape;
     int    *stride;
     int     size;
} Matrix;

Matrix* create_matrix(int dim, int *shape);
void    free_matrix(Matrix *m);
void    print_matrix(Matrix *m);
void    randomize_matrix(Matrix *m);

int* init_indices(int dim);
int  increase_indices(int dim, int* indices, int* shape); // if indices overflow then return 1 else return 0
int  get_pos(int dim, int *indices, int *stride);
int* get_max_shape(int dim, int *s1, int *s2);
int get_pos_from_running_indices(int* running_indices, int dim, int *idxs, int *stride);

void transpose_matrix(int *order); // TODO

Matrix* add_matrices(Matrix *m1, Matrix *m2);
Matrix* sub_matrices(Matrix *m1, Matrix *m2);
Matrix* mul_matrices(Matrix *m1, Matrix *m2);
Matrix* div_matrices(Matrix *m1, Matrix *m2);

Matrix *einsum(int num_idx, int *idxs1, Matrix *m1, int *idxs2, Matrix *m2, int dim, int *idxs);
Matrix *einsum2(char *idxs, Matrix *m1, Matrix *m2);
#endif

