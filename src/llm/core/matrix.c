#include "llm/core/matrix.h"
#include "llm/utils/errors.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

Matrix* create_matrix(int dim, int* shape){
    Matrix *m = malloc(sizeof(Matrix));
    int *stride = malloc(sizeof(int) * dim);
    int *shape_ = malloc(sizeof(int) * dim);
    int size = 1;
    int acc = 1;
    
    for(int i = 0;i<dim;i++){
        shape_[i] = shape[i];
        size *= shape[i];
    }

    for(int i = dim - 1; i>=0;i--){
        if(shape[i] == 1)
            stride[i] = 0; // to be able to broadcast
        else
            stride[i] = acc;
        acc *= shape[i];
    }

    m->data = malloc(sizeof(float) * size);
    m->dim = dim;
    m->shape = shape_;
    m->stride = stride;
    m->size = size;
    return m;
}

void free_matrix(Matrix *m){
    free(m->data);
    free(m->shape);
    free(m->stride);
    free(m);
}

void print_matrix(Matrix *m){
    printf("Dimension: %d\n", m->dim);
    printf("Shape: ");
    for(int i = 0;i<m->dim;i++){
        printf("%d ",m->shape[i]);
    }
    printf("\n");
    printf("Data: ");
    if(m->size > 3){
        printf("%.2f %.2f ... %.2f",m->data[0],m->data[1],m->data[m->size-1]);
    }
    else{
        for(int i = 0;i<m->size;i++){
            printf("%.2f ",m->data[i]);
        }
    }
    printf("\n");
}

void randomize_matrix(Matrix *m){
    srand(time(NULL));
    for(int i = 0;i<m->size;i++){
        m->data[i] = 1.0 * rand() / RAND_MAX;
    }
}

int* init_indices(int dim){
    int* indices = malloc(sizeof(dim));
    for(int i = 0;i<dim;i++)
        indices[i] = 0;
    return indices;
}

int increase_indices(int dim, int* indices, int* shape){
   int c = 1;
   for(int i = dim-1; i>=0 ; i--){
        indices[i] += c;
        c = indices[i] / shape[i];
        indices[i] %= shape[i];
   }
   return c;
}

int get_pos(int dim, int *indices, int *stride){
    int s = 0;
    for(int i = 0; i<dim ; i++)
        s += stride[i] * indices[i];
    return s;
}

int* get_max_shape(int dim, int *s1, int *s2){
   int* new_shape = malloc(sizeof(int) * dim);
   for(int i = 0;i<dim;i++)
        new_shape[i] = s1[i] > s2[i] ? s1[i] : s2[i];
  return new_shape; 
}

Matrix* add_matrices(Matrix *m1, Matrix *m2){
   assert(m1->dim == m2->dim, "dimension of the matrics do not match");
   int dim = m1->dim;
   int *shape = get_max_shape(dim, m1->shape, m2->shape);
   int *indices = init_indices(dim);
   int pos1, pos2, pos;

   Matrix *o = create_matrix(dim, shape);
   
   do{
    pos1 = get_pos(dim, indices, m1->stride);
    pos2 = get_pos(dim, indices, m2->stride);
    pos = get_pos(dim, indices, o->stride);
    o->data[pos] = m1->data[pos1] + m2->data[pos2];
   }while(!increase_indices(dim, indices, shape));

   return o;
}

Matrix* sub_matrices(Matrix *m1, Matrix *m2){
   assert(m1->dim == m2->dim, "dimension of the matrics do not match");
   int dim = m1->dim;
   int *shape = get_max_shape(dim, m1->shape, m2->shape);
   int *indices = init_indices(dim);
   int pos1, pos2, pos;

   Matrix *o = create_matrix(dim, shape);
   
   do{
    pos1 = get_pos(dim, indices, m1->stride);
    pos2 = get_pos(dim, indices, m2->stride);
    pos = get_pos(dim, indices, o->stride);
    o->data[pos] = m1->data[pos1] - m2->data[pos2];
   }while(!increase_indices(dim, indices, shape));

   return o;
}

Matrix* mul_matrices(Matrix *m1, Matrix *m2){
   assert(m1->dim == m2->dim, "dimension of the matrics do not match");
   int dim = m1->dim;
   int *shape = get_max_shape(dim, m1->shape, m2->shape);
   int *indices = init_indices(dim);
   int pos1, pos2, pos;

   Matrix *o = create_matrix(dim, shape);
   
   do{
    pos1 = get_pos(dim, indices, m1->stride);
    pos2 = get_pos(dim, indices, m2->stride);
    pos = get_pos(dim, indices, o->stride);
    o->data[pos] = m1->data[pos1] * m2->data[pos2];
   }while(!increase_indices(dim, indices, shape));

   return o;
}

Matrix* div_matrices(Matrix *m1, Matrix *m2){
   assert(m1->dim == m2->dim, "dimension of the matrics do not match");
   int dim = m1->dim;
   int *shape = get_max_shape(dim, m1->shape, m2->shape);
   int *indices = init_indices(dim);
   int pos1, pos2, pos;

   Matrix *o = create_matrix(dim, shape);
   
   do{
    pos1 = get_pos(dim, indices, m1->stride);
    pos2 = get_pos(dim, indices, m2->stride);
    pos = get_pos(dim, indices, o->stride);
    o->data[pos] = m1->data[pos1] / m2->data[pos2];
   }while(!increase_indices(dim, indices, shape));

   return o;
}


