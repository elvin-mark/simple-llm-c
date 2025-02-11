#include "llm/core/tensor.h"
#include "llm/utils/errors.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

Tensor* create_tensor(int dim, int* shape){
    Tensor *m = malloc(sizeof(Tensor));
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

Tensor *create_tensor2(char *shape_string){
    int *shape = NULL;
    int dim = 0;
    Tensor *o;
    char buffer[256] = {};
    strncpy(buffer, shape_string, sizeof(buffer));
    char *token = strtok(buffer, " ");
    while(token){
        dim++;
        shape = realloc(shape, sizeof(int) * dim);
        shape[dim - 1] = atoi(token);
        token = strtok(NULL, " ");
    }
    o = create_tensor(dim, shape);
    free(shape);
    return o;
}

void free_tensor(Tensor *m){
    if(m == NULL) return;
    free(m->data);
    free(m->shape);
    free(m->stride);
    free(m);
}

void print_tensor(Tensor *m){
    printf("Dimension: %d\n", m->dim);
    
    printf("Shape: ");
    for(int i = 0;i<m->dim;i++)
        printf("%d ",m->shape[i]);
    printf("\n");

    printf("Data: ");
    if(m->size > 3)
        printf("%.2f %.2f ... %.2f",m->data[0],m->data[1],m->data[m->size-1]);
    else
        for(int i = 0;i<m->size;i++)
            printf("%.2f ",m->data[i]);
    printf("\n");
}

void randomize_tensor(Tensor *m){
    if(m == NULL) return;
    srand(time(NULL));
    for(int i = 0;i<m->size;i++){
        m->data[i] = 1.0 * rand() / RAND_MAX;
    }
}

Tensor* clone_tensor(Tensor *m){
    Tensor *o = create_tensor(m->dim, m->shape);
    for(int i = 0;i<m->size;i++)
        o->data[i] = m->data[i];
    return o;
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

void transpose_tensor(Tensor *m, int *order){
    int new_stride[m->dim] = {};
    for(int i=0; i<m->dim; i++)
        new_stride[i] = m->stride[order[i]];
    for(int i=0; i<m->dim; i++)
        m->stride[i] = new_stride[i];
}

int* get_max_shape(int dim, int *s1, int *s2){
    int* new_shape = malloc(sizeof(int) * dim);
    for(int i = 0;i<dim;i++)
        new_shape[i] = s1[i] > s2[i] ? s1[i] : s2[i];
    return new_shape; 
}

Tensor* add_tensors(Tensor *m1, Tensor *m2){
    assert(m1->dim == m2->dim, "dimension of the tensors do not match");
    int dim = m1->dim;
    int *shape = get_max_shape(dim, m1->shape, m2->shape);
    int *indices = init_indices(dim);
    int pos1, pos2, pos;

    Tensor *o = create_tensor(dim, shape);

    do{
        pos1 = get_pos(dim, indices, m1->stride);
        pos2 = get_pos(dim, indices, m2->stride);
        pos = get_pos(dim, indices, o->stride);
        o->data[pos] = m1->data[pos1] + m2->data[pos2];
    }while(!increase_indices(dim, indices, shape));
    free(indices);
    free(shape);
    return o;
}

Tensor* sub_tensors(Tensor *m1, Tensor *m2){
    assert(m1->dim == m2->dim, "dimension of the tensors do not match");
    int dim = m1->dim;
    int *shape = get_max_shape(dim, m1->shape, m2->shape);
    int *indices = init_indices(dim);
    int pos1, pos2, pos;

    Tensor *o = create_tensor(dim, shape);

    do{
        pos1 = get_pos(dim, indices, m1->stride);
        pos2 = get_pos(dim, indices, m2->stride);
        pos = get_pos(dim, indices, o->stride);
        o->data[pos] = m1->data[pos1] - m2->data[pos2];
    }while(!increase_indices(dim, indices, shape));
    free(indices);
    free(shape);
    return o;
}

Tensor* mul_tensors(Tensor *m1, Tensor *m2){
    assert(m1->dim == m2->dim, "dimension of the tensors do not match");
    int dim = m1->dim;
    int *shape = get_max_shape(dim, m1->shape, m2->shape);
    int *indices = init_indices(dim);
    int pos1, pos2, pos;

    Tensor *o = create_tensor(dim, shape);

    do{
        pos1 = get_pos(dim, indices, m1->stride);
        pos2 = get_pos(dim, indices, m2->stride);
        pos = get_pos(dim, indices, o->stride);
        o->data[pos] = m1->data[pos1] * m2->data[pos2];
    }while(!increase_indices(dim, indices, shape));
    free(indices);
    free(shape);
    return o;
}

Tensor* div_tensors(Tensor *m1, Tensor *m2){
    assert(m1->dim == m2->dim, "dimension of the tensors do not match");
    int dim = m1->dim;
    int *shape = get_max_shape(dim, m1->shape, m2->shape);
    int *indices = init_indices(dim);
    int pos1, pos2, pos;

    Tensor *o = create_tensor(dim, shape);

    do{
        pos1 = get_pos(dim, indices, m1->stride);
        pos2 = get_pos(dim, indices, m2->stride);
        pos = get_pos(dim, indices, o->stride);
        o->data[pos] = m1->data[pos1] / m2->data[pos2];
    }while(!increase_indices(dim, indices, shape));
    free(indices);
    free(shape);
    return o;
}

int get_pos_from_running_indices(int* running_indices, int dim, int *idxs, int *stride){
    int s = 0;
    for(int i=0; i<dim; i++)
        s += stride[i] * running_indices[idxs[i]];
   return s; 
}

Tensor *einsum(int num_idx, int *idxs1, Tensor *m1, int *idxs2, Tensor *m2, int dim, int *idxs){
    int *running_indices = init_indices(num_idx);
    int *running_shape = malloc(sizeof(int) * num_idx);
    int *shape = malloc(sizeof(int) * dim);
    int pos1, pos2, pos;
    
    for(int i=0; i<m1->dim; i++)
        running_shape[idxs1[i]] = m1->shape[i];
    for(int i=0; i<m2->dim; i++)
        running_shape[idxs2[i]] = m2->shape[i];
    
    for(int i=0; i<dim; i++)
        shape[i] = running_shape[idxs[i]];

    Tensor *o = create_tensor(dim, shape);
    
    do{
        pos1 = get_pos_from_running_indices(running_indices, m1->dim, idxs1, m1->stride);
        pos2 = get_pos_from_running_indices(running_indices, m2->dim, idxs2, m2->stride);
        pos  = get_pos_from_running_indices(running_indices, o->dim ,idxs, o->stride);
        o->data[pos] += m1->data[pos1] * m2->data[pos2];
    }while(!increase_indices(num_idx, running_indices, running_shape));

    free(running_indices);
    free(running_shape);
    return o;
}

Tensor *einsum2(char *indices_rule, Tensor *m1, Tensor *m2){
    int *idxs1, *idxs2, *idxs;
    int num_idx=-1, dim;
    Tensor *o;
    int idx_map[256] = {0};
    int i;

    i = -1;
    while(indices_rule[++i]){
        if(indices_rule[i] == ' ') continue;
        if(idx_map[indices_rule[i]]) continue;
        idx_map[indices_rule[i]] = ++num_idx;
    }

    idxs1 = malloc(sizeof(int) * m1->dim);
    idxs2 = malloc(sizeof(int) * m2->dim);
    dim = strlen(indices_rule + m1->dim + 1 + m2->dim + 1);
    idxs = malloc(sizeof(int) * dim);

    for(i=0; i<m1->dim; i++)
        idxs1[i] = idx_map[indices_rule[i]] - 1;
    
    i++;
    for(int j=0; j<m2->dim; i++, j++)
        idxs2[j] = idx_map[indices_rule[i]] - 1;

    i++;
    for(int j=0; j<dim; i++, j++)
        idxs[j] = idx_map[indices_rule[i]] - 1;
    
    o = einsum(num_idx, idxs1, m1, idxs2, m2, dim, idxs);
    
    free(idxs1);
    free(idxs2);
    free(idxs);

    return o;
}

Tensor *matmul(Tensor *m1, Tensor *m2){
    assert(m1->dim == m2->dim, "tensor has to have dimension 2 to use matmul");
    assert(m1->shape[1] == m2->shape[0], "shape of tensors no appropiate for matmul");
    return einsum2("ij jk ik", m1, m2);
}

Tensor* apply_fn_to_tensor(Tensor *m, float (*fn)(float)){
    Tensor *o = create_tensor(m->dim, m->shape);
    for(int i = 0; i<m->dim; i++)
        o->data[i] = fn(m->data[i]);
    return o;
}
