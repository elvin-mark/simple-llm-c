#include "llm/core/matrix.h"
#include <stdio.h>

void test_matrix_creation(){
    int shape[3] = {2,3,4};
    Matrix *m = create_matrix(3,shape);
    randomize_matrix(m);
    print_matrix(m);
    free_matrix(m);
}

void test_basic_matrix_operations(){
    int shape[2] = {2,3};
    Matrix *m1 = create_matrix(2,shape);
    Matrix *m2 = create_matrix(2,shape);

    randomize_matrix(m1);
    randomize_matrix(m2);

    print_matrix(m1);
    print_matrix(m2);

    Matrix *o = add_matrices(m1, m2);
    print_matrix(o);

    free_matrix(m1);
    free_matrix(m2);
    free_matrix(o);
}

void test_einsum(){
    int shape[2] = {2,2};
    Matrix *m1 = create_matrix(2,shape);
    Matrix *m2 = create_matrix(2,shape);
    
    randomize_matrix(m1);
    randomize_matrix(m2);

    for(int i=0 ; i<4; i++)
        printf("%.2f ", m1->data[i]);
    printf("\n");

    print_matrix(m1);
    print_matrix(m2);
    
    int idxs1[2] = {0,1};
    int idxs2[2] = {1,2};
    int idxs[2] = {0,2};
    Matrix *o = einsum(3,idxs1, m1, idxs2, m2, 2, idxs); 
    print_matrix(o);

    free_matrix(m1);
    free_matrix(m2);
    free_matrix(o);
}

void test_einsum2(){
    int shape[2] = {2,2};
    Matrix *m1 = create_matrix(2,shape);
    Matrix *m2 = create_matrix(2,shape);
    
    randomize_matrix(m1);
    randomize_matrix(m2);

    for(int i=0 ; i<4; i++)
        printf("%.2f ", m1->data[i]);
    printf("\n");

    print_matrix(m1);
    print_matrix(m2);
    
    Matrix *o = einsum2("ij jk ik\0", m1, m2); 
    print_matrix(o);

    free_matrix(m1);
    free_matrix(m2);
    free_matrix(o);
}

int main(){
    test_matrix_creation();
    test_basic_matrix_operations();
    test_einsum();
    test_einsum2();
    return 0;
}
