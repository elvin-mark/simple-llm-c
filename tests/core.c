#include "llm/core/tensor.h"
#include <stdio.h>

void test_tensor_creation(){
    int shape[3] = {2,3,4};
    Tensor *m = create_tensor(3,shape);
    randomize_tensor(m);
    print_tensor(m);
    free_tensor(m);
}

void test_basic_tensor_operations(){
    int shape[2] = {2,3};
    Tensor *m1 = create_tensor(2,shape);
    Tensor *m2 = create_tensor(2,shape);

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

void test_einsum(){
    int shape[2] = {2,2};
    Tensor *m1 = create_tensor(2,shape);
    Tensor *m2 = create_tensor(2,shape);
    
    randomize_tensor(m1);
    randomize_tensor(m2);

    for(int i=0 ; i<4; i++)
        printf("%.2f ", m1->data[i]);
    printf("\n");

    print_tensor(m1);
    print_tensor(m2);
    
    int idxs1[2] = {0,1};
    int idxs2[2] = {1,2};
    int idxs[2] = {0,2};
    Tensor *o = einsum(3,idxs1, m1, idxs2, m2, 2, idxs); 
    print_tensor(o);

    free_tensor(m1);
    free_tensor(m2);
    free_tensor(o);
}

void test_einsum2(){
    int shape[2] = {2,2};
    Tensor *m1 = create_tensor(2,shape);
    Tensor *m2 = create_tensor(2,shape);
    
    randomize_tensor(m1);
    randomize_tensor(m2);

    for(int i=0 ; i<4; i++)
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

int main(){
    test_tensor_creation();
    test_basic_tensor_operations();
    test_einsum();
    test_einsum2();
    return 0;
}
