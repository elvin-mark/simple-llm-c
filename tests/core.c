#include "llm/core/matrix.h"

void matrix_creation(){
    int shape[3] = {2,3,4};
    Matrix *m = create_matrix(3,shape);
    randomize_matrix(m);
    print_matrix(m);
    free_matrix(m);
}

void basic_matrix_operations(){
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

int main(){
    matrix_creation();
    basic_matrix_operations();
    return 0;
}
