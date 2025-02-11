#ifndef NN_ACT_FUN_H
#define NN_ACT_FUN_H

#include <math.h>

float sigmoid(float x){
    return 1.0 / (1.0 + exp(-x));
}

float tanh2(float x){
    float a = exp(x);
    float b = 1.0 / a;
    return (a - b) / (a + b);
}

float relu(float x){
    return x > 0? x : 0.0;
}

#endif
