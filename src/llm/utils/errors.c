#include "llm/utils/errors.h"
#include <stdio.h>
#include <stdlib.h>

void assert(int cond, char* message){
    if(!cond){
        printf("error: %s", message);
        exit(1);
    }
}
