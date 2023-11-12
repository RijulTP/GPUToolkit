#include <stdio.h>
int a[] = {1,2,3,4,5,6};
int b[] = {7,8,9,10,11,12};
int c[6];

int main() {
    int N = 6;  // Number of elements

    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    
    for (int i = 0; i < N; i++) {
        printf("c[%d] = %d\n", i, c[i]);
    }
    
    return 0;
}


