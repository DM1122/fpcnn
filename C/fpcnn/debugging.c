#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    for(int i = 0; i<5; i++)
        printf(" %d ", rand() % 5);
   
    printf("%i ", UINT16_MAX);
    return 0;
}