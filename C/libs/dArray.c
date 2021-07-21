#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct
{
    //Can add conditional (i think?) for which unint?_t
    uint16_t*  array; //8 bc i think thats how big a point has to be? unsure
    uint32_t  size;
    uint32_t  used; 
} dArray;

int deinit_dArray(dArray* array)
{
    free(array->array);
    free(array);

    return 0;
}

dArray* init_dArray(uint32_t size)
{
    //Add check that size >= 1
    if(size < 1)
    {
        printf("WARNING: size provided < 1. Defaulting to size = 1\n");
        size = 1;
    }

    dArray* array = malloc(sizeof(dArray));
    array->array = calloc(size, sizeof(uint32_t));
    array->size  = size;
    array->used  = 0;

    return array;
}

void insert_dArray(dArray* array, uint32_t element)
{
    if(array->used + 1 >= array->size)
    {   
        array->array = realloc(array->array, array->size * 2);
        array->size *= 2;
        printf("Resizing Array; Current size %i\n", array->size);
    }
    
    array->array[array->used] = element;
    array->used++;

}

/*int main(void)
{   
    dArray* array = init_dArray(10);
    
    for(int x = 0; x < 25; ++x)
    {
        insert_dArray(array, rand() % 25);
        printf("%i /", array->used);
        printf("%i: ", array->size);
        printf("%i\n", array->array[x]);
    }
    printf("\n");
    for(int x = 0; x < 25; ++x)
    {
        printf("%i ", array->array[x]);
    }
    deinit_dArray(array);

    return 0;
}*/