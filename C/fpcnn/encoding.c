#include "../libs/dArray.c"
#include "../libs/mathlib.c"
#include "encoding.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>

uint16_t* map_residuals(uint16_t* data, int data_size)
{
    printf("map_residuals\n");
    //Assert (how do i do this in c?, ill make a helper, or maybe its already a thing)
    int n = data_size;
    uint16_t* output = calloc(n, sizeof(uint16_t));

    for(int i = 0; i < n; ++i)
    {
        int x = data[i];

        if(x >= 0)
            x = 2 * x;
        else
            x = -2 * x - 1;
        
        output[i] = x;
    }
    return output;
}

dArray* grc_encode(uint16_t* data, int data_size, int m)
{
    printf("grc_encode\n");
    int M = pow(2, m);

    dArray* code = init_dArray(1);

    for(int idn = 0; idn < data_size; ++idn)
    {   
        int n = data[idn];

        //Replace with bit mask implementation
        int q = n / M;
        int r = n % M;

        //Quotient Code
        bool u[q + 1];
        for(int t = 0; t < q; ++t)
            u[t] = 1;
        u[q] = 0;

        //Remainder Code
        int c = pow(2, m + 1) - M;

        dArray* v = malloc(sizeof(dArray));
        if(r < c) //What's width
            v = dec_to_bin(r);
        else
            v = dec_to_bin(r + c);
        
        for(int t = 0; t < q + 1; ++t)
            insert_dArray(code, u[t]); //I think the max size these reach is 4, can just set init size to 4. one of em goes to 8 actually so idk.
        for(int t = 0; t < v->used; ++t)
            insert_dArray(code, v->array[t]);

    }

    return code;

}

dArray* grc_decode(dArray* code, int m)
{
    printf("grc_decode\n");
    int M = pow(2, m);
    
    dArray* data = init_dArray(0);
    int q = 0;
    int i = 0;

    while(i < code->used)
    {
        if(code->array[i] == 1)
        {
            q += 1;
            i += 1;
        }
        else if(code->array[i] == 0)
        {
            i += 1;
            dArray* v = init_dArray(0);
            for(int t = i; t < i + m; ++t)
                insert_dArray(v, code->array[t]);
            int r = bin_to_dec(v);

            int n = q * M + r;
            insert_dArray(data, n);

            //Reset and move to next code word
            q = 0;
            i += m;
        }
    }
    
    return data;
}

uint16_t* remap_residuals(uint16_t* data, int data_size)
{
    printf("remap_residuals\n");
    int n = data_size;
    uint16_t* output = calloc(data_size, sizeof(uint16_t));

    for(int i = 0; i < n; ++i)
    {
        int x = data[i];

        if(x % 2 == 0)
            x = x / 2;
        else
            x = (x + 1) / -2;
        
        output[i] = x;
    }

    return output;
}

int main(void)
{
    //Create 3D data Block for testing functions:
    int data_size_x = 5;
    int data_size_y = 5;
    int data_size_z = 5;

    uint16_t data[data_size_x][data_size_y][data_size_z];
    

    for(int x = 0; x < data_size_x; ++x)
    {
        printf("\n");
        for(int y = 0; y < data_size_y; ++y)
        {
            printf(" ");
            for(int z = 0; z < data_size_z; ++z)
            {
                //printf(" %d ", (rand() % 5));
                data[x][y][z] = rand() % UINT16_MAX;
                printf("%u ", data[x][y][z]);
            }
        }
    }
    printf("\n");

    //Flatten Test
    uint16_t* data_flat = malloc(sizeof(uint16_t*));
    data_flat = flatten(data_size_x, data_size_y, data_size_z, data);

    for(int i = 0; i < data_size_x * data_size_y * data_size_z; ++i)
       printf("%d ", data_flat[i]);

    // //Entropy test
    // int data_size_x = 10;
    // int data_size_y = 20;
    // int data_size_z = 30;

    // uint16_t data[data_size_x][data_size_y][data_size_z];

    // for(int x = 0; x < data_size_x; ++x)
    //     for(int y = 0; y < data_size_y; ++y)
    //         for(int z = 0; z < data_size_z; ++z)
    //             data[x][y][z] = rand() % 65536;
    
    // double entropy = get_entropy(data, data_size_x, data_size_y, data_size_z);
    // printf("Entropy: %f", entropy);

    
    //remap_residuals test
    // uint16_t data[10] = {6, 2, 9, 5, 0, 4, 7, 3, 8, 1};
    // int data_size = 10;

    // //uint16_t* one = map_residuals(data, data_size);
    
    // dArray* two = malloc(sizeof(dArray));
    // two = grc_encode(data, data_size, 2);
    
    // dArray* three = malloc(sizeof(dArray));
    // three = grc_decode(two, 2);
    
    // for(int i = 0; i < three->used; ++i)
    //     printf("%i ", three->array[i]);

    //uint16_t* four = remap_residuals(three->array, data_size);

    //for(int i = 0; i < data_size; ++i)
    //    printf("%i ", four[i]);





    
    //dec_to_bin and bin_to_dec test
    // int dec = 11;
    // dArray* bin = malloc(sizeof(dArray));
    // bin = dec_to_bin(dec);
    // for(int t = 0; t < bin->size; ++t)
    //     printf("%i\n", bin->array[t]);
    // dec = bin_to_dec(bin);
    // printf("dec: %i\n", dec);
    
    
    
    
    /*
    printf("Main\n");
    int data_size = 10;
    int* data = calloc(data_size, sizeof(int)); //What type is data?

    //Make random vector
    for(int i = 0; i < data_size; i++)
    {
        data[i] = rand() % 25;
        printf("%d ", data[i]);
    }
    
    uint16_t* output = map_residuals(data, data_size);
    for(int i = 0; i < data_size; ++i)
        printf("%i ", output[i]);*/

    return 0;
}

//Comments:
// Implemented a dynamic array.
//      Idk if it'll be faster to overestimate the size of the array, instead of growing as needed.
//      Might be?
// If we know the size of data, then data_size variable unnecessary. 
// Will create a v2 with the GLS_BLAS library.
//What's width in the dec_to_bin function?
