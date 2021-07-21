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
            insert_dArray(code, u[t]);
        for(int t = 0; t < v->used; ++t)
            insert_dArray(code, v->array[t]);

    }

    return code;

}

dArray* grc_decode(dArray* code, int m)
{
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
    // //remap_residuals test
    // uint16_t data[100] = {83, 84, 96, 66, 73, 99, 26, 71, 45, 76, 55, 85, 24, 91, 61, 9, 33, 7, 43, 69, 4, 1, 34, 44, 40, 14, 18, 97, 42, 80, 15, 13, 16, 23, 50, 22, 59, 29, 31, 5, 51, 6, 98, 100, 46, 58, 94, 11, 37, 65, 86, 92, 8, 72, 2, 63, 88, 35, 32, 60, 25, 81, 3, 30, 57, 74, 78, 82, 70, 68, 20, 75, 19, 52, 27, 28, 49, 62, 41, 10, 21, 79, 17, 38, 36, 39, 77, 48, 56, 89, 53, 64, 12, 90, 67, 95, 47, 87, 54, 93};
    // int data_size = 100;

    // uint16_t* one = map_residuals(data, data_size);
    
    // dArray* two = malloc(sizeof(dArray));
    // two = grc_encode(one, data_size, 2);
    
    // dArray* three = malloc(sizeof(dArray));
    // three = grc_decode(two, 2);
    
    // for(int i = 0; i < three->used; ++i)
    //     printf("%i ", three->array[i]);

    // uint16_t* four = remap_residuals(three->array, data_size);

    // for(int i = 0; i < data_size; ++i)
    //     printf("%i ", four[i]);





    
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
