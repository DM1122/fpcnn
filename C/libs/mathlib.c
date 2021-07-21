#include <stdio.h>
#include <math.h>

int bin_to_dec(dArray* bin)
{
    int dec = 0;
    for(int i = 0; i < bin->used; ++i)
        dec += pow(2, i) * bin->array[i];
    
    return dec;
}

dArray* dec_to_bin(int x)
{
    dArray* bin = init_dArray(1);

    while(x != 0)
    {
        insert_dArray(bin, x % 2);
        x /= 2;
    }

    return bin;
}