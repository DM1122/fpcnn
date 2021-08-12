#include <stdio.h>
#include <math.h>

//Get entropy of an information source
double get_entropy(uint16_t*** data, int data_size_x, int data_size_y, int data_size_z)
{
    /*
    Input: Unsigned 16 bit 3D integer arrays
    Output: Unsigned double (can probably be changed to float)
    */
    float probs[65536] = {0};
    for(int x = 0; x < data_size_x; ++x)
        for(int y = 0; y < data_size_y; ++y)
            for(int z = 0; z < data_size_z; ++z)
                probs[data[x][y][z]] += 1;
    
    double entropy = 0
    for(int x = 0; x < data_size_x; ++x)
        for(int y = 0; y < data_size_y; ++y)
            for(int z = 0; z < data_size_z; ++z)
                entropy += probs[x][y][z] * log2l(1 / probs[x][y][z]);
    
    return entropy;
}

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

int main(void)
{

}