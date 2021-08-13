#include <stdio.h>
#include <math.h>

//Flatten
uint16_t* flatten(int data_size_x, int data_size_y, int data_size_z, uint16_t data[data_size_x][data_size_y][data_size_z])
{
    /*
    Flattens 3D array with specified size into 1 dimensional array.

    Args:
        uint16_t array [data_size_x][data_size_y][data_size_z]
    Returns:
        uint16_t array [data_size_x * data_size_y * data_size_z]
    */
    int i = 0;
    uint16_t* data_flat = malloc(sizeof(uint16_t) * data_size_x * data_size_y * data_size_z);
    for(int x = 0; x < data_size_x; ++x)
        for(int y = 0; y < data_size_y; ++y)
            for(int z = 0; z < data_size_z; ++z)
                data_flat[i++] = data[x][y][z];

    return data_flat;
}


//Get entropy of an information source
double get_entropy(uint16_t* data, int data_size)
{
    /*
    Calculates the entropy of a flattened 3D image.

    Args:
        Unsigned 16 bit 1D integer array, and its size.
    Returns:
        Unsigned double (can probably be changed to float).
    */
    float probs[65536] = {0};
    for(int i = 0; i < data_size; ++i)
        probs[data[i]] += 1;
    
    double entropy = 0;
    for(int i = 0; i < 65536; ++i)
        if(probs[i] != 0)
        {
            printf("%f\n", log2(1 / (probs[i] / data_size)));
            entropy += (probs[i] / data_size) * log2(1 / (probs[i] / data_size));
        }
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