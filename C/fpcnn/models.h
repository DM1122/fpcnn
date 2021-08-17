#include <stdio.h>
#include <time.h>


//Hyperparameters struct [Can also put this into a config file that is read] [might be better for updating parameters]
typedef struct 
{
    //Can set the type to char to lower the number of bits to 8 from 16 for a short
    unsigned short int track_spatial_length;
    unsigned short int track_spatial_width;
    unsigned short int track_spectral_length;
    unsigned short int track_spectral_width;
    unsigned short int track_fusion_length;
    unsigned short int track_fusion_width;
    float lr;

    short int context_offsets_spatial[12][3]; 
    short int context_offsets_spectral[4][3];
} Hyperparameters;

typedef struct
{
    Hyperparameters hyperparams;
    struct tm _logdate;
} FPCNN;

