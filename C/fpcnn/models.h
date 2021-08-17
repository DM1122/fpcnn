#include <stdio.h>
#include <time.h>
#include <gsl/gsl_blas.h>



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
    gsl_vector* w;
    double      b;
    gsl_vector* dw;
    double      db;
    
    double m_dw;
    double v_dw;
    double m_db;
    double v_db;
} Layer;

typedef struct
{
    //Optimizer params
    double learning_rate;
    double beta_1;
    double beta_2;
    double epsilon;

} Optimizer;

typedef struct
{
    Hyperparameters hyperparams;
    Optimizer Adam;
    struct tm _logdate;

    //Parameters
    gsl_vector* input_spatial;
    gsl_vector* input_spectral;

    Layer* spatial_extraction;
    Layer* spectral_extraction;
    Layer* merge;

    double output;
    
} FPCNN;


