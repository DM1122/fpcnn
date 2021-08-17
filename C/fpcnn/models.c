#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "models.h"

void _valideate_offsets(Hyperparameters* hyperparams);
void _build_model(FPCNN* model);


void __init__(FPCNN* model)
{
    //Assigning Hyperparameter Values
    Hyperparameters params;
    params.track_spatial_length  = 1;
    params.track_spatial_width   = 5;
    params.track_spectral_length = 1;
    params.track_spectral_width  = 5;
    params.track_fusion_length   = 1;
    params.track_fusion_width    = 5;
    params.lr                    = 0.01;

    short int context_offsets_spatial[12][3] =  {{-1, 0, 0}, //Is this a constant sized array, or do the dimensions change?
                                                {-1, -1, 0},
                                                {0, -1, 0},
                                                {1, -1, 0},
                                                {-1, 0, -1},
                                                {-1, -1, -1},
                                                {0, -1, -1},
                                                {1, -1, -1},
                                                {-1, 0, -2},
                                                {-1, -1, -2},
                                                {0, -1, -2},
                                                {1, -1, -2}};
                                                //{1, 0, 1}};
    short int context_offsets_spectral[4][3] =   {{0, 0, -1}, {0, 0, -2}, {0, 0, -3}, {0, 0, -4}};

    memcpy(params.context_offsets_spatial, context_offsets_spatial, sizeof(context_offsets_spatial));
    memcpy(params.context_offsets_spectral, context_offsets_spectral, sizeof(context_offsets_spectral));
    model->hyperparams = params;

    //Validate offsets
    _valideate_offsets(&model->hyperparams);

    //Get datetime
    time_t t = time(NULL);
    model->_logdate = *localtime(&t);
    printf("%s", asctime(&model->_logdate));

    //Model instantiation
    _build_model(model);

}

void _build_layer(Layer* layer, size_t size)
{
    Layer temp;

    temp.w  = gsl_vector_calloc(size);
    temp.dw = gsl_vector_calloc(size);

    temp.b  = 0; //Needs to be Glorot Uniform initialized (i think???).
    temp.db = 0;

    temp.m_dw = 0;
    temp.v_dw = 0;
    temp.m_db = 0;
    temp.v_db = 0;

    layer = &temp;
}

void _build_model(FPCNN* model)
{
    //Need to implement Glorot Uniform initialization (what are the limits and seed)
    model->input_spatial  = gsl_vector_calloc(12); //12 because thats the len(context_offsets_spatial). Might change idk
    model->input_spectral = gsl_vector_calloc(4); //4 because thats the len(context_offsets_spectral). Might change idk

    _build_layer(model->spatial_extraction, model->hyperparams.track_spatial_width);
    _build_layer(model->spectral_extraction , model->hyperparams.track_spectral_width);
    _build_layer(model->merge, model->hyperparams.track_fusion_width);
    
    model->output = 0; //Needs to be Glorot Uniform initialized.
    
    //Optimizer
    model->Adam.learning_rate = model->hyperparams.lr;
    model->Adam.beta_1 = 0.9;
    model->Adam.beta_2 = 0.999;
    model->Adam.epsilon = 1e-7;
       
}

void _optimizer_Adam(FPCNN* model)
{
    //https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
    //Following the above link.

    //Get dw 
    //get db
    //run update
    //calculate loss


}

//@staticmethod?
void _valideate_offsets(Hyperparameters* hyperparams)
{
    /*
    Check to make sure offset selection does not access unseen voxels.

    Args:
        hyperparams (Hyperparameters*): pointer to struct of initialized hyperparameters.

    Raises:
        ValueError: In the case a context offset in invalid, ValueError is raised.
    */

    for(int i = 0; i < 12; ++i)
    {
        if((hyperparams->context_offsets_spatial[i][0] >= 0  &&
            hyperparams->context_offsets_spatial[i][1] == 0  && 
            hyperparams->context_offsets_spatial[i][2] >= 0) || 
           (hyperparams->context_offsets_spatial[i][1] > 0   && 
            hyperparams->context_offsets_spatial[i][2] >= 0))
           {
               printf("ValueError: Offset {%hu, %hu, %hu} is invalid. Attempted to access future voxel.\n", hyperparams->context_offsets_spatial[i][0],hyperparams->context_offsets_spatial[i][1],hyperparams->context_offsets_spatial[i][2]);
               exit(0);
           }
    }

    for(int i = 0; i < 4; ++i)
    {
        if((hyperparams->context_offsets_spectral[i][0] >= 0  &&
            hyperparams->context_offsets_spectral[i][1] == 0  && 
            hyperparams->context_offsets_spectral[i][2] >= 0) || 
           (hyperparams->context_offsets_spectral[i][1] > 0   && 
            hyperparams->context_offsets_spectral[i][2] >= 0))
           {
               printf("ValueError: Offset {%hu, %hu, %hu} is invalid. Attempted to access future voxel.\n", hyperparams->context_offsets_spectral[i][0],hyperparams->context_offsets_spectral[i][1],hyperparams->context_offsets_spectral[i][2]);
               exit(0);
           }
    }
}

int main(void)
{
    //models.h test
    FPCNN* model = malloc(sizeof(FPCNN)); 
    Hyperparameters* hyperparams = malloc(sizeof(Hyperparameters));
    __init__(model);

}