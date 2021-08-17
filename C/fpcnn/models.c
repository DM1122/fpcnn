#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "models.h"

void _valideate_offsets(Hyperparameters* hyperparams);


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

    //Validate Offsets
    _valideate_offsets(&model->hyperparams);

    //Get datetime
    time_t t = time(NULL);
    model->_logdate = *localtime(&t);
    printf("%s", asctime(&model->_logdate));
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