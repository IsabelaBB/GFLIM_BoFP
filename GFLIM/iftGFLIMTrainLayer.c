#include "ift.h"

/**
 * \file iftGFLIMTrainLayer.c
 * \brief Train a layer model using GFLIM.
 * \details Let X+1 be the layer you want to train. If X > 0, this program already creates the folders needed. 
 *      Otherwise, the model directory must contain the folders: 
 *      1) layerX and labelsX with the training graphs and their corresponding superpixel images, respectively. 
 *          Use iftCreateGFLIMGraph to create them from images or iftCreateGFLIMGraphFromLabels if you already have the superpixel images.
 *      2) img_seeds with txt files containing the number of coordinates, image size, and the xy coordinates of the labeled pixels along with their marker label. 
 *          Use FLIMBuilder in demo/Qt/FLIMBuilder or iDSIF from https://github.com/IsabelaBB/iDISF to create these files by manually adding markers on your training images. 
 *          You can also use CreateSeedsFromAllPix.py to create a seed file with all the pixels as seeds. The class label in this case will be in the filename 
 *          (i.e., image name "00001_0030.png" will have class label 1 for all pixels)
 * \note You cannot run this program for layer 2 if you have not trained layer 1 yet.
 * \author Isabela Borlido 
 */

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 4)
        iftError("Usage: iftCreateLayerModel P1 P2 P3 \n"
                 "P1: model directory\n"
                 "P2: network architecture (.json)\n"
                 "P3: layer number \n",
                 "main");

    tstart = iftTic();
    
    char *model_dir = argv[1];
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    char layer_input[200], layer_output[200], seeds_dir[200];

    if(!iftDirExists(model_dir)){
        iftError("Model directory missing or does not exist: %s", "main", model_dir);
    }

    sprintf(layer_input, "%s/layer%d", model_dir, layer-1);
    if(!iftDirExists(layer_input)){
        iftError("Graphs directory missing or does not exist: %s", "main", layer_input);
    }

    sprintf(layer_output, "%s/layer%d", model_dir, layer);
    if(iftDirExists(layer_output)){
        iftWarning("Deleting the existing layer directory %s.", "main_train", layer_output);
        iftRemoveDir(layer_output);
    }
    iftMakeDir(layer_output);

    sprintf(seeds_dir, "%s/img_seeds", model_dir);
    if(!iftDirExists(seeds_dir)){
        iftError("Seeds directory missing or does not exist: %s", "main", seeds_dir);
    }
    
    if(layer > 1){
        char file_kernels[255], file_mean[255], file_stdev[255];
        sprintf(file_kernels, "%s/model/conv%d-kernels.npy", model_dir, layer-1);
        sprintf(file_mean, "%s/model/conv%d-mean.txt", model_dir, layer-1);
        sprintf(file_stdev, "%s/model/conv%d-stdev.txt", model_dir, layer-1);
        
        if (!iftFileExists(file_kernels) && !iftFileExists(file_mean) && !iftFileExists(file_stdev))
        {
            if (!iftFileExists(file_kernels))
                printf("File %s not found \n", file_kernels);
            if (!iftFileExists(file_mean))
                printf("File %s not found \n", file_mean);
            if (!iftFileExists(file_stdev))
                printf("File %s not found \n", file_stdev);
            iftError("Files not found", __func__);
        }
    }

    iftGFLIMArch *save_arch = (iftGFLIMArch *)iftAlloc(1, sizeof(iftGFLIMArch));
    save_arch->nlayers = 1;
    save_arch->stdev_factor = arch->stdev_factor;
    save_arch->apply_intrinsic_atrous = arch->apply_intrinsic_atrous;
    save_arch->layer = (iftGFLIMLayer *)iftAlloc(1, sizeof(iftGFLIMLayer));

    save_arch->layer[0].kernel_size = arch->layer[layer - 1].kernel_size;
    save_arch->layer[0].dilation_rate = arch->layer[layer - 1].dilation_rate;
    save_arch->layer[0].pool_size = arch->layer[layer - 1].pool_size;
    
    save_arch->layer[0].nkernels_per_image = arch->layer[layer - 1].nkernels_per_image;
    save_arch->layer[0].nkernels_per_marker = arch->layer[layer - 1].nkernels_per_marker;
    save_arch->layer[0].noutput_channels = arch->layer[layer - 1].noutput_channels;
    save_arch->layer[0].relu = arch->layer[layer - 1].relu;
    
    save_arch->layer[0].pool_type = iftAllocCharArray(strlen(arch->layer[layer - 1].pool_type) + 1);
    strcpy(save_arch->layer[0].pool_type, arch->layer[layer - 1].pool_type);
    save_arch->layer[0].pool_stride = arch->layer[layer - 1].pool_stride;
    iftDestroyGFLIMArch(&arch);
    
    char param_dir[255];
    sprintf(param_dir, "%s/model", model_dir);
    iftGFLIMTrainLayer(layer_input, seeds_dir, param_dir, layer, save_arch, layer_output);

    iftDestroyGFLIMArch(&save_arch);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
