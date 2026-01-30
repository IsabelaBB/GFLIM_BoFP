#include "ift.h"

/**
 * \file iftGFLIMCreateLayerModel.c
 * \brief Create the model for a given layer using GFLIM-BoFP.
 * It creates kernels, biases, and true labels per training image.
 * \note The kernels, biases, and true labels are saved in the model directory per training image. To merge them use iftGFLIMMergeLayerModel.
 */

// usado quando kernel_neighbors == "color" ou stride_neighbors == "color"
// mas funciona quando sao "act" também

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 9)
        iftError("Usage: iftGFLIMGSCreateLayerModel <P1> <P2> <P3> <P4> <P5> <P6> <P7> <P8> \n"
                 "P1: model directory\n"
                 "P2: network architecture (.json)\n"
                 "P3: layer number (1, 2, 3, etc)\n"
                 "P4: kernel neighbors type (color, act)\n"
                 "P5: stride neighbors type (color, act)\n"
                 "P6: graph feature (MEAN, SUM) \n"
                 "P7: aggregation option \n"
                 "P8: original images path \n"
                 ,
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    char *kernel_neighbors = argv[4];
    char *stride_neighbors = argv[5];
    iftGFLIMGraphNodeFeature graph_feat = iftGFLIMGraphGetNodeFeature(argv[6]);
    iftGFLIMAggregation aggregation_opt = iftGFLIMGraphGetAggregation(argv[7]);
    char *orig_images_path = argv[8];

    char graph_dir[1024], labels_dir[1024];

    if (layer < 1)
        iftError("Invalid layer number: %d", "main", layer);
    if(layer > arch->nlayers)
        iftError("Layer number %d is greater than the number of layers in the network (%d)", "main", layer, arch->nlayers);

    char markers_dir[1024];
    sprintf(markers_dir, "%s/markers", model_dir);
    
    iftFileSet *fs = NULL;
    if(iftDirExists(markers_dir))
      fs = iftLoadFileSetFromDirBySuffix(markers_dir, "-seeds_graph.txt", 1);
    else{
      if(!iftFileExists(markers_dir))
        iftError("Graph markers directory or file does not exist: %s", "main", markers_dir);
      fs = iftLoadFileSetFromCSV(markers_dir, false);
    }
    if (fs == NULL || fs->n < 1)
        iftError("Graph markers directory is empty: %s", "main", markers_dir);

    sprintf(graph_dir, "%s/layer%d", model_dir, layer - 1);
    if (!iftDirExists(graph_dir))
        iftError("Graph directory does not exist: %s", "main", graph_dir);

    sprintf(labels_dir, "%s/labels%d", model_dir, layer - 1);
    if (!iftDirExists(labels_dir))
        iftError("Labels directory does not exist: %s", "main", labels_dir);
    
    char *filename = iftAllocCharArray(1024);
    sprintf(filename, "%s/model", model_dir);
    iftMakeDir(filename);

    for (int i = 0; i < fs->n; i++)
    {
        iftLabeledSet *S = iftReadSeedsGraph(fs->files[i]->path);
        if(S == NULL) continue;

        char *basename = iftFilename(fs->files[i]->path, "-seeds_graph.txt");

        // read the superpixel images for the first and current layer
        iftImage *label_img0 = NULL, *label_img = NULL;
        sprintf(filename, "%s/labels0/%s%s", model_dir, basename, ".pgm");
        if(!iftFileExists(filename)) iftError("Labels file %s does not exist", __func__, filename);
        label_img0 = iftReadImageByExt(filename);
        sprintf(filename, "%s/%s.pgm", labels_dir, basename);
        if(!iftFileExists(filename)) iftError("Labels file %s does not exist", __func__, filename);
        label_img = iftReadImageByExt(filename);

        // read the graph
        sprintf(filename, "%s/%s%s", graph_dir, basename, ".json");
        if(!iftFileExists(filename)) iftError("Graph file %s does not exist", "main", filename);
        iftGFLIMGraph *graph = iftReadGFLIMGraph(filename);

        // stride
        if(strcmp(stride_neighbors, "color") == 0)
        {
            if (label_img->xsize != label_img0->xsize || label_img->ysize != label_img0->ysize)
                iftGFLIMUpdateMarkersFromPrecomputedGraphs(&S, arch->layer[layer - 1], label_img, label_img0);
        }else{
            if (graph->num_nodes != iftMaximumValue(label_img0))
                iftGFLIMUpdateGraphMarkers(S, label_img, label_img0);
        }

        // compute the neighbors matrix based on the original image
        iftMatrix *neighbors = NULL;
        if(strcmp(kernel_neighbors, "color") == 0)
        {
            sprintf(filename, "%s/NeighborsMatrices/%s_%d.npy", model_dir, basename, layer-1);
            if(strcmp(stride_neighbors, "color") == 0 && iftFileExists(filename))
                neighbors = iftReadMatrix(filename); 
            else
            {
                int num_neighbors = iftMax(arch->layer[layer - 1].kernel_size, arch->layer[layer - 1].pool_size);
                iftImage *orig_image = NULL;
                sprintf(filename, "%s/%s.png", orig_images_path, basename);
                orig_image = iftReadImageByExt(filename);
                neighbors = iftGFLIMConstructGraphRelation(orig_image, 
                                                        label_img, 
                                                        graph_feat, 
                                                        num_neighbors);
            }
        }

        // create the kernels, bias and truelabels for the current graph
        float *bias;
        int *truelabel;
        iftMatrix *kernels = iftGFLIMCreateLayerModelV3(graph, S, 
                                                arch->layer[layer - 1], 
                                                arch->stdev_factor, 
                                                &bias, &truelabel, 
                                                neighbors,
                                                kernel_neighbors,
                                                aggregation_opt);
        
        if(neighbors != NULL)
        {
            iftDestroyMatrix(&neighbors);
        }
        
        /* save kernels, biases, and truelabels of the kernels */
        sprintf(filename, "%s/model/%s-conv%d-kernels.npy", model_dir, basename, layer);
        iftWriteMatrix(kernels, filename);
        sprintf(filename, "%s/model/%s-conv%d", model_dir, basename, layer);
        iftGFLIMSaveBias(filename, bias, kernels->ncols);
        iftGFLIMSaveKernelLabels(filename, truelabel, kernels->ncols);
        
        iftFree(bias);
        iftFree(truelabel);
        iftDestroyMatrix(&kernels);
        iftFree(basename);
        iftDestroyGFLIMGraph(&graph);
        iftDestroyLabeledSet(&S);
        iftDestroyImage(&label_img0);
        if(label_img != NULL) iftDestroyImage(&label_img);
    }

    iftDestroyFileSet(&fs);
    iftDestroyGFLIMArch(&arch);
    iftFree(filename);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
