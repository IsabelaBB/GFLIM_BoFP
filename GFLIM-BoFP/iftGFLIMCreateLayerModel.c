#include "ift.h"

/**
 * \file iftGFLIMCreateLayerModel.c
 * \brief Create the model for a given layer using GFLIM-BoFP.
 * It creates kernels, biases, and true labels per training image.
 * \note The kernels, biases, and true labels are saved in the model directory per training image. To merge them use iftGFLIMMergeLayerModel.
 */

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 7)
        iftError("Usage: iftGFLIMGSCreateLayerModel <P1> <P2> <P3> <P4> <P5> <P6> \n"
                 "P1: model directory\n"
                 "P2: network architecture (.json)\n"
                 "P3: layer number (1, 2, 3, etc)\n"
                 "P4: stride type (none, image-based, graph-based) \n"
                 "P5: attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE,...) \n"
                 "P6: aggregation strategy (CONCAT, MEAN, WMEAN) \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[4]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[5]);
    iftGFLIMAggregation aggregation_opt = iftGFLIMGraphGetAggregation(argv[6]);

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

    if(stride_type != IFT_GFLIM_NOSTRIDE)
        sprintf(labels_dir, "%s/labels%d", model_dir, layer - 1);
    else 
        sprintf(labels_dir, "%s/labels0", model_dir);
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

        // read the graph
        sprintf(filename, "%s/%s%s", graph_dir, basename, ".json");
        if(!iftFileExists(filename)) iftError("Graph file %s does not exist", "main", filename);
        iftGFLIMGraph *graph = iftReadGFLIMGraph(filename);
        
        // read the superpixel image for the first and current layer
        iftImage *label_img0 = NULL, *label_img = NULL;
        sprintf(filename, "%s/labels0/%s%s", model_dir, basename, ".pgm");
        if(!iftFileExists(filename)) iftError("Labels file %s does not exist", __func__, filename);
        label_img0 = iftReadImageByExt(filename);

        if(stride_type != IFT_GFLIM_NOSTRIDE)
        {
            sprintf(filename, "%s/%s.pgm", labels_dir, basename);
            if(!iftFileExists(filename)) iftError("Labels file %s does not exist", __func__, filename);
            label_img = iftReadImageByExt(filename);
        }

        // create the kernels, bias and truelabels for the current graph
        float *bias;
        int *truelabel;
        //iftMatrix *kernels = iftGFLIMCreateReducedLayerModel(graph, S, label_img0, label_img, arch->layer[layer - 1], arch->stdev_factor, stride_type, &bias, &truelabel, attention_opt, aggregation_opt);
        iftMatrix *kernels = iftGFLIMCreateLayerModel(graph, S, label_img0, label_img, arch->layer[layer - 1], arch->stdev_factor, stride_type, &bias, &truelabel, attention_opt, aggregation_opt);

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
