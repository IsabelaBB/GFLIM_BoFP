#include "ift.h"

/**
 * \file iftGFLIMGSEncodeMergedLayer.c
 * \brief Encode a layer using the merged model created by iftGFLIMMergeLayerModels.
 * Used to create the first half of the GFLIM-BoFP-U model.
 * \note The second half of the model is created by iftGFLIMCreateLayerModelUDecoder.
 */


iftGFLIMGraph *iftGFLIMUEncodeLayer(iftGFLIMGraph *graph,
                                  iftMatrix *Kernels, float *bias,
                                  iftGFLIMLayer layer, 
                                  iftGFLIMGraphAttention attention_opt, 
                                  int *kernel_labels,
                                  iftGFLIMAggregation aggregation_opt,
                                  bool reduce)
{

    iftMImage *activ = iftGFLIMConvLayer(graph, Kernels, layer.kernel_size, attention_opt, kernel_labels, aggregation_opt, reduce);

    iftMImage *activ_relu = NULL;
    if (layer.relu)
        activ_relu = iftMReLU(activ, bias, true);
    else
    {
        if (bias != NULL)
            activ_relu = iftGFLIMSumMImgToFloatArray(activ, bias, true);
        else
            activ_relu = activ;
    }
    activ = NULL;

#ifdef PRINT_VALS
    printf("activ_relu: %d %d %d %ld \n", activ_relu->xsize, activ_relu->ysize, activ_relu->zsize, activ_relu->m);
    for (int i = 0; i < activ_relu->n; i++)
    {
        for (int j = 0; j < activ_relu->m; j++)
            printf("%f ", activ_relu->val[i][j]);
        printf("\n");
    }
    printf("\n");
#endif

    iftGFLIMGraph *output_graph = iftMImageToGFLIMGraph(activ_relu, graph);
    iftDestroyMImage(&activ_relu);
    
    return output_graph;
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 8 && argc != 7)
        iftError("Usage: iftGFLIMGSEncodeMergedLayer <P1> <P2> <P3> <P4> <P5> [<P6>] \n"
                 "[1] model directory \n"
                 "[2] architecture of the network (.json) \n"
                 "[3] layer number (1, 2, 3) \n"
                 "[4] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE, ...) \n"
                 "[5] aggregation (CONCAT, MEAN, WMEAN) \n"
                 "[6] reduce output channels based on attention values (1/0). It is necessary a compatible attention mechanism \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[4]);
    iftGFLIMAggregation aggregation_opt = iftGLIMGraphGetAggregation(argv[5]);
    bool kernel_removal = false;
    if (argc == 7)
        kernel_removal = atoi(argv[6]);

    if (layer < 1)
        iftError("Invalid layer number: %d", "main", layer);

    if (layer > arch->nlayers)
        iftError("Invalid layer number: %d", "main", layer);

    char input_dir[1024], labels_dir[1024], output_dir[1024], out_labels_dir[1024], noPool_dir[1024];
    char *filename = iftAllocCharArray(1024);

    sprintf(input_dir, "%s/layer%d", model_dir, layer - 1);
    if (!iftDirExists(input_dir))
        iftError("Graph directory does not exist", "main");

    sprintf(output_dir, "%s/layer%d", model_dir, layer);
    iftMakeDir(output_dir);

    sprintf(labels_dir, "%s/labels%d", model_dir, layer - 1);
    if (!iftDirExists(labels_dir))
        iftError("Labels directory does not exist", "main");
    sprintf(out_labels_dir, "%s/labels%d", model_dir, layer);
    iftMakeDir(out_labels_dir);

    if(arch->layer[layer - 1].pool_stride != 1){
        sprintf(noPool_dir, "%s/noPool%d", model_dir, layer-1);
        iftMakeDir(noPool_dir);
    }

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", 1);

    iftMatrix *Kmerged = NULL;
    float *bias_merged = NULL;

    // read kernels and bias
    sprintf(filename, "%s/model/conv%d", model_dir, layer);
    iftGFLIMLoadMergedModel(filename, &Kmerged, &bias_merged);

    // read the kernel labels
    int *true_labels = NULL;
    sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer);
    if (!iftFileExists(filename))
        iftError("File %s does not exist", "main", filename);
    int nkernels = 0;
    true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);

    /* Encode layer with merged model */

    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, ".json");

        // read the graph and superpixel image
        iftGFLIMGraph *graph = iftReadGFLIMGraph(fs->files[i]->path);
        iftImage *input_labels = NULL;
        sprintf(filename, "%s/%s.pgm", labels_dir, basename);
        input_labels = iftReadImageByExt(filename);

        /* conv + att + bias */
        iftGFLIMGraph *conv_graph = iftGFLIMUEncodeLayer(graph, Kmerged, bias_merged, arch->layer[layer - 1], attention_opt, true_labels, aggregation_opt, kernel_removal);
        iftDestroyGFLIMGraph(&graph);

        // save no pool graph to be used in skip connection
        if(arch->layer[layer - 1].pool_stride != 1){
            sprintf(filename, "%s/%s.json", noPool_dir, basename);
            iftWriteGFLIMGraph(conv_graph, filename);
        }

        // pooling
        if(strcmp(arch->layer[layer - 1].pool_type, "no_pool") != 0){
            iftGFLIMPoolingLayer(conv_graph, arch->layer[layer - 1].pool_type, arch->layer[layer - 1].pool_size, 1);
        }
            
        /* stride model */
        if(arch->layer[layer - 1].pool_stride != 1)
        {
            iftGFLIMGraph *output_graph = NULL;
            iftImage *output_labels = NULL;
            int nodes_map[conv_graph->num_nodes];
            output_labels = iftCreateImage(input_labels->xsize, input_labels->ysize, input_labels->zsize);
            output_graph = iftGFLIMGraphStride(conv_graph, arch->layer[layer - 1].pool_stride, nodes_map);
            for (int j = 0; j < input_labels->n; j++)
                output_labels->val[j] = nodes_map[input_labels->val[j] - 1] + 1;

            iftDestroyGFLIMGraph(&conv_graph);
            conv_graph = output_graph;
            iftDestroyImage(&input_labels);
            input_labels = output_labels;
        }

        /* save output */
        sprintf(filename, "%s/%s.pgm", out_labels_dir, basename);
        iftWriteImageByExt(input_labels, filename);
        iftDestroyImage(&input_labels);

        sprintf(filename, "%s/%s.json", output_dir, basename);
        iftWriteGFLIMGraph(conv_graph, filename);
        iftDestroyGFLIMGraph(&conv_graph);
        iftFree(basename);
    }

    iftFree(true_labels);
    iftDestroyMatrix(&Kmerged);
    iftFree(bias_merged);
    iftFree(filename);
    iftDestroyFileSet(&fs);
    iftDestroyGFLIMArch(&arch);
    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
