#include "ift.h"

/**
 * \file iftGFLIMGSEncodeMergedLayer.c
 * \brief Encode a layer using the non-merged kernels, bias, and labels created by iftGFLIMCreateLayerModel
 * \warning This function only encode feature from the training graphs.
 */

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 7)
        iftError("Usage: iftGFLIMGSEncodeLayer <P1> <P2> <P3> <P4> <P5> <P6> \n"
                 "[1] model directory \n"
                 "[2] architecture of the network (.json) \n"
                 "[3] layer number (1, 2, 3) \n"
                 "[4] stride type (none, image-based, graph-based) \n"
                 "[5] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE) \n"
                 "[6] aggregation (CONCAT, MEAN, WMEAN) \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[4]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttentionType(argv[5]);
    iftGFLIMAggregationType aggregation_opt = iftGLIMGraphGetAggregationType(argv[6]);
    
    if (layer < 1)
        iftError("Invalid layer number: %d", "main", layer);

    if (layer > arch->nlayers)
        iftError("Invalid layer number: %d", "main", layer);

    char input_dir[200], labels_dir[200], output_dir[200], out_labels_dir[200];
    char *filename = iftAllocCharArray(512);

    sprintf(input_dir, "%s/layer%d", model_dir, layer - 1);
    if (!iftDirExists(input_dir))
        iftError("Graph directory does not exist", "main");

    if (stride_type != IFT_GFLIM_NOSTRIDE)
    {
        sprintf(labels_dir, "%s/labels%d", model_dir, layer - 1);
        if (!iftDirExists(labels_dir))
            iftError("Labels directory does not exist", "main");
        
        sprintf(out_labels_dir, "%s/labels%d", model_dir, layer);
        iftMakeDir(out_labels_dir);
    }

    sprintf(filename, "%s/markers", model_dir);
    if (!iftDirExists(filename))
        iftError("Markers directory does not exist", "main");
    
    sprintf(output_dir, "%s/layer%d", model_dir, layer);
    iftMakeDir(output_dir);

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(filename, "-seeds_graph.txt", 1);

    /* Encode layer */
    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, "-seeds_graph.txt");

        // read the graph and kernels
        sprintf(filename, "%s/%s.json", input_dir, basename);
        if (!iftFileExists(filename)){
            iftWarning("File %s does not exist", "main", filename);
            iftFree(basename);
            continue;
        }
        iftGFLIMGraph *graph = iftReadFLIMGraph(filename);

        // check if there are kernels created from the graph
        sprintf(filename, "%s/model/%s-conv%d-kernels.npy", model_dir, basename, layer);
        if (!iftFileExists(filename)){
            iftWarning("File %s does not exist", "main", filename);
            iftFree(basename);
            iftDestroyFLIMGraph(&graph);
            continue;
        }
        iftMatrix *K = iftReadMatrix(filename);

        // read the bias
        sprintf(filename, "%s/model/%s-conv%d", model_dir, basename, layer);
        float *bias = iftGFLIMLoadBias(filename);
        if(bias == NULL) iftError("Bias is NULL", "main");

        // read the superpixel image
        iftImage *input_labels = NULL, *output_labels = NULL;
        if (stride_type != IFT_GFLIM_NOSTRIDE)
        {
            sprintf(filename, "%s/%s.pgm", labels_dir, basename);
            input_labels = iftReadImageByExt(filename);
        }

        // read the kernel labels
        int *true_labels = NULL;
        if (attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE) // several attention mechanisms use the kernel labels
        {
            sprintf(filename, "%s/model/%s-conv%d-labels.txt", model_dir, basename, layer);
            if (!iftFileExists(filename))
                iftError("File %s does not exist", "main", filename);
            int nkernels = 0;
            true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
        }

        // encode the layer
        iftGFLIMGraph *output_graph = iftGFLIMEncodeLayer(graph, K, bias, arch->layer[layer - 1], stride_type, input_labels, &output_labels, attention_opt, true_labels, aggregation_opt, false);
        if(true_labels != NULL) iftFree(true_labels);

        /* save superpixel labels and graph (layer output) */
        if (stride_type != IFT_GFLIM_NOSTRIDE)
        {
            sprintf(filename, "%s/%s.pgm", out_labels_dir, basename);
            iftWriteImageByExt(output_labels, filename);
            iftDestroyImage(&output_labels);
            iftDestroyImage(&input_labels);
        }
        sprintf(filename, "%s/%s.json", output_dir, basename);
        iftWriteFLIMGraph(output_graph, filename);
        iftDestroyFLIMGraph(&output_graph);

        iftFree(basename);
        iftDestroyMatrix(&K);
        iftFree(bias);
        iftDestroyFLIMGraph(&graph);
    }

    iftFree(filename);
    iftDestroyFileSet(&fs);
    iftDestroyGFLIMArch(&arch);
    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
