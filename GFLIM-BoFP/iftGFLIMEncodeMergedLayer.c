#include "ift.h"

/**
 * \file iftGFLIMGSEncodeMergedLayer.c
 * \brief Encode a layer using the merged model created by iftGFLIMMergeLayerModels.
 */

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 8 && argc != 7)
        iftError("Usage: iftGFLIMGSEncodeMergedLayer <P1> <P2> <P3> <P4> <P5> <P6> [<P7>] \n"
                 "[1] model directory \n"
                 "[2] architecture of the network (.json) \n"
                 "[3] layer number (1, 2, 3) \n"
                 "[4] stride type (none, image-based, graph-based) \n"
                 "[5] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE, ...) \n"
                 "[6] aggregation (CONCAT, MEAN, WMEAN) \n"
                 "[7] reduce output channels based on attention values (1/0). It is necessary a compatible attention mechanism \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[4]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[5]);
    iftGFLIMAggregation aggregation_opt = iftGLIMGraphGetAggregation(argv[6]);
    bool kernel_removal = false;
    if (argc == 8)
        kernel_removal = atoi(argv[7]);
    
    if (layer < 1)
        iftError("Invalid layer number: %d", "main", layer);

    if (layer > arch->nlayers)
        iftError("Invalid layer number: %d", "main", layer);

    char input_dir[1024], labels_dir[1024], output_dir[1024], out_labels_dir[1024];
    char *filename = iftAllocCharArray(1024);

    sprintf(input_dir, "%s/layer%d", model_dir, layer - 1);
    if (!iftDirExists(input_dir))
        iftError("Graph directory does not exist", "main");

    sprintf(output_dir, "%s/layer%d", model_dir, layer);
    iftMakeDir(output_dir);

    if (stride_type != IFT_GFLIM_NOSTRIDE)
    {
        sprintf(labels_dir, "%s/labels%d", model_dir, layer - 1);
        if (!iftDirExists(labels_dir))
            iftError("Labels directory does not exist", "main");
        sprintf(out_labels_dir, "%s/labels%d", model_dir, layer);
        iftMakeDir(out_labels_dir);
    }

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", 1);

    iftMatrix *Kmerged = NULL;
    float *bias_merged = NULL;

    // read kernels and bias
    sprintf(filename, "%s/model/conv%d", model_dir, layer);
    iftGFLIMLoadMergedModel(filename, &Kmerged, &bias_merged);

    // read the kernel labels
    int *true_labels = NULL;
    if (attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE) // several attention mechanisms use the kernel labels
    {
        sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer);
        if (!iftFileExists(filename))
            iftError("File %s does not exist", "main", filename);
        int nkernels = 0;
        true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
    }
    
    /* Encode layer with merged model */

    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, ".json");

        // read the graph and superpixel image
        iftGFLIMGraph *graph = iftReadGFLIMGraph(fs->files[i]->path);
        iftImage *input_labels = NULL, *output_labels = NULL;
        if (stride_type != IFT_GFLIM_NOSTRIDE)
        {
            sprintf(filename, "%s/%s.pgm", labels_dir, basename);
            input_labels = iftReadImageByExt(filename);
        }
        
        // encode layer
        iftGFLIMGraph *output_graph = iftGFLIMEncodeLayer(graph, Kmerged, bias_merged, arch->layer[layer - 1], stride_type, input_labels, &output_labels, attention_opt, true_labels, aggregation_opt, kernel_removal);

        iftDestroyImage(&input_labels);
        iftDestroyGFLIMGraph(&graph);

        /* save output */
        if (stride_type != IFT_GFLIM_NOSTRIDE)
        {
            sprintf(filename, "%s/%s.pgm", out_labels_dir, basename);
            iftWriteImageByExt(output_labels, filename);
            iftDestroyImage(&output_labels);
        }

        sprintf(filename, "%s/%s.json", output_dir, basename);
        iftWriteGFLIMGraph(output_graph, filename);
        iftDestroyGFLIMGraph(&output_graph);
        iftFree(basename);
    }

    if(true_labels != NULL) iftFree(true_labels);
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
