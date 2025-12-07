#include "ift.h"

/**
 * \file iftGFLIMGSEncodeMergedLayerBatch.c
 * \brief Encode the layer using the merged kernels, bias, and labels
 * \note It does not decode the layer
 */

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 8 && argc != 9)
        iftError("Usage: iftGFLIMGSEncodeMergedLayerBatch <P1> <P2> <P3> <P4> <P5> <P6> <P7> [<P8>] \n"
                 "[1] model directory \n"
                 "[2] architecture of the network (.json) \n"
                 "[3] layer number (1, 2, 3) \n"
                 "[4] batch size (1, 8, 16, 32, etc) \n"
                 "[5] stride type (none, image-based, graph-based) \n"
                 "[6] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE) \n"
                 "[7] aggregation (CONCAT, MEAN, WMEAN) \n"
                 "[8] reduce output channels based on attention values (1/0). It is necessary a compatible attention mechanism \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    int batchsize = atoi(argv[4]);
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[5]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[6]);
    iftGFLIMAggregation aggregation_opt = iftGLIMGraphGetAggregation(argv[7]);
    bool kernel_removal = false;
    if (argc == 9)
        kernel_removal = atoi(argv[8]);

    if (layer < 1)
        iftError("Invalid layer number: %d", "main", layer);

    if (layer > arch->nlayers)
        iftError("Invalid layer number: %d", "main", layer);

    if (batchsize < 1)
        iftError("Invalid batch size: %d", "main", batchsize);

    char input_dir[1024], labels_dir[1024], output_dir[1024], out_labels_dir[1024];
    char *filename = iftAllocCharArray(1024);

    sprintf(input_dir, "%s/layer%d", model_dir, layer - 1);
    if (!iftDirExists(input_dir))
        iftError("Graph directory does not exist", "main");

    sprintf(output_dir, "%s/layer%d", model_dir, layer);
    iftMakeDir(output_dir);

    if(stride_type != IFT_GFLIM_NOSTRIDE)
    {
        sprintf(labels_dir, "%s/labels%d", model_dir, layer - 1);
        if (!iftDirExists(labels_dir))
            iftError("Labels directory does not exist", "main");

        sprintf(out_labels_dir, "%s/labels%d", model_dir, layer);
        iftMakeDir(out_labels_dir);
    }

    iftFileSet *fs = NULL;

    if(iftDirExists(input_dir))
        fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", 1);
    else{
        if(!iftFileExists(input_dir))
            iftError("Graph directory or file does not exist: %s", "main", input_dir);
        fs = iftLoadFileSetFromCSV(input_dir, false);
    }

    if(fs == NULL || fs->n < 1)
        iftError("Graph directory or file is empty or does not exist: %s", "main", input_dir);

    // load the merged model
    iftMatrix *Kmerged = NULL;
    float *bias_merged = NULL;
    int *true_labels = NULL;

    // read kernels and bias
    sprintf(filename, "%s/model/conv%d", model_dir, layer);
    iftGFLIMLoadMergedModel(filename, &Kmerged, &bias_merged);

    // read the kernel labels
    if(attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE)
    {
        sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer);
        if (!iftFileExists(filename))
            iftError("File %s does not exist", "main", filename);
        int nkernels = 0;
        true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
    }

    batchsize = iftMin(batchsize, fs->n);
    int nbatches = fs->n / batchsize;
    int nremaining_graphs = fs->n - nbatches * batchsize;

    /* Encode layer with merged model */
    printf("\n");
    for (int batch = 0; batch < nbatches; batch++)
    {
        char *basename[batchsize];

        printf("Processing batch %d of %d batches\r", batch + 1, nbatches);
        fflush(stdout);

        /* read graphs */
        iftGFLIMGraphArray *input_graph = iftCreateGFLIMGraphArray(batchsize);
        iftImageArray *label_imgs = iftCreateImageArray(batchsize);
        for (int j = 0; j < batchsize; j++)
        {
            basename[j] = iftFilename(fs->files[batch * batchsize + j]->path, ".json");
            input_graph->val[j] = iftReadGFLIMGraph(fs->files[batch * batchsize + j]->path);
            sprintf(filename, "%s/%s.pgm", labels_dir, basename[j]);
            label_imgs->val[j] = iftReadImageByExt(filename);
        }

        // encode layer
        iftGFLIMEncodeLayerBatch(input_graph, Kmerged, arch->layer[layer - 1], 
                                label_imgs, bias_merged, stride_type, attention_opt, 
                                true_labels, aggregation_opt, kernel_removal);

        /* save output */
        for (int j = 0; j < batchsize; j++)
        {
            sprintf(filename, "%s/%s.json", output_dir, basename[j]);
            iftWriteGFLIMGraph(input_graph->val[j], filename);
            if (stride_type != IFT_GFLIM_NOSTRIDE)
            {
                sprintf(filename, "%s/%s.pgm", out_labels_dir, basename[j]);
                iftWriteImageByExt(label_imgs->val[j], filename);
            }
            iftFree(basename[j]);
        }
        iftDestroyGFLIMGraphArray(&input_graph);
        iftDestroyImageArray(&label_imgs);
    }

    int first = fs->n - nremaining_graphs, last = fs->n - 1;
    if (first <= last)
    {
        char *basename[last - first + 1];

        printf("Processing remaining %d images\r", last - first + 1);
        fflush(stdout);

        /* read graphs */
        iftGFLIMGraphArray *input_graph = iftCreateGFLIMGraphArray(last - first + 1);
        iftImageArray *label_imgs = iftCreateImageArray(last - first + 1);
        for (int j = 0; j < (last - first + 1); j++)
        {
            basename[j] = iftFilename(fs->files[first + j]->path, ".json");
            input_graph->val[j] = iftReadGFLIMGraph(fs->files[first + j]->path);
            sprintf(filename, "%s/%s.pgm", labels_dir, basename[j]);
            label_imgs->val[j] = iftReadImageByExt(filename);
        }

        // encode layer
        iftGFLIMEncodeLayerBatch(input_graph, Kmerged, arch->layer[layer - 1], 
                                label_imgs, bias_merged, stride_type, attention_opt, 
                                true_labels, aggregation_opt, kernel_removal);

        /* write activations */
        for (int j = 0; j < (last - first + 1); j++)
        {
            sprintf(filename, "%s/%s.json", output_dir, basename[j]);
            iftWriteGFLIMGraph(input_graph->val[j], filename);
            if (stride_type != IFT_GFLIM_NOSTRIDE)
            {
                sprintf(filename, "%s/%s.pgm", out_labels_dir, basename[j]);
                iftWriteImageByExt(label_imgs->val[j], filename);
            }
            iftFree(basename[j]);
        }
        iftDestroyGFLIMGraphArray(&input_graph);
        iftDestroyImageArray(&label_imgs);
    }
    printf("\n");

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
