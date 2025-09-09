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
    iftFLIMArch *arch = iftReadFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    int batchsize = atoi(argv[4]);
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[5]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttentionType(argv[6]);
    iftGFLIMAggregationType aggregation_opt = iftGLIMGraphGetAggregationType(argv[7]);
    bool kernel_removal = false;
    if (argc == 9)
        kernel_removal = atoi(argv[8]);

    if (layer < 1)
        iftError("Invalid layer number: %d", "main", layer);

    if (layer > arch->nlayers)
        iftError("Invalid layer number: %d", "main", layer);

    if (batchsize < 1)
        iftError("Invalid batch size: %d", "main", batchsize);

    char input_dir[200], labels_dir[200], output_dir[200], out_labels_dir[200];
    char *filename = iftAllocCharArray(512);

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

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", 1);

    // load the merged model
    iftMatrix *Kmerged = NULL;
    float *bias_merged = NULL;
    int *kernel_labels = NULL;
    if(attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE)
    {
        sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer);
        if (!iftFileExists(filename))
            iftError("File %s does not exist", "main", filename);
        int nkernels = 0;
        kernel_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
    }
    sprintf(filename, "%s/model/conv%d", model_dir, layer);
    iftGFLIMLoadMergedModel(filename, &Kmerged, &bias_merged);

    /* Encode layer with merged model */

    for (int i = 0; i < fs->n; i += batchsize)
    {
        int first = i;
        int last = iftMin(i + batchsize - 1, fs->n - 1);
        char **basename = (char **)calloc(last - first + 1, sizeof(char *));

        /* read graphs */
        iftGFLIMGraphArray *input_graph = iftCreateEmptyFLIMGraphArray(last - first + 1);
        iftImageArray *label_imgs = iftCreateImageArray(last - first + 1);
        for (int j = 0; j < (last - first + 1); j++)
        {
            char ext[10];
            sprintf(ext, "%s", iftFileExt(fs->files[i + j]->path));
            basename[j] = iftFilename(fs->files[i + j]->path, ext);
            input_graph->val[j] = iftReadFLIMGraph(fs->files[i + j]->path);
            sprintf(filename, "%s/%s.pgn", labels_dir, basename[j]);
            label_imgs->val[j] = iftReadImageByExt(filename);
        }

        // encode layer
        iftGFLIMEncodeLayerBatch_v2(input_graph, Kmerged, arch->layer[layer - 1], label_imgs, bias_merged, stride_type, attention_opt, kernel_labels, aggregation_opt, kernel_removal);

        /* write activations */
        for (int j = first; j <= last; j++)
        {
            sprintf(filename, "%s/%s.json", output_dir, basename[j - first]);
            iftWriteFLIMGraph(input_graph->val[j - first], filename);
            sprintf(filename, "%s/%s.pgm", out_labels_dir, basename[j - first]);
            iftWriteImageByExt(label_imgs->val[j - first], filename);
            iftFree(basename[j - first]);
        }
        iftFree(basename);
        iftDestroyFLIMGraphArray(&input_graph);
        iftDestroyImageArray(&label_imgs);
    }

    int nremaining_graphs = fs->n % batchsize;
    int first = fs->n - nremaining_graphs, last = fs->n - 1;
    if (first <= last)
    {
        char **basename = (char **)calloc(last - first + 1, sizeof(char *));
        /* read graphs */
        iftGFLIMGraphArray *input_graph = iftCreateEmptyFLIMGraphArray(last - first + 1);
        iftImageArray *label_imgs = iftCreateImageArray(last - first + 1);
        for (int j = 0; j < (last - first + 1); j++)
        {
            char ext[10];
            sprintf(ext, "%s", iftFileExt(fs->files[first + j]->path));
            basename[j] = iftFilename(fs->files[first + j]->path, ext);
            input_graph->val[j] = iftReadFLIMGraph(fs->files[first + j]->path);
            sprintf(filename, "%s/%s.pgm", labels_dir, basename[j]);
            label_imgs->val[j] = iftReadImageByExt(filename);
        }

        // encode layer
        iftGFLIMEncodeLayerBatch_v2(input_graph, Kmerged, arch->layer[layer - 1], label_imgs, bias_merged, stride_type, attention_opt, kernel_labels, aggregation_opt, kernel_removal);

        /* write activations */
        for (int j = first; j <= last; j++)
        {
            sprintf(filename, "%s/%s.json", output_dir, basename[j - first]);
            iftWriteFLIMGraph(input_graph->val[j - first], filename);
            sprintf(filename, "%s/%s.pgm", out_labels_dir, basename[j - first]);
            iftWriteImageByExt(label_imgs->val[j - first], filename);
            iftFree(basename[j - first]);
        }
        iftFree(basename);
        iftDestroyFLIMGraphArray(&input_graph);
        iftDestroyImageArray(&label_imgs);
    }

    if(kernel_labels != NULL) iftFree(kernel_labels);
    iftDestroyMatrix(&Kmerged);
    iftFree(bias_merged);
    iftFree(filename);
    iftDestroyFileSet(&fs);
    iftDestroyFLIMArch(&arch);
    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
