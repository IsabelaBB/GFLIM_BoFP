#include "ift.h"

/**
 * \file iftGFLIMGSEncodeMergedLayerBatch.c
 * \brief Encode the layer using the merged kernels, bias, and labels
 * \note It does not decode the layer
 */

void processBatch(iftFileSet *fs,
                  int first, int last,
                  iftMatrix *Kmerged,
                  float *bias_merged,
                  iftGFLIMLayer layer,
                  int layer_id,
                  iftGFLIMGraphAttention attention_opt,
                  int *true_labels,
                  char *orig_images_path,
                  iftGFLIMGraphNodeFeature node_feat,
                  char *labels_dir,
                  char *output_dir,
                  char *out_labels_dir)
{
    int batchsize = last - first + 1;
    char *basename[batchsize];
    char filename[1024];

    /* read graphs and superpixel labels */
    iftGFLIMGraphArray *input_graph = iftCreateGFLIMGraphArray(batchsize);
    iftImageArray *label_imgs = iftCreateImageArray(batchsize);
    iftMatrix **neighbors = (iftMatrix **)iftAlloc(batchsize, sizeof(iftMatrix *));
    for (int j = 0; j < batchsize; j++)
    {
        basename[j] = iftFilename(fs->files[first + j]->path, ".json");
        input_graph->val[j] = iftReadGFLIMGraph(fs->files[first + j]->path);
        sprintf(filename, "%s/%s.pgm", labels_dir, basename[j]);
        label_imgs->val[j] = iftReadImageByExt(filename);

        sprintf(filename, "%s/%s.png", orig_images_path, basename[j]);  
        iftImage *orig_img = iftReadImageByExt(filename);
        neighbors[j] = iftGFLIMConstructGraphRelation(orig_img, label_imgs->val[j], node_feat, 
                                                        iftMax(layer.kernel_size, layer.pool_size));
        iftDestroyImage(&orig_img);
    }

    // encode layer
    iftGFLIMEncodeLayerBatchV2(input_graph, Kmerged, layer, 
                            bias_merged, attention_opt, 
                            true_labels, neighbors,
                            kernel_neighbors,
                            pool_neighbors);

    /* stride */
    iftGFLIMStrideBatch(input_graph, layer, label_imgs, IFT_GFLIM_GRAPHSTRIDE);

    /* save output */
    for (int j = 0; j < batchsize; j++)
    {
        sprintf(filename, "%s/%s.json", output_dir, basename[j]);
        iftWriteGFLIMGraph(input_graph->val[j], filename);
        if(layer_id > 1)
        {
            sprintf(filename, "%s/%s.pgm", out_labels_dir, basename[j]);
            iftWriteImageByExt(label_imgs->val[j], filename);
        }
        iftFree(basename[j]);
        iftDestroyMatrix(&neighbors[j]);
    }
    iftDestroyGFLIMGraphArray(&input_graph);
    iftDestroyImageArray(&label_imgs);
    iftFree(neighbors);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 8)
        iftError("Usage: iftGFLIMGSEncodeMergedLayerBatch <P1> <P2> <P3> <P4> <P5> <P6> <P7> \n"
                 "[1] model directory \n"
                 "[2] architecture of the network (.json) \n"
                 "[3] layer number (1, 2, 3) \n"
                 "[4] batch size (1, 8, 16, 32, etc) \n"
                 "[5] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE) \n"
                 "[6] graph node feature (MEAN, SUM) \n"
                 "[7] original images path (used to compute node neighbors) --- it must have at least the training images \n"
                 ,
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    int batchsize = atoi(argv[4]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[5]);
    iftGFLIMGraphNodeFeature node_feat = iftGFLIMGraphGetNodeFeature(argv[6]);
    char *orig_images_path = argv[7];

    if (layer < 1)
        iftError("Invalid layer number: %d", "main", layer);

    if (layer > arch->nlayers)
        iftError("Invalid layer number: %d", "main", layer);

    if (batchsize < 1)
        iftError("Invalid batch size: %d", "main", batchsize);

    char input_dir[1024], labels_dir[1024], output_dir[1024], out_labels_dir[1024];
    char *filename = iftAllocCharArray(1024);

    // set graphs input directory
    sprintf(input_dir, "%s/layer%d", model_dir, layer - 1);
    if (!iftDirExists(input_dir))
        iftError("Graph directory does not exist", "main");

    sprintf(output_dir, "%s/layer%d", model_dir, layer);
    iftMakeDir(output_dir);

    // set labels directory
    if(layer > 1)
    {
        sprintf(labels_dir, "%s/labels%d", model_dir, layer - 1);
        if (!iftDirExists(labels_dir))
            iftError("Labels directory does not exist", "main");

        sprintf(out_labels_dir, "%s/labels%d", model_dir, layer);
        iftMakeDir(out_labels_dir);
    }else{
        sprintf(labels_dir, "%s/labels0", model_dir);
        if (!iftDirExists(labels_dir))
            iftError("Labels directory does not exist", "main");
    }

    iftFileSet *fs = NULL;

    // iterate over the training graphs
    if (iftDirExists(input_dir))
        fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", 1);
    else
    {
        if (!iftFileExists(input_dir))
            iftError("Graph directory or file does not exist: %s", "main", input_dir);
        fs = iftLoadFileSetFromCSV(input_dir, false);
    }
    if (fs == NULL || fs->n < 1)
        iftError("Graph directory or file is empty or does not exist: %s", "main", input_dir);

    // load the merged model
    iftMatrix *Kmerged = NULL;
    float *bias_merged = NULL;
    int *true_labels = NULL;

    // read kernels and bias
    sprintf(filename, "%s/model/conv%d", model_dir, layer);
    iftGFLIMLoadMergedModel(filename, &Kmerged, &bias_merged);

    // read the kernel labels
    if (attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE)
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
        printf("Processing batch %d of %d batches\r", batch + 1, nbatches);
        fflush(stdout);

        processBatch(fs,
                  batch * batchsize, 
                  batch * batchsize + batchsize - 1,
                  Kmerged,
                  bias_merged,
                  arch->layer[layer - 1], layer,
                  attention_opt,
                  true_labels,
                  orig_images_path,
                  node_feat,
                  labels_dir,
                  output_dir,
                  out_labels_dir);
    }

    int first = fs->n - nremaining_graphs, last = fs->n - 1;
    if (first <= last)
    {
        printf("Processing remaining %d graphs\r", last - first + 1);
        fflush(stdout);
        processBatch(fs,
                  first, last,
                  Kmerged,
                  bias_merged,
                  arch->layer[layer - 1], layer,
                  attention_opt,
                  true_labels,
                  orig_images_path,
                  node_feat,
                  labels_dir,
                  output_dir,
                  out_labels_dir);
    }
    printf("\n");

    if (true_labels != NULL)
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
