#include "ift.h"

/**
 * \file iftGFLIMEncodeMergedLayerBatchV3.c
 * \brief Encode the layer using the merged kernels, bias, and labels from iftGFLIMMergeLayerModels
 * \note It does not decode the layer
 * \note Used when kernel_neighbors, pool_neighbors, and stride_neighbors are "color" (i.e., computed using color information). 
 * \author Isabela Borlido
 */

void processBatch(iftFileSet *fs,
                  int first, int last,
                  iftMatrix *Kmerged,
                  float *bias_merged,
                  iftGFLIMLayer layer,
                  iftGFLIMGraphAttention attention_opt,
                  int *true_labels,
                  char *labels_dir,
                  char *output_dir,
                  char *out_labels_dir,
                  char *model_dir,
                  int layer_id,
                  char *kernel_neighbors,
                  char *pool_neighbors,
                  char *stride_neighbors,
                  iftGFLIMGraphNodeFeature graph_feat,
                  char *orig_images_path,
                  iftGFLIMAggregation aggregation_opt)
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

        neighbors[j] = NULL;
        if (strcmp(kernel_neighbors, "color") == 0 || strcmp(pool_neighbors, "color") == 0)
        {
            sprintf(filename, "%s/NeighborsMatrices/%s_%d.npy", model_dir, basename[j], layer_id-1);
            if(strcmp(stride_neighbors, "color") == 0 && iftFileExists(filename))
            {
                neighbors[j] = iftReadMatrix(filename); 
            }else{
                int num_neighbors = iftMax(layer.kernel_size, layer.pool_size);
                iftImage *orig_image = NULL;
                sprintf(filename, "%s/%s.png", orig_images_path, basename[j]);
                orig_image = iftReadImageByExt(filename);
                neighbors[j] = iftGFLIMConstructGraphRelation(orig_image, label_imgs->val[j], graph_feat, num_neighbors);
            }
        }
    }

    // encode layer
    iftGFLIMEncodeLayerBatchV2(input_graph, Kmerged, layer,
                            bias_merged, attention_opt,
                            true_labels, neighbors,
                            kernel_neighbors, pool_neighbors,
                            aggregation_opt);
    
    /* stride */
    if (strcmp(stride_neighbors, "color") == 0)
    {
        if (layer.pool_stride > 1)
        {
            for (int k = 0; k <= last - first; k++)
            {
                // read the precomputed graph
                iftGFLIMGraph *graph_resized = NULL;
                sprintf(filename, "%s/%s.json", output_dir, basename[k]);
                graph_resized = iftReadGFLIMGraph(filename);

                // read the precomputed superpixel images
                iftImage *label_img_resized = NULL;
                sprintf(filename, "%s/%s.pgm", out_labels_dir, basename[k]);
                label_img_resized = iftReadImageByExt(filename);

                iftGFLIMGraph *graph = input_graph->val[k];
                iftImage *label_img = label_imgs->val[k];

                // resize original image and mask
                float scale[3];
                scale[0] = (float)label_img_resized->xsize / (float)label_img->xsize;
                scale[1] = (float)label_img_resized->ysize / (float)label_img->ysize;
                scale[2] = 1.0;

                // reallocate the node vectors
                for (int j = 0; j < graph_resized->num_nodes; j++)
                {
                    iftFree(graph_resized->nodes[j].feats);
                    graph_resized->nodes[j].feats = (double *)iftAlloc(graph->num_feats, sizeof(double));
                }
                graph_resized->num_feats = graph->num_feats;

                iftGFLIMStridePrecomputed(graph,
                                          graph_resized,
                                          label_img,
                                          label_img_resized,
                                          scale);

                label_img = NULL;
                iftDestroyImage(&label_imgs->val[k]);
                label_imgs->val[k] = label_img_resized;
                label_img_resized = NULL;

                graph = NULL;
                iftDestroyGFLIMGraph(&input_graph->val[k]);
                input_graph->val[k] = graph_resized;
                graph_resized = NULL;
            }
        }
    }
    else
    {
        iftGFLIMStrideBatch(input_graph, layer, 
                    label_imgs, IFT_GFLIM_GRAPHSTRIDE);
    }

    /* save output */
    for (int j = 0; j < batchsize; j++)
    {
        sprintf(filename, "%s/%s.json", output_dir, basename[j]);
        iftWriteGFLIMGraph(input_graph->val[j], filename);
        sprintf(filename, "%s/%s.pgm", out_labels_dir, basename[j]);
        iftWriteImageByExt(label_imgs->val[j], filename);
        iftFree(basename[j]);
        if (strcmp(kernel_neighbors, "color") == 0 || strcmp(pool_neighbors, "color") == 0)
            iftDestroyMatrix(&neighbors[j]);
    }

    iftDestroyGFLIMGraphArray(&input_graph);
    iftDestroyImageArray(&label_imgs);
    iftFree(neighbors);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 12)
        iftError("Usage: iftGFLIMEncodeMergedLayerBatchV3 P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 \n"
                 "P1 : model directory \n"
                 "P2 : architecture of the network (.json) \n"
                 "P3 : layer number (1, 2, 3) \n"
                 "P4 : batch size (1, 8, 16, 32, etc) \n"
                 "P5 : attention. See iftGFLIMGraphAttention at iftGFLIMGraph.h for details \n"
                 "P6 : kernel neighbors (act, color) \n"
                 "P7 : pool neighbors (act, color) \n"
                 "P8 : stride neighbors (act, color) \n"
                 "P9 : graph feature. See iftGFLIMGraphNodeFeature at iftGFLIMGraph.h for details \n"
                 "P10: aggregation strategy. See iftGFLIMAggregation at iftGFLIMGraph.h for details \n"
                 "P11: original images path \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    int batchsize = atoi(argv[4]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[5]);
    char *kernel_neighbors = argv[6];
    char *pool_neighbors = argv[7];
    char *stride_neighbors = argv[8];
    iftGFLIMGraphNodeFeature graph_feat = iftGFLIMGraphGetNodeFeature(argv[9]);
    iftGFLIMAggregation aggregation_opt = iftGFLIMGraphGetAggregation(argv[10]);
    char *orig_images_path = argv[11];
    
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
    sprintf(labels_dir, "%s/labels%d", model_dir, layer - 1);
    if (!iftDirExists(labels_dir))
        iftError("Labels directory does not exist", "main");

    sprintf(out_labels_dir, "%s/labels%d", model_dir, layer);
    iftMakeDir(out_labels_dir);

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
                     arch->layer[layer - 1],
                     attention_opt,
                     true_labels,
                     labels_dir,
                     output_dir,
                     out_labels_dir,
                     model_dir,
                     layer,
                     kernel_neighbors,
                     pool_neighbors,
                     stride_neighbors,
                     graph_feat,
                     orig_images_path,
                     aggregation_opt);
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
                     arch->layer[layer - 1],
                     attention_opt,
                     true_labels,
                     labels_dir,
                     output_dir,
                     out_labels_dir,
                     model_dir,
                     layer,
                     kernel_neighbors,
                     pool_neighbors,
                     stride_neighbors,
                     graph_feat,
                     orig_images_path,
                     aggregation_opt);
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
