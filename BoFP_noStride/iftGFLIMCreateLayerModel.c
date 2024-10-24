#include "ift.h"

/**
 * @brief Extract patches from feature points of each image, compute
 * filters, biases, truelabel of the filters.
 * @param graph Input graph
 * @param S Graph markers
 * @param layer Layer of the network
 * @param scale Scale of the image
 */
iftDataSet *ExtractGraphNodeFeatures(iftFLIMGraph *graph, /*iftMImage *mimg, */
                                iftLabeledSet *S, iftFLIMLayer layer/*, float *scale*/)
{
    int nsamples = iftLabeledSetSize(S);
    int num_adj = layer.kernel_size[0] * layer.kernel_size[1];
    iftDataSet *Z = iftCreateDataSet(nsamples, num_adj * graph->num_feats);

    iftLabeledSet *Saux = S;
    int incr = 0;
    while (Saux != NULL)
    {
        int l = Saux->label;
        if (l == 0)
        {
            incr = 1;
            break;
        }
        Saux = Saux->next;
    }

    int s = 0;
    Saux = S;
    while (Saux != NULL)
    {
        Z->sample[s].id = Saux->elem;
        Z->sample[s].truelabel = Saux->label + incr; /* vary from 1 to c */
        if (Z->sample[s].truelabel > Z->nclasses)
            Z->nclasses = Z->sample[s].truelabel;

        iftIntArray *neighbors = iftGetKNeighborsFLIMGraph(Saux->elem, graph, 
                                        num_adj, layer.dilation_rate[0] * layer.dilation_rate[1], true);

        int i = 0;
        while (i < neighbors->n)
        {
            for (int b = 0; b < graph->num_feats; b++)
                Z->sample[s].feat[b + i * graph->num_feats] = (float)(graph->nodes[neighbors->val[i]].feats[b]);
            i++;
        }
        iftDestroyIntArray(&neighbors);

        // caso não tenha vizinhos suficientes,
        // completa o kernel com zeros ou com o valor de fill_band_with
        while (i < num_adj)
        {
            for (int b = 0; b < graph->num_feats; b++)
                Z->sample[s].feat[b + i * graph->num_feats] = 0;
            i++;
        }
        
        s++;
        Saux = Saux->next;
    }

    iftSetStatus(Z, IFT_TRAIN);
    iftAddStatus(Z, IFT_SUPERVISED);
    return (Z);
}

void SaveBias(char *basepath, float *bias, int number_of_kernels)
{
    char filename[200];
    FILE *fp;

    sprintf(filename, "%s-bias.txt", basepath);
    fp = fopen(filename, "w");
    fprintf(fp, "%d\n", number_of_kernels);
    for (int k = 0; k < number_of_kernels; k++)
    {
        fprintf(fp, "%f ", bias[k]);
    }
    fclose(fp);
}

void SaveKernelLabels(char *basepath, int *truelabel, int number_of_kernels)
{
    char filename[200];
    FILE *fp;

    sprintf(filename, "%s-labels.txt", basepath);
    fp = fopen(filename, "w");
    fprintf(fp, "%d\n", number_of_kernels);
    for (int k = 0; k < number_of_kernels; k++)
        fprintf(fp, "%d ", truelabel[k]);

    fclose(fp);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 4)
        iftError("Usage: iftCreateLayerModel P1 P2 P3 \n"
                 "P1: input model directory\n"
                 "P2: input network architecture (.json)\n"
                 "P3: input layer for patch definition (1, 2, 3, etc)\n",
                 "main");

    tstart = iftTic();
    
    char *model_dir = argv[1];
    iftFLIMArch *arch = iftReadFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    char img_markers_dir[200], graph_dir[200];
    
    if (layer < 1)
        iftError("Invalid layer number: %d", "main", layer);
    
    sprintf(img_markers_dir, "%s/markers", model_dir);
    if (!iftDirExists(img_markers_dir))
        iftError("Image markers directory does not exist: %s", "main", img_markers_dir);

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(img_markers_dir, "-seeds_graph.txt", 1);
    if (fs->n < 1)
        iftError("Graph markers directory is empty: %s", "main", img_markers_dir);
    
    sprintf(graph_dir, "%s/layer%d", model_dir, layer - 1);
    if (!iftDirExists(graph_dir))
        iftError("Graph directory does not exist: %s", "main", graph_dir);

    char *filename = iftAllocCharArray(512);
    sprintf(filename, "%s/model", model_dir);
    iftMakeDir(filename);

    /* Extract patches from feature points of each image, compute
       filters, biases, truelabel of the filters. */

    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, "-seeds_graph.txt");

        sprintf(filename, "%s/%s%s", graph_dir, basename, ".json");
        if(!iftFileExists(filename)) iftError("Graph file %s does not exist", "main", filename);
        iftFLIMGraph *graph = iftReadFLIMGraph(filename);
        iftLabeledSet *S = iftReadSeedsGraph(fs->files[i]->path);

        /* Extract patch dataset */
        iftDataSet *Z = ExtractGraphNodeFeatures(graph, S, arch->layer[layer - 1]);
        iftNormalizeDataSetByZScoreInPlace(Z, NULL, arch->stdev_factor);
        iftMatrix *kernels = iftCreateMatrix(Z->nsamples, Z->nfeats);
        int *truelabel = iftAllocIntArray(Z->nsamples);

        /* compute filters and their truelabels */
        for (int s = 0, col = 0; s < Z->nsamples; s++)
        {
            iftUnitNorm(Z->sample[s].feat, Z->nfeats);
            truelabel[s] = Z->sample[s].truelabel;
            for (int row = 0; row < Z->nfeats; row++)
            {
                iftMatrixElem(kernels, col, row) = Z->sample[s].feat[row];
            }
            col++;
        }

        /* compute biases and update filters */
        float *bias = NULL;
        bias = iftAllocFloatArray(kernels->ncols);
        for (int col = 0; col < kernels->ncols; col++)
        {
            for (int row = 0; row < Z->nfeats; row++)
            {
                iftMatrixElem(kernels, col, row) =
                    iftMatrixElem(kernels, col, row) / Z->fsp.stdev[row];
                bias[col] -= (Z->fsp.mean[row] * iftMatrixElem(kernels, col, row));
            }
        }

        /* save kernels, biases, and truelabels of the kernels */
        sprintf(filename, "%s/model/%s-conv%d-kernels.npy", model_dir, basename, layer);
        iftWriteMatrix(kernels, filename);
        sprintf(filename, "%s/model/%s-conv%d", model_dir, basename, layer);
        SaveBias(filename, bias, kernels->ncols);
        SaveKernelLabels(filename, truelabel, kernels->ncols);

        iftFree(bias);
        iftFree(truelabel);
        iftDestroyMatrix(&kernels);
        iftFree(basename);
        iftDestroyFLIMGraph(&graph);
        iftDestroyLabeledSet(&S);
        iftDestroyDataSet(&Z);
    }

    iftDestroyFileSet(&fs);
    iftDestroyFLIMArch(&arch);
    iftFree(filename);

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
