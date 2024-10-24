#include "ift.h"

iftVoxel GetVoxelCoord(int xsize, int ysize, int zsize, int p)
{
  iftVoxel u;
  div_t res1 = div(p, xsize * ysize);
  div_t res2 = div(res1.rem, xsize);

  u.x = res2.rem;
  u.y = res2.quot;
  u.z = res1.quot;
  u.t = 0;
  return (u);
}

/**
 * @brief Extract patches from feature points of each image, compute
 * filters, biases, truelabel of the filters.
 * @param graph Input graph
 * @param S Graph markers
 * @param layer Layer of the network
 * @param scale Scale of the image
 */
iftDataSet *ExtractGraphNodeFeatures(iftFLIMGraph *graph, /*iftMImage *mimg, */
                                     iftLabeledSet *S, iftFLIMLayer layer /*, float *scale*/)
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

iftLabeledSet *imageMarkersToGraphMarkers(char *labels_file, char *seeds_file)
{
    iftImage *label_img = iftReadImageByExt(labels_file);
    iftLabeledSet *S = NULL;
    iftLabeledSet *graph_seeds = NULL;
    float scale[3];

    if (!iftIs3DImage(label_img))
        S = iftReadLabeledSet(seeds_file, 2);
    else
        S = iftReadLabeledSet(seeds_file, 3);

    FILE *fp = fopen(seeds_file, "r");
    int nseeds;
    int w, h, d;
    if (!iftIs3DImage(label_img))
    { /* 2D Image */
        if (fscanf(fp, "%d %d %d", &nseeds, &w, &h) != 3)
            iftError("Reading error", "iftReadSeeds");
        d = 1;
    }
    else
    {
        if (fscanf(fp, "%d %d %d %d", &nseeds, &w, &h, &d) != 4)
            iftError("Reading error", "iftReadSeeds");
    }
    scale[0] = (float)w / (float)label_img->xsize;
    scale[1] = (float)h / (float)label_img->ysize;
    scale[2] = (float)d / (float)label_img->zsize;

    fclose(fp);

    int num_nodes = iftMaximumValue(label_img);
    if (num_nodes < 1)
        iftError("Number of nodes must be positive: %d", __func__, num_nodes);

    int *foreground_count = (int *)calloc(num_nodes, sizeof(int));
    int *background_count = (int *)calloc(num_nodes, sizeof(int));
    iftVoxel u;
    iftLabeledSet *tmp = S;
    while (tmp != NULL)
    {
        int l = tmp->label;
        int p = tmp->elem;
        
        if (iftIs3DImage(label_img))
        {
            u = GetVoxelCoord((int)(label_img->xsize * scale[0]),
                              (int)(label_img->ysize * scale[1]),
                              (int)(label_img->zsize * scale[2]), p);
        }
        else
        {
            u = GetVoxelCoord((int)(label_img->xsize * scale[0]),
                              (int)(label_img->ysize * scale[1]), 1, p);
        }
        u.x = u.x / scale[0];
        u.y = u.y / scale[1];
        u.z = u.z / scale[2];

        p = iftGetVoxelIndex(label_img, u);

        // define whether the superpixel is foreground or background
        if (l == 1)
            foreground_count[label_img->val[p] - 1]++;
        else
            background_count[label_img->val[p] - 1]++;
        
        tmp = tmp->next;
    }

    iftDestroyImage(&label_img);
    iftDestroyLabeledSet(&S);

    for (int p = 0; p < num_nodes; p++)
    {
        if (foreground_count[p] > 0 || background_count[p] > 0)
        {
            if (foreground_count[p] > background_count[p])
                iftUnionLabeledSetElemMarkerAndHandicap(&graph_seeds, p, 1, 1, 0);
            else
                iftUnionLabeledSetElemMarkerAndHandicap(&graph_seeds, p, 0, 0, 0);
        }
    }
    free(foreground_count);
    free(background_count);
    return graph_seeds;
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 5)
        iftError("Usage: iftCreateLayerModel P1 P2 P3 P4 \n"
                 "P1: input model directory\n"
                 "P2: input network architecture (.json)\n"
                 "P3: input layer for patch definition (1, 2, 3, etc)\n"
                 "P4: image markers (-seeds.txt)\n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftFLIMArch *arch = iftReadFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    char *img_markers_dir = argv[4];
    char graph_dir[200], labels_dir[200];

    if (layer < 1)
        iftError("Invalid layer number: %d", "main", layer);
    if(layer > arch->nlayers)
        iftError("Layer number %d is greater than the number of layers in the network (%d)", "main", layer, arch->nlayers);

    if (!iftDirExists(img_markers_dir))
        iftError("Image markers directory does not exist: %s", "main", img_markers_dir);

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(img_markers_dir, "-seeds.txt", 1);
    if (fs->n < 1)
        iftError("Image markers directory is empty: %s", "main", img_markers_dir);

    sprintf(graph_dir, "%s/layer%d", model_dir, layer - 1);
    if (!iftDirExists(graph_dir))
        iftError("Graph directory does not exist: %s", "main", graph_dir);

    sprintf(labels_dir, "%s/labels%d", model_dir, layer - 1);
    if (!iftDirExists(labels_dir))
        iftError("Labels directory does not exist: %s", "main", labels_dir);

    char *filename = iftAllocCharArray(512);
    sprintf(filename, "%s/model", model_dir);
    iftMakeDir(filename);

    /* Extract patches from feature points of each image, compute
       filters, biases, truelabel of the filters. */
    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s%s", graph_dir, basename, ".json");
        if(!iftFileExists(filename)) iftError("Graph file %s does not exist", "main", filename);
        iftFLIMGraph *graph = iftReadFLIMGraph(filename);
        sprintf(filename, "%s/%s%s", labels_dir, basename, ".pgm");
        if(!iftFileExists(filename))
            iftError("Labels file %s does not exist", __func__, filename);
        iftLabeledSet *S = imageMarkersToGraphMarkers(filename, fs->files[i]->path);
        
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
