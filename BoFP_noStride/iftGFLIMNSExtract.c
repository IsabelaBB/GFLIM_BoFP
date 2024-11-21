#include "ift.h"

typedef struct iftFLIMGraphArray{
  iftFLIMGraph **val;
  int n;
} iftFLIMGraphArray;

iftFLIMGraphArray *CreateEmptyFLIMGraphArray(int n)
{
  iftFLIMGraphArray *array = (iftFLIMGraphArray *)calloc(1, sizeof(iftFLIMGraphArray));
  array->val = (iftFLIMGraph **)calloc(n, sizeof(iftFLIMGraph *));
  array->n = n;
  return array;
}

int batchSizeCPU(iftFLIMArch *arch, int input_graph_nnodes, int input_graph_nfeatures, int num_edges, int max_out_channel, int nLayers)
{
    float freeMemoryMb, percMemory = 0.85;

    freeMemoryMb = (float)iftGetFreePhysicalSystemMemory() / 1024.0 / 1024.0;
    //printf("CPU: %.0fGB of free memory.\n", freeMemoryMb / 1024.0);

    float nMbytesPerDouble = sizeof(double) / 1024.0 / 1024.0;
    float nMbytesPerNode = sizeof(iftFLIMGraphNode) / 1024.0 / 1024.0;
    float nMbytesPerGraph = sizeof(iftFLIMGraph) / 1024.0 / 1024.0;
    float nMbytesPerSet = sizeof(iftSet) / 1024.0 / 1024.0;
    float ninput_channels = input_graph_nfeatures;
    float nMbytesInputGraphNode = nMbytesPerNode + ninput_channels * nMbytesPerDouble;
    float nMbytesInputGraph = nMbytesPerGraph + nMbytesInputGraphNode * input_graph_nnodes + num_edges * nMbytesPerSet;
    float max_requiredMemory = 0.0;
    for (int l = 0; l < nLayers; l++)
    {
        int KS2 = 1;
        if (arch->layer[l].kernel_size[2] != 0)
            KS2 = arch->layer[l].kernel_size[2];
        float kernelsize = arch->layer[l].kernel_size[0] * arch->layer[l].kernel_size[1] * KS2;
        float nMbytesKernels = max_out_channel * kernelsize * ninput_channels * nMbytesPerDouble;
        float nMbytesOutputGraphNode = nMbytesPerNode + max_out_channel * nMbytesPerDouble;
        float nMbytesOutputGraph = nMbytesPerGraph + nMbytesOutputGraphNode * input_graph_nnodes + num_edges * nMbytesPerSet;
        float requiredMemory = nMbytesKernels + nMbytesOutputGraph + nMbytesInputGraph;

        if (requiredMemory > max_requiredMemory)
        {
            max_requiredMemory = requiredMemory;
        }
        ninput_channels = max_out_channel;
        nMbytesInputGraph = nMbytesOutputGraph;  // / arch->layer[l].pool_stride;
        input_graph_nnodes = input_graph_nnodes; // / arch->layer[l].pool_stride;
    }
    return (int)floor(percMemory * freeMemoryMb / max_requiredMemory);
}

void iftDestroyLIMGraphArray(iftFLIMGraphArray **arr)
{
    iftFLIMGraphArray *aux = *arr;

    if (aux != NULL) {
        for(int i = 0; i < aux->n; i++) {
            if(aux->val[i] != NULL)
                iftDestroyFLIMGraph(&aux->val[i]);
        }
        iftFree(aux->val);
        iftFree(aux);
        *arr = NULL;
    }
}

int *LoadKernelLabels(char *filename, int *number_of_kernels)
{
  FILE *fp;
  int *truelabels;

  fp = fopen(filename, "r");
  fscanf(fp, "%d", number_of_kernels);
  truelabels = iftAllocIntArray(*number_of_kernels);
  for (int k = 0; k < *number_of_kernels; k++)
  {
    fscanf(fp, "%d ", &truelabels[k]);
  }
  fclose(fp);

  return (truelabels);
}

void LoadMergedModel(char *basepath, iftMatrix **K, float **bias, int *nkernels)
{
  char filename[200];
  FILE *fp;
  
  sprintf(filename, "%s-kernels.npy", basepath);
  *K = iftReadMatrix(filename);
  sprintf(filename, "%s-bias.txt", basepath);
  fp = fopen(filename, "r");
  fscanf(fp, "%d", nkernels);
  *bias = iftAllocFloatArray(*nkernels);
  for (int k = 0; k < *nkernels; k++) {
    fscanf(fp, "%f ", &((*bias)[k]));
  }  
  fclose(fp);
}

float *AdaptiveWeights2(iftMImage *mimg)
{
  float *weight = iftAllocFloatArray(mimg->m);
  float *mean = iftAllocFloatArray(mimg->m);
  float *maxval = iftAllocFloatArray(mimg->m);

  /* Normalize activations */
  for (int b = 0; b < mimg->m; b++)
  {
    maxval[b] = iftMMaximumValue(mimg, b);
  }

#pragma omp parallel for shared(mimg, maxval)
  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      mimg->val[p][b] /= maxval[b];
    }
  }
  iftFree(maxval);

  /* Estimate the mean activation per channel */

  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      mean[b] += mimg->val[p][b];
    }
  }

  for (int b = 0; b < mimg->m; b++)
  {
    mean[b] /= mimg->n;
  }

  /* Estimate mean and standard deviation of the above mean activations */

  float mu = 0.0, stdev = 0.0;
  for (int b = 0; b < mimg->m; b++)
  {
    mu += mean[b];
  }
  mu /= mimg->m;

  for (int b = 0; b < mimg->m; b++)
  {
    stdev += powf(mean[b] - mu, 2);
  }
  stdev = sqrtf(stdev / mimg->m) * sqrtf(stdev / mimg->m);

  /* Estimate the weight of each channel */

  for (int b = 0; b < mimg->m; b++)
  {
    if (mean[b] <= mu - stdev)
    {
      weight[b] = +1.0;
    }
    else
    {
      if (mean[b] >= mu + stdev)
      {
        weight[b] = -1.0;
      }
      else
      {
        weight[b] = 0.0;
      }
    }
  }

  iftFree(mean);

  return (weight);
}

iftFImage *FixedWeightDecoder(iftMImage *mimg, float *weight)
{
  iftFImage *salie = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);

  /* decode layer */

#pragma omp parallel for shared(salie, mimg, weight)
  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      salie->val[p] += mimg->val[p][b] * weight[b];
    }
    if (salie->val[p] < 0)
      salie->val[p] = 0; /* ReLU (Sigmoid?) */
  }
  return (salie);
}

iftFImage *AdaptiveDecoder2(iftMImage *mimg, float adj_radius, float *channel_label)
{
  iftFImage *salie = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);
  float *weight = iftAllocFloatArray(mimg->m);
  float *maxval = iftAllocFloatArray(mimg->m);
  int nobj = 0, nbkg = 0;

  /* Normalize activations per channel */

  for (int b = 0; b < mimg->m; b++)
  {
    maxval[b] = iftMMaximumValue(mimg, b);
    if (channel_label[b] > 0)
    { /* object channel */
      nobj++;
    }
    else
    { /* background channel */
      nbkg++;
    }
  }

  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      if (maxval[b] > IFT_EPSILON)
        mimg->val[p][b] = mimg->val[p][b] / maxval[b];
    }
  }
  iftFree(maxval);

  /* Estimate saliency of each pixel */

  iftAdjRel *A = NULL;
  if (iftIs3DMImage(mimg))
    A = iftSpheric(adj_radius);
  else
    A = iftCircular(adj_radius);

  for (int p = 0; p < mimg->n; p++)
  {

    /* estimate mean and stdev of activations across channels */

    float mean[2] = {0.0, 0.0}, var[2] = {0.0, 0.0};
    iftVoxel u = iftMGetVoxelCoord(mimg, p);

    for (int i = 0; i < A->n; i++)
    {
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);

      if (iftMValidVoxel(mimg, v))
      {
        int q = iftMGetVoxelIndex(mimg, v);

        for (int b = 0; b < mimg->m; b++)
        {
          if (channel_label[b] > 0)
          { /* object channel */
            mean[0] += mimg->val[q][b];
          }
          else
          { /* background channel */
            mean[1] += mimg->val[q][b];
          }
        }
      }
    }
    mean[0] /= (nobj * A->n);
    mean[1] /= (nbkg * A->n);

    for (int i = 0; i < A->n; i++)
    {
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);

      if (iftMValidVoxel(mimg, v))
      {
        int q = iftMGetVoxelIndex(mimg, v);

        for (int b = 0; b < mimg->m; b++)
        {
          if (channel_label[b] > 0)
          { /* object channel */
            var[0] += powf(mimg->val[q][b] - mean[0], 2);
          }
          else
          { /* background channel */
            var[1] += powf(mimg->val[q][b] - mean[1], 2);
          }
        }
      }
    }
    var[0] = var[0] / (nobj * A->n);
    var[1] = var[1] / (nbkg * A->n);

    /* It is expected two Gaussian distributions G0 and G1 of
       activations around mean[0] and mean[1]. We may expect that
       object pixels generate mean[0] >> mean[1] and the other way
       around for background pixels. We may use the the probability
       values of these distributions to estimate weights. */

    float P0, P1;
    for (int b = 0; b < mimg->m; b++)
    {
      if (var[0] > IFT_EPSILON)
        P0 = exp(-powf(mimg->val[p][b] - mean[0], 2) / (2.0 * var[0]));
      else
      {
        var[0] = 1.0;
        P0 = exp(-powf(mimg->val[p][b] - mean[0], 2) / (2.0 * var[0]));
      }
      if (var[1] > IFT_EPSILON)
        P1 = exp(-powf(mimg->val[p][b] - mean[1], 2) / (2.0 * var[1]));
      else
      {
        var[1] = 1.0;
        P1 = exp(-powf(mimg->val[p][b] - mean[1], 2) / (2.0 * var[1]));
      }

      if ((channel_label[b] > 0) && (P0 > P1))
      {
        weight[b] = 1; // P0;
      }
      else
      {
        if ((channel_label[b] < 0) && (P0 < P1))
        {
          weight[b] = -1; // P1
        }
        else
        {
          weight[b] = 0;
        }
      }
    }

    /* Estimate the salience map using one neuron per pixel -- it is
       no longer a point-wise convolution */

    for (int b = 0; b < mimg->m; b++)
    {
      salie->val[p] += mimg->val[p][b] * weight[b];
    }
    if (salie->val[p] < 0) /* RELU (Sigmoid?) */
      salie->val[p] = 0;
  }
  iftDestroyAdjRel(&A);
  iftFree(weight);

  return (salie);
}

int NumberOfTruelabels(int *truelabels, int n)
{
  int nlabels = 0;
  for (int i = 0; i < n; i++)
    if (truelabels[i] > nlabels)
      nlabels = truelabels[i];
  return (nlabels);
}

iftFLIMGraphArray *ImageToFLIMGraphBatch(iftFileSet *fs, char *labels_dir, 
                                int first, int last, 
                                int initial_seeds, int nsuperpixels,
                                char **basename)
{
    iftFLIMGraphArray *input = CreateEmptyFLIMGraphArray(last - first + 1);

#pragma omp parallel for
    for (int k = 0; k <= last - first; k++)
    {
        int i = k + first;
        char ext[10], label_file[200];

        sprintf(ext, "%s", iftFileExt(fs->files[i]->path));
        basename[k] = iftFilename(fs->files[i]->path, ext);
        sprintf(label_file, "%s/%s.pgm", labels_dir, basename[k]);

        iftImage *img = iftReadImageByExt(fs->files[i]->path);
        iftImage *label_img = NULL;

        input->val[k] = iftImageToFLIMGraphDISF(img, initial_seeds, nsuperpixels, &label_img);
        iftWriteImageByExt(label_img, label_file);
        iftDestroyImage(&label_img);
    }
    return input;
}

iftMImage *convLayer(iftFLIMGraph *graph, iftMatrix *kernels, int kernel_size)
{
  iftMatrix *XI = iftFLIMGraphToFeatureMatrix(graph, kernel_size, 0, true);
  iftMatrix *XJ = iftMultMatrices(XI, kernels);
  iftDestroyMatrix(&XI);
  iftMImage *activ = iftMatrixToMImage(XJ, XJ->nrows,
                                       1, 1,
                                       kernels->ncols, 'c');
  iftDestroyMatrix(&XJ);
  return activ;
}

iftMImageArray *convLayerBatch(iftFLIMGraphArray *graph, iftMatrix *kernels, int kernel_size)
{
  iftMImageArray *activ = iftCreateMImageArray(graph->n);
  for(int i = 0; i < graph->n; i++){
    activ->val[i] = convLayer(graph->val[i], kernels, kernel_size);
  }
  return activ;
}

void pooling(iftMImage *activ, iftFLIMGraph *graph, 
              char *pool_type, int *pool_size, int pool_stride)
{
  iftMImage *pool;

  if (strcmp(pool_type, "no_pool") != 0)
  {
    if (strcmp(pool_type, "avg_pool") == 0)
    {
      pool = iftFLIMGraphAtrousAveragePooling(activ,
                                              graph,
                                              pool_size[0],
                                              pool_size[1],
                                              pool_size[2],
                                              1,
                                              pool_stride, true);
    }
    else
    {
      if (strcmp(pool_type, "max_pool") == 0)
      {
        pool = iftFLIMGraphAtrousMaxPooling(activ,
                                            graph,
                                            pool_size[0],
                                            pool_size[1],
                                            pool_size[2],
                                            1,
                                            pool_stride, true);
      }
      else
        iftError("Invalid pooling %s", "pooling", pool_type);
    }

#pragma omp parallel for
    for (int j = 0; j < graph->num_nodes; j++)
    {
      for (int b = 0; b < graph->num_feats; b++)
      {
        graph->nodes[j].feats[b] = (double)pool->val[j][b];
      }
    }
    iftDestroyMImage(&pool);
  }
}

void poolingArray(iftMImageArray *activ_array, iftFLIMGraphArray *graph, 
                    char *pool_type, int *pool_size, int pool_stride)
{
  for(int i = 0; i < activ_array->n; i++){
    iftFLIMGraph *pooling_graph = iftMImageToFLIMGraph(activ_array->val[i], graph->val[i]);
    iftDestroyFLIMGraph(&(graph->val[i]));
    pooling(activ_array->val[i], pooling_graph, pool_type, pool_size, pool_stride);
    graph->val[i] = pooling_graph;
    pooling_graph = NULL;
  }
}

iftFLIMGraph *strideModel(iftFLIMGraph *graph, iftImage *input_labels, iftImage *output_labels, 
                          int stride)
{
  iftBMap *visited = iftCreateBMap(graph->num_nodes);
  iftAdjRel *A;

  // reduce labels image according to stride and compute mean color
  int num_nodes = 0;
  iftVoxel u;
  for (u.z = 0; u.z < input_labels->zsize; u.z = u.z + stride)
  {
    for (u.y = 0; u.y < input_labels->ysize; u.y = u.y + stride)
    {
      for (u.x = 0; u.x < input_labels->xsize; u.x = u.x + stride)
      {
        int p = iftMGetVoxelIndex(input_labels, u);
        int label = input_labels->val[p];
        if (!iftBMapValue(visited, label - 1))
        {
          iftBMapSet1(visited, label - 1);
          num_nodes++;
        }
      }
    }
  }

  int map_nodes[graph->num_nodes]; // map old to new node index
  int new_index = 0;
  for (int old_index = 0; old_index < graph->num_nodes; old_index++)
  {
    if (iftBMapValue(visited, old_index))
    {
      map_nodes[old_index] = new_index;
      new_index++;
    }
    else map_nodes[old_index] = -1;
  }
  
  if(new_index != num_nodes)
    iftError("Number of nodes is different from expected: %d %d", __func__, new_index, num_nodes);
  
  iftDestroyBMap(&visited);

  iftFLIMGraph *output_graph = iftCreateEmptyFLIMGraph(graph->num_feats, num_nodes);
  iftBMap *neighbors[num_nodes];

  if (iftIs3DImage(input_labels))
    A = iftSpheric(sqrtf(3.0));
  else
    A = iftCircular(sqrtf(2.0));

  for (int i = 0; i < num_nodes; i++)
    neighbors[i] = iftCreateBMap(num_nodes);

  int q = 0;
  for (u.z = 0; u.z < input_labels->zsize; u.z = u.z + stride)
  {
    for (u.y = 0; u.y < input_labels->ysize; u.y = u.y + stride)
    {
      for (u.x = 0; u.x < input_labels->xsize; u.x = u.x + stride)
      {
        int p = iftMGetVoxelIndex(input_labels, u);
        int old_label = input_labels->val[p] - 1;
        int new_label = map_nodes[old_label];
        iftFLIMGraphNode *node = &(output_graph->nodes[new_label]);

        output_labels->val[q] = new_label+1;
        int feat = 0;
        while (feat < graph->num_feats)
        {
          node->feats[feat] += graph->nodes[old_label].feats[feat];
          feat++;
        }

        node->numPixels++;
        q++;
      }
    }
  }

  q = 0;
  for (u.z = 0; u.z < input_labels->zsize; u.z = u.z + stride)
  {
    for (u.y = 0; u.y < input_labels->ysize; u.y = u.y + stride)
    {
      for (u.x = 0; u.x < input_labels->xsize; u.x = u.x + stride)
      {
        int p = iftMGetVoxelIndex(input_labels, u);
        int old_label = input_labels->val[p] - 1;
        int new_label = map_nodes[old_label];
        iftFLIMGraphNode *node = &(output_graph->nodes[new_label]);

        iftVoxel p_voxel = iftGetVoxelCoord(output_labels, q);
        for (int i = 1; i < A->n; i++)
        {
          iftVoxel q_voxel = iftGetAdjacentVoxel(A, p_voxel, i);
          if (iftValidVoxel(output_labels, q_voxel))
          {
            int q_index = iftMGetVoxelIndex(output_labels, q_voxel);
            int q_label = output_labels->val[q_index]-1;
            if (new_label > q_label && !iftBMapValue(neighbors[new_label], q_label))
            {
              iftInsertFLIMGraphNeighborNode(node, q_label);
              iftInsertFLIMGraphNeighborNode(&(output_graph->nodes[q_label]), new_label);
              iftBMapSet1(neighbors[new_label], q_label);
              output_graph->num_edges++;
            }
          }
        }
        q++;
      }
    }
  }

  // for each node, compute its mean color
  iftFLIMGraphNode *node = output_graph->nodes;
  for (int spx_index = 0; spx_index < num_nodes; spx_index++)
  {
    for (int band_index = 0; band_index < output_graph->num_feats; band_index++)
      node->feats[band_index] /= (double)node->numPixels;
    node++;
    iftDestroyBMap(&neighbors[spx_index]);
  }
  node = NULL;
  iftDestroyAdjRel(&A);

  return output_graph;
}

void strideModelBatch(iftFLIMGraphArray *graph, char **basename,
                    char *in_labels_dir, char *out_labels_dir,
                    int first, int last, iftFLIMLayer layer)
{
    for (int i = first; i <= last; i++)
    {
        char filename[200];
        /* stride model */
        sprintf(filename, "%s/%s.pgm", in_labels_dir, basename[i-first]);
        iftImage *input_labels = iftReadImageByExt(filename);

        int new_x = ceilf(input_labels->xsize / (float)layer.pool_stride);
        int new_y = ceilf(input_labels->ysize / (float)layer.pool_stride);
        int new_z = iftMax(ceilf(input_labels->zsize / (float)layer.pool_stride), 1);

        iftImage *output_labels = iftCreateImage(new_x, new_y, new_z);
        iftFLIMGraph *output_graph = strideModel(graph->val[i - first], input_labels, output_labels, layer.pool_stride);
        iftDestroyImage(&input_labels);

        /* save output */
        sprintf(filename, "%s/%s.pgm", out_labels_dir, basename[i-first]);
        iftWriteImageByExt(output_labels, filename);
        iftDestroyImage(&output_labels);

        iftDestroyFLIMGraph(&(graph->val[i - first]));
        graph->val[i - first] = output_graph;
        output_graph = NULL;
    }
}

void encodeLayerBatch(iftFLIMGraphArray *input, 
                iftFLIMLayer layer, int l, 
                char *model_dir, int first, int last,
                char **basename)
{
    /* Load model */
    iftMatrix *Kmerged = NULL;
    float *bias_merged = NULL;
    char filename[200];
    int nkernels;

    sprintf(filename, "%s/model/conv%d", model_dir, l + 1);
    LoadMergedModel(filename, &Kmerged, &bias_merged, &nkernels);
    
    /* convolution */
    int kernel_size = layer.kernel_size[0] * layer.kernel_size[1];
    iftMImageArray *activ = convLayerBatch(input, Kmerged, kernel_size);
    
    /* activation */
    iftMImageArray *activ_relu = iftMReLUArray(activ, bias_merged, layer.relu);
    if (activ_relu != activ)
        iftDestroyMImageArray(&activ);
    
    /* pooling */
    poolingArray(activ_relu, input, layer.pool_type, layer.pool_size, 1);
    iftDestroyMImageArray(&activ_relu);

    if (layer.pool_stride > 1)
    {
        char in_labels_dir[200], out_labels_dir[200];
        sprintf(in_labels_dir, "%s/labels%d", model_dir, l);
        sprintf(out_labels_dir, "%s/labels%d", model_dir, l + 1);
        strideModelBatch(input, basename, in_labels_dir, out_labels_dir, first, last, layer);
    }

    iftDestroyMatrix(&Kmerged);
    iftFree(bias_merged);
}

void decodeLayer(iftFLIMGraph *input, 
                char *model_dir, char *basename, bool has_stride,
                int nLayers, int nlabels, int nkernels, 
                int *truelabels)
{
    char filename[200];
    int Imax;
    
    sprintf(filename, "%s/labels%d/%s.pgm", model_dir, nLayers, basename);
    if(!iftFileExists(filename))
        sprintf(filename, "%s/labels0/%s.pgm", model_dir, basename);
    
    iftImage *label_img = iftReadImageByExt(filename);
    iftMImage *mimg = iftFLIMGraphToMImage(input, label_img);
    iftDestroyImage(&label_img);
    
    float scale[] = {1.0, 1.0, 1.0};

    if (has_stride)
    {
        sprintf(filename, "%s/labels0/%s.pgm", model_dir, basename);
        iftImage *labels0 = iftReadImageByExt(filename);
        scale[0] = (float)labels0->xsize / (float)mimg->xsize;
        scale[1] = (float)labels0->ysize / (float)mimg->ysize;
        scale[2] = (float)labels0->zsize / (float)mimg->zsize;
        iftDestroyImage(&labels0);
    }

    /* Estimate one salience map per object (class) */
    iftFImage **salie = NULL;
    if (nlabels != 0)
    {
        salie = (iftFImage **)calloc(nlabels, sizeof(iftFImage *));
        float *weight = iftAllocFloatArray(nkernels);
        for (int l = 0; l < nlabels; l++)
        {
            /* set weights */
            for (int k = 0; k < nkernels; k++)
            {
                if ((l + 1) == truelabels[k])
                    weight[k] = +1;
                else
                    weight[k] = -1;
            }
            // salie[l]  = FixedWeightDecoder(mimg,weight);
            //salie[l] = AdaptiveDecoder2(mimg, 1.5, weight);
            salie[l] = HybridAdaptativeDecoder(mimg, weight);
        }
        iftFree(weight);
    }
    else
    {
        salie = (iftFImage **)calloc(1, sizeof(iftFImage *));
        salie[0] = TriStateAdaptativeDecoder(mimg);
        nlabels = 1;
    }

    for (int l = 0; l < nlabels; l++)
    {
        /* Interpolate and normalize the salience maps */
        iftFImage *smap_temp = NULL;
        if (iftIs3DMImage(mimg))
        {
            smap_temp = iftFInterp(salie[l], scale[0], scale[1], scale[2]);
            Imax = 4095;
        }
        else
        {
            smap_temp = iftFInterp2D(salie[l], scale[0], scale[1]);
            Imax = 255;
        }

        iftImage *smap = iftFImageToImage(smap_temp, Imax);
        iftDestroyFImage(&smap_temp);
        
        char saliency_dir[200];
        sprintf(saliency_dir, "%s/saliency%d/label%d", model_dir, nLayers, l + 1);
        if (!iftDirExists(saliency_dir))
            iftMakeDir(saliency_dir);

        /* save the salience map */
        if (iftIs3DMImage(mimg))
        {
            sprintf(filename, "%s/%s_layer%d.nii.gz", saliency_dir, basename, nLayers);
        }
        else
        {
            sprintf(filename, "%s/%s_layer%d.png", saliency_dir, basename, nLayers);
        }

        iftWriteImageByExt(smap, filename);
        iftDestroyImage(&smap);
        iftDestroyFImage(&salie[l]);
    }
    iftDestroyMImage(&mimg);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 7)
        iftError("Usage: iftGFLIMNSExtract <P1> <P2> <P3> <P4> <P5> <P6> \n"
                 "[1] Images directory \n"
                 "[2] Model directory \n"
                 "[3] Architecture file \n"
                 "[4] Number of layers \n"
                 "[5] Initial seeds \n"
                 "[6] Number of superpixels \n"
                 //"[7] Batch size \n"
                 ,
                 "main");

    tstart = iftTic();

    char *images_dir = argv[1];
    char *model_dir = argv[2];
    iftFLIMArch *arch = iftReadFLIMArch(argv[3]);
    int nLayers = atoi(argv[4]);
    int initial_seeds = atoi(argv[5]);
    int nsuperpixels = atoi(argv[6]);

    if (nLayers < 1)
    iftError("Invalid number of layers: %d", "main", nLayers);

    if(nLayers > arch->nlayers)
      iftError("Invalid number of layers: %d", "main", nLayers);

    if(!iftDirExists(images_dir))
        iftError("Images directory does not exist", "main");
    
    if(!iftDirExists(model_dir))
        iftError("Model directory does not exist", "main");

    if(initial_seeds < 1)
        iftError("Invalid number of initial seeds: %d", "main", initial_seeds);
    
    if(nsuperpixels < 1)
        iftError("Invalid number of superpixels: %d", "main", nsuperpixels);
    
    if(nsuperpixels > initial_seeds)
        iftError("Number of superpixels (%d) must be greater than the number of initial seeds (%d)", "main", nsuperpixels, initial_seeds);

    iftFileSet *fs_imgs = iftLoadFileSetFromDir(images_dir, 1);
    int ngraphs = fs_imgs->n;
    int batchsize = 0, nbatches = 0;
    int nremaining_graphs = fs_imgs->n;
    int ninput_channels = 3;
    int max_pixels = 0;
    int max_edges = 0;

    if (fs_imgs->n < 1) iftError("Image list is empty", __func__);

    char filename[200];
    int number_of_kernels;
    sprintf(filename, "%s/model/conv1-labels.txt", model_dir);
    if(!iftFileExists(filename))
      iftError("File %s does not exist", "main", filename);
    FILE *fp = fopen(filename, "r");
    fscanf(fp, "%d", &number_of_kernels);
    fclose(fp);
    
    /* Verify if all graphs have the same dimension for batch processing */

    // Ensure that each thread accesses a different cache line (https://wgropp.cs.illinois.edu/courses/cs598-s15/lectures/lecture18.pdf)
    typedef struct{ int edges; int pixels; char pad[128]; } tvals;

    tvals maxVals[omp_get_max_threads()];

#pragma omp parallel shared(maxVals)
    {
        int id = omp_get_thread_num();
        maxVals[id].edges = 0;
        maxVals[id].pixels = 0;

        for (int i = 0; i < fs_imgs->n; i++)
        {
            iftImage *input_image = iftReadImageByExt(fs_imgs->files[i]->path);
            maxVals[id].edges = iftMax(maxVals[id].edges, (input_image->xsize - 2) * (input_image->ysize - 2) * 8 + (input_image->ysize + input_image->xsize - 4) * 10 + 12);
            maxVals[id].pixels = iftMax(maxVals[id].pixels, input_image->n);
            iftDestroyImage(&input_image);
        }

#pragma omp flush(maxVals)
#pragma omp master
        {
            int nt = omp_get_num_threads();
            max_edges = maxVals[0].edges;
            max_pixels = maxVals[0].pixels;
            for (int i = 1; i < nt; i++)
            {
                if (maxVals[i].edges > max_edges)
                    max_edges = maxVals[i].edges;
                if (maxVals[i].pixels > max_pixels)
                    max_pixels = maxVals[i].pixels;
            }
        }
    }

    max_edges -= (max_pixels - nsuperpixels);
    if(max_edges > nsuperpixels*(nsuperpixels-1)/2) max_edges = nsuperpixels*(nsuperpixels-1)/2;

    /* Select batch size for GPU/CPU processing */
    
    batchsize = batchSizeCPU(arch, nsuperpixels, ninput_channels, max_edges, number_of_kernels, nLayers);
    
    batchsize = iftMin(batchsize, ngraphs);
    nbatches = ngraphs / batchsize;
    nremaining_graphs = nremaining_graphs - nbatches * batchsize;

    for (int batch = 0; batch < nbatches; batch++)
    {
        int first = batch * batchsize, last = first + batchsize - 1;
        printf("Processing batch %d of %d batches\n", batch + 1, nbatches);
        fflush(stdout);

        iftFLIMGraphArray *input = NULL;
        int ngraphs_batch = last - first + 1;
        char **basename = (char **)calloc(ngraphs_batch, sizeof(char *));
        char filename[200];

        /* Create graphs */
        sprintf(filename, "%s/labels0", model_dir);
        if(!iftDirExists(filename))
            iftMakeDir(filename);
        input = ImageToFLIMGraphBatch(fs_imgs, filename, first, last, initial_seeds, nsuperpixels, basename);

        bool has_stride = false;
        
        /* For each layer do */
        for (int l = 0; l < nLayers; l++)
        {
            encodeLayerBatch(input, arch->layer[l], l, model_dir, first, last, basename);
            if(arch->layer[l].pool_stride > 1)
                has_stride = true;
            
            /*char tmp[255], out[255], labels[255];
            sprintf(tmp, "%s/layer%d", model_dir, l + 1);
            sprintf(out, "%s/activation/val%d", model_dir, l + 1);
            iftMakeDir(out);
            sprintf(labels, "%s/labels%d", model_dir, l + 1);
            iftWriteActivationsImg(tmp, out, "pgm", labels);*/

        }
        
        /* Load labels indicating class (object) per channel */
        int *truelabels = NULL, nlabels = 0, nkernels;
        sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, nLayers);
        if (iftFileExists(filename))
        { /* load merged model */
          truelabels = LoadKernelLabels(filename, &nkernels);
          nlabels = NumberOfTruelabels(truelabels, nkernels);
        }

        // decode
#pragma omp parallel for
        for (int k = 0; k < ngraphs_batch; k++)
        {
            decodeLayer(input->val[k], model_dir, basename[k], has_stride, nLayers, nlabels, nkernels, truelabels);
            iftDestroyFLIMGraph(&input->val[k]);
            iftFree(basename[k]);
        }
        iftFree(input);
        iftFree(basename);
        iftFree(truelabels);
    }

    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last)
    {
        printf("Processing remaining %d images\n", last - first + 1);
        fflush(stdout);

        iftFLIMGraphArray *input = NULL;
        int ngraphs_batch = last - first + 1;
        char **basename = (char **)calloc(ngraphs_batch, sizeof(char *));
        char filename[200];

        /* Create graphs */
        sprintf(filename, "%s/labels0", model_dir);
        if(!iftDirExists(filename))
            iftMakeDir(filename);
        input = ImageToFLIMGraphBatch(fs_imgs, filename, first, last, initial_seeds, nsuperpixels, basename);

        bool has_stride = false;

        /* For each layer do */
        for (int l = 0; l < nLayers; l++)
        {
            encodeLayerBatch(input, arch->layer[l], l, model_dir, first, last, basename);
            if(arch->layer[l].pool_stride > 1)
                has_stride = true;
            
            /*char tmp[255], out[255], labels[255];
            sprintf(tmp, "%s/layer%d", model_dir, l + 1);
            sprintf(out, "%s/activation/val%d", model_dir, l + 1);
            iftMakeDir(out);
            sprintf(labels, "%s/labels%d", model_dir, l + 1);
            iftWriteActivationsImg(tmp, out, "pgm", labels);*/
        }

        /* Load labels indicating class (object) per channel */
        int *truelabels = NULL, nlabels = 0, nkernels;
        sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, nLayers);
        if (iftFileExists(filename))
        { /* load merged model */
          truelabels = LoadKernelLabels(filename, &nkernels);
          nlabels = NumberOfTruelabels(truelabels, nkernels);
        }

        // decode
#pragma omp parallel for
        for (int k = 0; k < ngraphs_batch; k++)
        {
            decodeLayer(input->val[k], model_dir, basename[k], has_stride, nLayers, nlabels, nkernels, truelabels);
            iftDestroyFLIMGraph(&input->val[k]);
            iftFree(basename[k]);
        }
        iftFree(input);
        iftFree(basename);
        iftFree(truelabels);
    }

    iftDestroyFileSet(&fs_imgs);
    iftDestroyFLIMArch(&arch);

    printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
    return 0;
}
