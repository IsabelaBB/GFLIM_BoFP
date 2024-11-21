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

    int batchsize = (int)floor(percMemory * freeMemoryMb / max_requiredMemory);
    return (batchsize);
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

int element_link(int x, int y, int *rank, int *parent)
{
  // If necessary, rename variables to ensure that
  // x has rank at least as large as that of y
  if (rank[x] < rank[y])
  {
    int t = x;
    x = y;
    y = t;
  }

  // If necessary, increment the rank of x
  if (rank[x] == rank[y])
    rank[x] = rank[x] + 1;
  
  // Make x the new root
  parent[y] = x;
  return x;
}

int element_find(int x, int *parent)
{ 
  if (parent[x] != x) 
    parent[x] = element_find(parent[x], parent);  
  return parent[x];
}

int union_set(int x, int y, int *parent, int *rank){
    // Replace nodes by roots
    x = element_find(x, parent);
    y = element_find(y, parent);

    if (x == y){
        return -1; // x and y are already in the same set
    }

    return element_link(x, y, rank, parent);
}

iftFLIMGraph *strideModel(iftFLIMGraph *graph, int stride, int *nodes_map)
{
  int num_nodes_out = graph->num_nodes / stride;
  double *distance = iftAllocDoubleArray(graph->num_nodes);
	int closest_neigh[graph->num_nodes];
	iftDHeap *heap = iftCreateDHeap(graph->num_nodes, distance);
  
  int rank[graph->num_nodes];
  int parent[graph->num_nodes];

  // weight nodes accoding to the minimum distance to their neighbors
	for(int i = 0; i < graph->num_nodes; i++){
		iftFLIMGraphNode *node = &(graph->nodes[i]);
    iftSet *adj = node->adj;

    double min_dist = INFINITY;
    int cn = -1;
    while(adj != NULL){
      double val = 0;
      for (int j = 0; j < graph->num_feats; j++) { // euclidean distance
        val += (node->feats[j] - graph->nodes[adj->elem].feats[j]) * (node->feats[j] - graph->nodes[adj->elem].feats[j]);
      }
      val = sqrt(val);
      if(val < min_dist){
        min_dist = val;
        cn = adj->elem;
      }
      adj = adj->next;
    }

		closest_neigh[i] = cn;
		distance[i] = min_dist;
		iftInsertDHeap(heap, i);
    rank[i] = 1;
    parent[i] = i;
    nodes_map[i] = -1;
  }
  iftFree(distance);
  
  // merge nodes until achieve the required number of nodes
  int removed_nodes = 0;
  while(!iftEmptyDHeap(heap) && removed_nodes != graph->num_nodes - num_nodes_out)
  {
    int x = iftRemoveDHeap(heap);
    int y = closest_neigh[x];

    int p = union_set(x,y,parent,rank);
    if(p == -1) continue;
    removed_nodes++;
  }
  iftDestroyDHeap(&heap);
  
  if(removed_nodes != graph->num_nodes - num_nodes_out){
    iftWarning("The number of output nodes was changed from %d to %d", __func__, num_nodes_out, graph->num_nodes-removed_nodes);
    num_nodes_out = graph->num_nodes-removed_nodes;
  }

  iftFLIMGraph *output_graph = iftCreateEmptyFLIMGraph(graph->num_feats, num_nodes_out);
  int j = 0;
  for(int i = 0; i < graph->num_nodes; i++)
  {
    int p = element_find(i, parent);
    if(nodes_map[p] == -1){
      nodes_map[p] = j;
      j++;
    }
    int q = nodes_map[p];
    nodes_map[i] = q;
    
    iftFLIMGraphNode *node = &(output_graph->nodes[q]);
    iftFLIMGraphNode *remove_node = &(graph->nodes[i]);

    // merge node features
    for(int b = 0; b < graph->num_feats; b++){
      node->feats[b] *= node->numPixels;
      node->feats[b] += (remove_node->feats[b] * remove_node->numPixels);
      node->feats[b] /= (node->numPixels + remove_node->numPixels);
    }

    // merge their number of pixels
    node->numPixels += remove_node->numPixels;

    // merge neighbors
    iftSet *adj = remove_node->adj;
    while(adj != NULL)
    {
      int adj_label = nodes_map[adj->elem];
      if(adj->elem < i && 
          adj_label != q && 
          (iftUnionSetElem(&(node->adj), adj_label) == 1))
      {
        node->numNeighbors++;
        iftInsertFLIMGraphNeighborNode(&(output_graph->nodes[adj_label]), q);
      }
      adj = adj->next;
    }
  }

  return output_graph;
}

void remapGraphLabels(iftImage *input_labels, int *nodes_map)
{
  for(int i = 0; i < input_labels->n; i++)
    input_labels->val[i] = nodes_map[input_labels->val[i]-1]+1;
}

void strideModelBatch(iftFLIMGraphArray *graph, char **basename,
                    char *in_labels_dir, char *out_labels_dir,
                    int first, int last, iftFLIMLayer layer)
{
    for (int i = first; i <= last; i++)
    {
        char filename[200];
        /* stride model */
        int nodes_map[graph->val[i - first]->num_nodes];
        iftFLIMGraph *output_graph = strideModel(graph->val[i - first], layer.pool_stride, nodes_map);
      
        sprintf(filename, "%s/%s.pgm", in_labels_dir, basename[i-first]);
        iftImage *input_labels = iftReadImageByExt(filename);
        remapGraphLabels(input_labels, nodes_map);

        /* save output */
        sprintf(filename, "%s/%s.pgm", out_labels_dir, basename[i-first]);
        iftWriteImageByExt(input_labels, filename);
        iftDestroyImage(&input_labels);

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
        iftError("Label file %s does not exist", "decodeLayer", filename);
    
    iftImage *label_img = iftReadImageByExt(filename);
    iftMImage *mimg = iftFLIMGraphToMImage(input, label_img);
    iftDestroyImage(&label_img);
    
    float scale[] = {1.0, 1.0, 1.0};

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
        iftError("Usage: iftGFLIMGSExtract <P1> <P2> <P3> <P4> <P5> <P6> \n"
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

    int number_of_kernels;
    char filename[200];
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
