#include "ift.h"
#include "iftGFLIM.h"

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

void LoadMergedModel(char *basepath, iftMatrix **K, float **bias)
{
  char filename[200];
  FILE *fp;
  int nkernels;
  
  sprintf(filename, "%s-kernels.npy", basepath);
  *K = iftReadMatrix(filename);
  sprintf(filename, "%s-bias.txt", basepath);
  fp = fopen(filename, "r");
  fscanf(fp, "%d", &nkernels);
  *bias = iftAllocFloatArray(nkernels);
  for (int k = 0; k < nkernels; k++) {
    fscanf(fp, "%f ", &((*bias)[k]));
  }  
  fclose(fp);
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

int main(int argc, char *argv[]) 
{
    timer *tstart;

    if (argc != 5)
    iftError("Usage: iftEncodeMergedLayer <P1> <P2> <P3> \n"
             "[1] model directory \n"
             "[2] architecture of the network (.json) \n"
             "[3] layer number (1, 2, 3) \n"
             "[4] batch size (1, 8, 16, 32, etc) \n",
             "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftFLIMArch *arch = iftReadFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    int batchsize = atoi(argv[4]);

    if (layer < 1)
    iftError("Invalid layer number: %d", "main", layer);

    if(layer > arch->nlayers)
      iftError("Invalid layer number: %d", "main", layer);

    if (batchsize < 1)
      iftError("Invalid batch size: %d", "main", batchsize);

    char input_dir[200], output_dir[200];
    char *filename = iftAllocCharArray(512);

    sprintf(input_dir, "%s/layer%d", model_dir, layer - 1);
    if (!iftDirExists(input_dir))
      iftError("Graph directory does not exist", "main");

    sprintf(output_dir, "%s/layer%d", model_dir, layer);
    iftMakeDir(output_dir);

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", 1);

    int kernel_size = arch->layer[layer-1].kernel_size[0] * arch->layer[layer-1].kernel_size[1];
    
    iftMatrix *Kmerged = NULL;
    float *bias_merged = NULL;

    sprintf(filename, "%s/model/conv%d", model_dir, layer);
    LoadMergedModel(filename, &Kmerged, &bias_merged);
    
    /* Encode layer with merged model */      

    for (int i = 0; i < fs->n; i += batchsize) 
    {
      int first = i;
      int last = iftMin(i+batchsize-1, fs->n-1);

      /* read graphs */
      iftFLIMGraphArray *input_graph = CreateEmptyFLIMGraphArray(last - first + 1);
      for (int j = 0; j < (last - first + 1); j++)
        input_graph->val[j] = iftReadFLIMGraph(fs->files[i + j]->path);

      /* convolution */
      iftMImageArray *activ = convLayerBatch(input_graph, Kmerged, kernel_size);
      
      /* activation */
      iftMImageArray *activ_relu = iftMReLUArray(activ, bias_merged, arch->layer[layer - 1].relu);
      if (activ_relu != activ)
	      iftDestroyMImageArray(&activ);
      
      /* pooling */
      poolingArray(activ_relu, input_graph, arch->layer[layer - 1].pool_type, arch->layer[layer - 1].pool_size, 1);
      iftDestroyMImageArray(&activ_relu);
      
      /* write activations */
      for (int i = first; i <= last; i++)
      {
        char filename[200];
        char *basename = iftFilename(fs->files[i]->path, ".json");
        sprintf(filename, "%s/%s.json", output_dir, basename);
        iftWriteFLIMGraph(input_graph->val[i - first], filename);
        iftFree(basename);
      }
      
      iftDestroyLIMGraphArray(&input_graph);
    }
    
    iftDestroyMatrix(&Kmerged);
    iftFree(bias_merged);
    iftFree(filename);
    iftDestroyFileSet(&fs);
    iftDestroyFLIMArch(&arch);
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
    return (0);
}
