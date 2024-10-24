#include "ift.h"

iftMImageArray *convLayerBatch(iftFLIMGraph **graph, int ngraphs, iftMatrix *kernels, int kernel_size, bool relu){
  
  iftMImageArray *activ = iftCreateMImageArray(ngraphs);
  for(int i = 0; i < ngraphs; i++){
    iftMatrix *XI = iftFLIMGraphToFeatureMatrix(graph[i], kernel_size, 0, true);
    iftMatrix *XJ = iftMultMatrices(XI, kernels);
    iftDestroyMatrix(&XI);
    activ->val[i] = iftMatrixToMImage(XJ, XJ->nrows,
                                         1, 1,
                                         kernels->ncols, 'c');

    iftDestroyMatrix(&XJ);
  }
  return activ;
}

iftFLIMGraph *pooling(iftMImage **activ_ptr, iftFLIMGraph *graph, 
                    char *pool_type, int *pool_size, int pool_stride)
{
  iftMImage *activ = *activ_ptr;
  iftFLIMGraph *output_graph = iftMImageToFLIMGraph(activ, graph);

  if (strcmp(pool_type, "no_pool") != 0)
  {
    iftMImage *pool = NULL;
    if (strcmp(pool_type, "avg_pool") == 0)
    {
      pool = iftFLIMGraphAtrousAveragePooling(activ,
                                              output_graph,
                                              pool_size[0],
                                              pool_size[1],
                                              pool_size[2],
                                              1,
                                              pool_stride, true);
      iftDestroyMImage(&activ);
      activ = pool;
    }
    else
    {
      if (strcmp(pool_type, "max_pool") == 0)
      {
        pool = iftFLIMGraphAtrousMaxPooling(activ,
                                            output_graph,
                                            pool_size[0],
                                            pool_size[1],
                                            pool_size[2],
                                            1,
                                            pool_stride, true);
        iftDestroyMImage(&activ);
        activ = pool;
      }
      else
        iftError("Invalid pooling %s", "pooling", pool_type);
    }

#pragma omp parallel for
    for (int j = 0; j < output_graph->num_nodes; j++)
    {
      for (int b = 0; b < output_graph->num_feats; b++)
      {
        output_graph->nodes[j].feats[b] = (double)activ->val[j][b];
      }
    }
  }
  return output_graph;
}


void poolingArray(iftMImageArray *activ_array, iftFLIMGraph **graph, 
                    char *pool_type, int *pool_size, int pool_stride)
{
  for(int i = 0; i < activ_array->n; i++){
    iftFLIMGraph *output_graph = pooling(&activ_array->val[i], graph[i], pool_type, pool_size, pool_stride);
    iftDestroyFLIMGraph(&graph[i]);
    graph[i] = output_graph;
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

int main(int argc, char *argv[]) {
    timer *tstart;

    /* Example: iftEncodeMergedLayer arch.json 1 flim_models */
    
    if (argc != 7)
      iftError("Usage: iftEncodeMergedLayer <P1> <P2> <P3> <P4> <P5> <P6>\n"
         "[1] Layer input directory (with .json files) \n"
	       "[2] architecture of the network (.json) \n"
	       "[3] layer number (1, 2, 3) \n"
	       "[4] folder with the models \n"
	       "[5] batch size (1, 8, 16, 32, etc) \n",
         "[6] Layer output directory (with .json files) \n"
	       "main");

    char     *input_dir = argv[1];
    iftFLIMArch *arch   = iftReadFLIMArch(argv[2]);
    int          layer  = atoi(argv[3]);
    char    *model_dir  = argv[4];
    char    *filename   = iftAllocCharArray(512);
    int      batchsize  = atoi(argv[5]);
    char    *output_dir = argv[6];

    if(!iftDirExists(input_dir)) iftError("Input directory does not exist: %s", "main", input_dir);
    if(!iftDirExists(model_dir)) iftError("Model directory does not exist: %s", "main", model_dir);
    if(layer < 1) iftError("Invalid layer number: %d", "main", layer);
    if(batchsize < 1) iftError("Invalid batch size: %d", "main", batchsize);
    
    tstart = iftTic();
    
    iftMakeDir(output_dir);

    iftFileSet *fs   = iftLoadFileSetFromDirBySuffix(input_dir, ".json", true);

    int kernel_size = arch->layer[layer-1].kernel_size[0] * arch->layer[layer-1].kernel_size[1];
    int pool_size[3];
    pool_size[0] = arch->layer[layer-1].pool_size[0];
    pool_size[1] = arch->layer[layer-1].pool_size[1];
    pool_size[2] = arch->layer[layer-1].pool_size[2];
    int stride = arch->layer[layer-1].pool_stride;
    char *pool_type = arch->layer[layer-1].pool_type;
    bool relu = arch->layer[layer-1].relu;
    iftDestroyFLIMArch(&arch);

    iftMatrix *Kmerged     = NULL;
    float     *bias_merged = NULL;

    sprintf(filename,"%s/conv%d",model_dir,layer);
    LoadMergedModel(filename,&Kmerged,&bias_merged);
    
    /* Encode layer with merged model */      

    for (int i=0; i < fs->n; i += batchsize) {
      int first    = i;
      int last     = iftMin(i+batchsize-1,fs->n-1);

      /* read graphs */
      iftFLIMGraph **input_graph = (iftFLIMGraph**)malloc((last-first+1) * sizeof(iftFLIMGraph*));
      for (int j=0; j < (last-first+1); j++)
        input_graph[j] = iftReadFLIMGraph(fs->files[i+j]->path);

      /* convolution */
      iftMImageArray *convArray = convLayerBatch(input_graph, last-first+1, Kmerged, kernel_size, true);
      
      /* activation */
      iftMImageArray *reluArray  = iftMReLUArray(convArray, bias_merged, relu);
      if (reluArray != convArray)
	      iftDestroyMImageArray(&convArray);
      
      /* pooling */
      poolingArray(reluArray, input_graph, pool_type, pool_size, stride);
      iftDestroyMImageArray(&reluArray);
      
      /* write activations */
      for (int i = first; i <= last; i++)
      {
        char filename[200];
        char *basename = iftFilename(fs->files[i]->path, ".json");
        sprintf(filename, "%s/%s.json", output_dir, basename);
        iftWriteFLIMGraph(input_graph[i - first], filename);
        iftDestroyFLIMGraph(&input_graph[i - first]);
        iftFree(basename);
      }
      free(input_graph);
    }
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    iftDestroyMatrix(&Kmerged);
    iftFree(bias_merged);
    iftFree(filename);
    iftDestroyFileSet(&fs);
    
    return (0);
}
