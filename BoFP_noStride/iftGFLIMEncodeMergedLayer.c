#include "ift.h"
#include "iftGFLIM.h"

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
  for (int k = 0; k < nkernels; k++)
  {
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

int main(int argc, char *argv[])
{
  timer *tstart;

  if (argc != 4)
    iftError("Usage: iftGFLIMEncodeMergedLayer <P1> <P2> <P3> \n"
             "[1] model directory \n"
             "[2] architecture of the network (.json) \n"
             "[3] layer number (1, 2, 3) \n",
             "main");

  tstart = iftTic();

  char *model_dir = argv[1];
  iftFLIMArch *arch = iftReadFLIMArch(argv[2]);
  int layer = atoi(argv[3]);

  if (layer < 1)
    iftError("Invalid layer number: %d", "main", layer);

  if(layer > arch->nlayers)
    iftError("Invalid layer number: %d", "main", layer);

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

  for (int i = 0; i < fs->n; i++)
  {
    char *basename;
    basename = iftFilename(fs->files[i]->path, ".json");

    sprintf(filename, "%s/%s.json", input_dir, basename);
    iftFLIMGraph *graph = iftReadFLIMGraph(filename);

    iftMImage *activ = convLayer(graph, Kmerged, kernel_size);

    iftMImage *activ_relu = iftMReLU(activ, bias_merged, arch->layer[layer - 1].relu);
    if (activ_relu != activ)
	    iftDestroyMImage(&activ);

    /* pooling */
    iftFLIMGraph *pooling_graph = iftMImageToFLIMGraph(activ_relu, graph);
    iftDestroyFLIMGraph(&graph);
    pooling(activ_relu, pooling_graph, arch->layer[layer - 1].pool_type, arch->layer[layer - 1].pool_size, 1);
    iftDestroyMImage(&activ_relu);

    sprintf(filename, "%s/%s.json", output_dir, basename);
    iftWriteFLIMGraph(pooling_graph, filename);
    iftDestroyFLIMGraph(&pooling_graph);

    iftFree(basename);
  }

  iftDestroyMatrix(&Kmerged);
  iftFree(bias_merged);
  iftFree(filename);
  iftDestroyFileSet(&fs);
  iftDestroyFLIMArch(&arch);
  printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
  return (0);
}
