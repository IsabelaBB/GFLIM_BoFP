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

int main(int argc, char *argv[])
{
  timer *tstart;

  if (argc != 4)
    iftError("Usage: iftEncodeMergedLayer <P1> <P2> <P3> \n"
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
  
  char input_dir[200], labels_dir[200], output_dir[200], out_labels_dir[200];
  char *filename = iftAllocCharArray(512);

  sprintf(input_dir, "%s/layer%d", model_dir, layer - 1);
  if (!iftDirExists(input_dir))
    iftError("Graph directory does not exist", "main");

  sprintf(labels_dir, "%s/labels%d", model_dir, layer - 1);
  if (!iftDirExists(labels_dir))
    iftError("Labels directory does not exist", "main");

  sprintf(output_dir, "%s/layer%d", model_dir, layer);
  iftMakeDir(output_dir);

  sprintf(out_labels_dir, "%s/labels%d", model_dir, layer);
  iftMakeDir(out_labels_dir);

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

    /* stride model */
    sprintf(filename, "%s/%s.pgm", labels_dir, basename);
    iftImage *input_labels = iftReadImageByExt(filename);

    int new_x = ceilf(input_labels->xsize / (float)arch->layer[layer - 1].pool_stride);
    int new_y = ceilf(input_labels->ysize / (float)arch->layer[layer - 1].pool_stride);
    int new_z = iftMax(ceilf(input_labels->zsize / (float)arch->layer[layer - 1].pool_stride), 1);

    iftImage *output_labels = iftCreateImage(new_x, new_y, new_z);

    iftFLIMGraph *output_graph = strideModel(pooling_graph, input_labels, output_labels, arch->layer[layer - 1].pool_stride);
    iftDestroyFLIMGraph(&pooling_graph);
    iftDestroyImage(&input_labels);

    
    /* save output */
    sprintf(filename, "%s/%s.pgm", out_labels_dir, basename);
    iftWriteImageByExt(output_labels, filename);
    iftDestroyImage(&output_labels);

    sprintf(filename, "%s/%s.json", output_dir, basename);
    iftWriteFLIMGraph(output_graph, filename);
    iftDestroyFLIMGraph(&output_graph);

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
