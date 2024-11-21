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

int test(int argc, char *argv[])
{
  iftImage *input_labels = iftCreateImage(5, 5, 1);
  int labels [] = {2, 1, 1, 6, 6, 
                  2, 1, 1, 6, 6,
                  2, 7, 3, 3, 3,
                  7, 7, 5, 5, 4, 
                  7, 7, 5, 5, 4};
  for(int i = 0; i < input_labels->n; i++)
    input_labels->val[i] = labels[i];

  iftImage *img = iftCreateImage(5, 5, 1);
  int color [] = {4, 7, 7, 10, 10,
                  4, 7, 7, 10, 10,
                  4, 0, 15, 15, 15,
                  0, 0, 2, 2, 14, 
                  0, 0, 2, 2, 14};
  for(int i = 0; i < img->n; i++)
    img->val[i] = color[i];
  
  iftMImage *mimg = iftImageToMImage(img, GRAY_CSPACE);
  iftDestroyImage(&img);
  iftAdjRel *A = iftCircular(sqrtf(2.0));
  iftFLIMGraph *graph = iftSuperpixelImageToFLIMGraph(7, A, mimg, input_labels);

  // print graph
  /*printf("Input graph\n");
  printf("Num nodes: %ld\n", graph->num_nodes);
  printf("Num feats: %ld\n", graph->num_feats);
  for(int i = 0; i < graph->num_nodes; i++){
    iftFLIMGraphNode *node = &(graph->nodes[i]);
    printf("Node %d: ", i);
    for(int j = 0; j < graph->num_feats; j++){
      printf("%f ", node->feats[j]);
    }
    printf("\n");
    printf("Num pixels: %ld\n", node->numPixels);
    // print adjacents
    printf("Neighbors %ld: ", node->numNeighbors);
    iftSet *adj = node->adj;
    while(adj != NULL){
      printf("%d ", adj->elem);
      adj = adj->next;
    }
    printf("\n");
  }
  printf("\n");*/

  printf("Testing input graph... ");
  if(graph->num_nodes != 7)
    iftError("Invalid number of nodes: %d", "main", graph->num_nodes);
  if(graph->num_feats != 1)
    iftError("Invalid number of features: %d", "main", graph->num_feats);
  
  double feats[] = {7.0, 4.0, 15.0, 14.0, 2.0, 10.0, 0.0};
  int num_pixels[] = {4, 3, 3, 2, 4, 4, 5};
  int num_neighbors[] = {4, 2, 5, 2, 3, 2, 4};
  int neighbors[][5] = {{2,6,5,1,0}, {6,0,0,0,0}, {3,4,0,6,5}, {2,4,0,0,0}, {3,2,6,0,0}, {2,0,0,0,0}, {4,0,2,1,0}};
  for(int i = 0; i < graph->num_nodes; i++){
    iftFLIMGraphNode *node = &(graph->nodes[i]);
    if(node->numPixels != num_pixels[i])
      iftError("Invalid number of pixels %ld in node %d", "main", node->numPixels, i);
    if(node->feats[0] != feats[i])
      iftError("Invalid feature value %f in node %d", "main", node->feats[0], i);
    if(node->numNeighbors != num_neighbors[i])
      iftError("Invalid number of neighbors %ld in node %d", "main", node->numNeighbors, i);
    iftSet *adj = node->adj;
    int ok = 0;
    while(adj != NULL){
      int j = 0;
      for(int j = 0; j < node->numNeighbors; j++){
        if(j < num_neighbors[i] && neighbors[i][j] == adj->elem){
          j = node->numNeighbors;
          ok++;
          break;
        }
      }
      if(j == node->numNeighbors-1)
          iftError("Invalid neighbor %d in node %d", "main", adj->elem, i);
      adj = adj->next;
    }
    if(ok != num_neighbors[i])
      iftError("Invalid neighbors in node %d", "main", i);
  }
  printf("OK\n");

  int nodes_map[graph->num_nodes];
  iftFLIMGraph *output_graph = strideModel(graph, 2, nodes_map);

  printf("Testing nodes map... ");
  int correct_nodes_map[] = {0, 0, 1, 1, 2, 0, 2};
  for(int i = 0; i < graph->num_nodes; i++){
    if(nodes_map[i] != correct_nodes_map[i])
      iftError("Invalid node map %d in node %d", "main", nodes_map[i], i);
  }
  printf("OK\n");

  printf("Testing output graph... ");
  if(output_graph->num_nodes != 3)
    iftError("Invalid number of nodes: %d", "main", output_graph->num_nodes);
  if(output_graph->num_feats != 1)
    iftError("Invalid number of features: %d", "main", output_graph->num_feats);
  
  double feats2[] = {80.0/11.0, 14.6, 8.0/9.0};
  int num_pixels2[] = {11, 5, 9};
  int num_neighbors2[] = {2, 2, 2};
  int neighbors2[][2] = {{2,1}, {2,0}, {0,1}};
  for(int i = 0; i < output_graph->num_nodes; i++){
    iftFLIMGraphNode *node = &(output_graph->nodes[i]);
    if(node->numPixels != num_pixels2[i])
      iftError("Invalid number of pixels %ld in node %d", "main", node->numPixels, i);
    if(node->feats[0] != feats2[i])
      iftError("Invalid feature value %f in node %d", "main", node->feats[0], i);
    if(node->numNeighbors != num_neighbors2[i])
      iftError("Invalid number of neighbors %ld in node %d", "main", node->numNeighbors, i);
    iftSet *adj = node->adj;
    int ok = 0;
    while(adj != NULL){
      int j = 0;
      for(int j = 0; j < node->numNeighbors; j++){
        if(j < num_neighbors2[i] && neighbors2[i][j] == adj->elem){
          j = node->numNeighbors;
          ok++;
          break;
        }
      }
      if(j == node->numNeighbors-1)
          iftError("Invalid neighbor %d in node %d", "main", adj->elem, i);
      adj = adj->next;
    }
    if(ok != num_neighbors2[i])
      iftError("Invalid neighbors in node %d", "main", i);
  }
  printf("OK\n");

  // print labels_map
  /*printf("Labels map\n");
  for(int i = 0; i < graph->num_nodes; i++){
    printf("%d -> %d\n", i, nodes_map[i]);
  }
  printf("\n");*/

  // print output graph
  /*printf("Output graph\n");
  for(int i = 0; i < output_graph->num_nodes; i++){
    iftFLIMGraphNode *node = &(output_graph->nodes[i]);
    printf("Node %d: ", i);
    for(int j = 0; j < output_graph->num_feats; j++)
      printf("%f ", node->feats[j]);
    printf("\n");
    printf("Num pixels: %ld\n", node->numPixels);
    // print adjacents
    printf("Neighbors %ld: ", node->numNeighbors);
    iftSet *adj = node->adj;
    while(adj != NULL){
      printf("%d ", adj->elem);
      adj = adj->next;
    }
    printf("\n");
  }*/

  iftDestroyFLIMGraph(&graph);
  remapGraphLabels(input_labels, nodes_map);

  int correct_out_labels[] = {1,1,1,1,1,
                            1,1,1,1,1,
                            1,3,2,2,2,
                            3,3,3,3,2,
                            3,3,3,3,2};

  printf("Testing output labels... ");
  if(input_labels->n != 25)
    iftError("Invalid number of labels: %d", "main", input_labels->n);
  for(int i = 0; i < input_labels->n; i++){
    if(input_labels->val[i] != correct_out_labels[i])
      iftError("Invalid label %d in position %d", "main", input_labels->val[i], i);
  }
  printf("OK\n");

  // print labels
  /*printf("Labels\n");
  for(int i = 0; i < input_labels->n; i++)
    printf("%d ", input_labels->val[i]);
  printf("\n");*/

  iftDestroyImage(&input_labels);
  iftDestroyMImage(&mimg);
  iftDestroyAdjRel(&A);
  iftDestroyFLIMGraph(&output_graph);

  printf("Test passed\n");

  return 0;
}

int main(int argc, char *argv[])
{
  timer *tstart;

  if (argc != 4)
    iftError("Usage: iftGFLIMGSEncodeMergedLayer <P1> <P2> <P3> \n"
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
    char *basename = iftFilename(fs->files[i]->path, ".json");

    iftFLIMGraph *graph = iftReadFLIMGraph(fs->files[i]->path);
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
    int nodes_map[pooling_graph->num_nodes];
    iftFLIMGraph *output_graph = strideModel(pooling_graph, arch->layer[layer - 1].pool_stride, nodes_map);
    iftDestroyFLIMGraph(&pooling_graph);
    sprintf(filename, "%s/%s.pgm", labels_dir, basename);
    iftImage *input_labels = iftReadImageByExt(filename);
    remapGraphLabels(input_labels, nodes_map);
    
    /* save output */
    sprintf(filename, "%s/%s.pgm", out_labels_dir, basename);
    iftWriteImageByExt(input_labels, filename);
    iftDestroyImage(&input_labels);

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
