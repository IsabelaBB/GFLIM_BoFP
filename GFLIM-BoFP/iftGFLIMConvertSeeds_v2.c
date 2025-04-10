/**
 * \file iftReduceSeedSet_v2.c
 * \brief Convert image-based markers to graph-based markers, performs a k-means per graph marker, and another k-means per label. 
 * \warning The markers coordinates must be in "*-seeds.txt" files.
 */

#include "ift.h"

/**
 * \brief Perform a k-means per marker on S and selects the seeds closest to the cluster centers.
 * \param graph the gram from which S was computed
 * \param S The labeled set with the seeds.
 * \param kernel_size The size of the kernel (number of neighbors) used to extract the features.
 * \param nclusters_per_marker The number of clusters per marker.
 * \param attention_opt The attention option to be used. @see iftFLIMGraphAttention for more details.
 * \param aggregation_opt The aggregation strategy to be used. @see iftGFLIMAggregationType for more details.
 * \return The dataset with the seeds.
 */
iftDataSet *iftGFLIMSelectClusterCenters_perGraph(iftFLIMGraph *graph,
                                            iftLabeledSet *S,
                                            int kernel_size,
                                            int nclusters_per_marker,
                                            iftFLIMGraphAttention attention_opt,
                                            iftGFLIMAggregationType aggregation_opt)
{
#ifdef IFT_DEBUG
  printf("%s \n", __func__);
  if (graph == NULL)
    printf("\t graph: NULL \n");
  else
    printf("\t graph nodes: %ld, feats: %ld, edges: %ld \n", graph->num_nodes, graph->num_feats, graph->num_edges);
  if (S == NULL)
    printf("\t S: NULL \n");
  else
    printf("\t S size: %d \n", iftLabeledSetSize(S));
  printf("\t kernel_size: %d\n", kernel_size);
  printf("\t nclusters_per_marker: %d\n", nclusters_per_bkg_marker);
  printf("\t attention_opt: %d\n", attention_opt);
  printf("\t aggregation_opt: %d\n", aggregation_opt);
#endif

  int obj_label_map[graph->num_nodes];
  // relabel the seeds based on their connected components (markers)
  iftGFLIMRelabelConnectedComponentsInPlace(graph, S, obj_label_map); 

  // extract neighbor information and compute the iftDataSet
  iftDataSet *Z = iftGFLIMExtractMarkerDataSet(graph, S, kernel_size, attention_opt, aggregation_opt);
  iftDataSet *Z_out = iftCreateDataSet(Z->nclasses*nclusters_per_marker, Z->nfeats); // assumes that the number of kernels is the same for foreground and background

  int k = 0; // index for the output dataset
  int max_class = 0; // get the number of classes for the output dataset

  /* Use kmeans to reduce the number of seeds */
  for (int c = 1; c <= Z->nclasses; c++)
  {
    iftDataSet *Z1;
    Z1 = iftExtractSamplesFromClass(Z, c);
    
    int ngroups = iftMin(Z1->nsamples, nclusters_per_marker);
    iftKMeans(Z1, ngroups, 100, 0.001, NULL, NULL, iftEuclideanDistance);
    for (int s = 0; s < Z1->nsamples; s++)
    {
      if (iftHasSampleStatus(Z1->sample[s], IFT_PROTOTYPE))
      {
        Z_out->sample[k].id = Z1->sample[s].id;
        Z_out->sample[k].truelabel = obj_label_map[Z1->sample[s].id];
        if (Z_out->sample[k].truelabel > max_class)
          max_class = Z_out->sample[k].truelabel;
        for (int f = 0; f < Z1->nfeats; f++)
          Z_out->sample[k].feat[f] = Z1->sample[s].feat[f];
        k++;
      }
    }
    iftDestroyDataSet(&Z1);
  }
  Z_out->nclasses = max_class;
  iftDestroyDataSet(&Z);
  return Z_out;
}

/**
 * \brief Perform a k-means per label on Z and selects the seeds closest to the cluster centers.
 * \param Z The dataset with the seeds for all graphs.
 * \param ngraphs The number of graphs.
 * \param nkernels_per_label The number of kernels per label.
 * \return The labeled set after k-means.
 */
iftLabeledSet **iftGFLIMSelectClusterCenters_allGraphs(iftDataSet **Z,
                                              int ngraphs,
                                              int nkernels_per_label)
{
#ifdef IFT_DEBUG
  printf("%s \n", __func__);
  if (Z == NULL)
    printf("\t Z: NULL \n");
  printf("\t ngraphs: %d \n", ngraphs);
  for(int i = 0; i < ngraphs; i++)
  {
    if (Z[i] == NULL)
      printf("\t Z[%d]: NULL \n", i);
    else
      printf("\t Z[%d] nsamples: %d, nclasses: %d \n", i, Z[i]->nsamples, Z[i]->nclasses);
  }
  printf("\t nkernels_per_label: %d\n", nkernels_per_label);
#endif

  iftLabeledSet **T = iftAlloc(ngraphs, sizeof(iftLabeledSet *));
  
  /* Perform a k-means per label */
  for (int c = 0; c <= 1; c++)
  {
    // count the number of seeds with label c
    int nsamples = 0;
    for (int i = 0; i < ngraphs; i++)
    {
      iftDataSet *Z2 = iftExtractSamplesFromClass(Z[i], c);
      nsamples += Z2->nsamples;
      iftDestroyDataSet(&Z2);
    }
    
    int seeds_map[nsamples]; // map from Z1 index to seed value 
    int k = 0; // index for Z1

    /* Create a new dataset with the seeds with label c from all graphs */
    iftDataSet *Z1 = iftCreateDataSet(nsamples, Z[0]->nfeats);
    for(int i = 0; i < ngraphs; i++)
    {
      iftDataSet *Z2 = iftExtractSamplesFromClass(Z[i], c);
      for (int s = 0; s < Z2->nsamples; s++)
      {
        seeds_map[k] = Z2->sample[s].id;
        Z1->sample[k].id = k;
        Z1->sample[k].truelabel = i;
        
        for (int f = 0; f < Z2->nfeats; f++)
          Z1->sample[k].feat[f] = Z2->sample[s].feat[f];
        k++;
      }
      iftDestroyDataSet(&Z2);
    }

    /* perform k-means and put the seeds closest to the centers in T */
    int ngroups = iftMin(Z1->nsamples, nkernels_per_label);
    iftKMeans(Z1, ngroups, 100, 0.001, NULL, NULL, iftEuclideanDistance);
    for (int s = 0; s < Z1->nsamples; s++)
    {
      if (iftHasSampleStatus(Z1->sample[s], IFT_PROTOTYPE))
      {
        int graph_id = Z1->sample[s].truelabel;
        int p = seeds_map[Z1->sample[s].id];
        iftInsertLabeledSetMarkerAndHandicap(&T[graph_id], p,
                                                     c,
                                                     c,
                                                     0);
      }
    }
    iftDestroyDataSet(&Z1);
  }
  return T;
}

int main(int argc, char *argv[]) {
    timer *tstart;
    
    if (argc != 8)
      iftError("Usage: iftReduceSeedSet <P1> <P2> <P3> <P4> <P5> <P6> <P7> \n"
	       "P1: model folder\n"
	       "P2: image markers folder \n"
         "P3: number of feature points per marker \n"
         "P4: kernel size \n"
         "P5: number of feature points per label \n"
         "P6: attention option (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE) \n"
         "P7: aggregation strategy (CONCAT, MEAN, WMEAN) \n",
	       "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    char *image_markers_dir = argv[2];
    int nfpts_per_marker = atoi(argv[3]); 
    int kernel_size = atoi(argv[4]);
    int nfpts_per_label = atoi(argv[5]);
    iftFLIMGraphAttention attention_opt = iftGLIMGraphGetAttentionType(argv[6]);
    iftGFLIMAggregationType aggregation_opt = iftGLIMGraphGetAggregationType(argv[7]);
    
    if(!iftDirExists(model_dir))
      iftError("Model directory does not exist: %s", "main", model_dir);
    if(!iftDirExists(image_markers_dir))
      iftError("Image markers directory does not exist: %s", "main", image_markers_dir);

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(image_markers_dir,"-seeds.txt", 1);
    if(fs->n < 1)
      iftError("No files found in the markers directory: %s", "main", image_markers_dir);
    
    char graph_dir[200], labels_dir[200], out_dir[200];
    
    sprintf(graph_dir, "%s/layer0", model_dir);
    if(!iftDirExists(graph_dir))
      iftError("Graph directory does not exist: %s", "main", graph_dir);

    sprintf(labels_dir, "%s/labels0", model_dir);
    if(!iftDirExists(labels_dir))
      iftError("Labels directory does not exist: %s", "main", labels_dir);

    sprintf(out_dir, "%s/markers/", model_dir);
    iftMakeDir(out_dir);

    iftDataSet **Z = iftAlloc(fs->n, sizeof(iftDataSet *));
    for (int i = 0; i < fs->n; i++)
    {
        char filename[512];
        char *basename = iftFilename(fs->files[i]->path, "-seeds.txt");
        
        // red the superpixel image
        sprintf(filename, "%s/%s%s", labels_dir, basename, ".pgm");
        if(!iftFileExists(filename)) iftError("Labels file %s does not exist", __func__, filename);
        iftImage *label_img = iftReadImageByExt(filename);
        
        iftLabeledSet *S = iftReadSeeds(label_img, fs->files[i]->path); // read the image-based seed set
        
        // read the graph
        sprintf(filename, "%s/%s.json", graph_dir, basename);
        if(!iftFileExists(filename)) iftError("Graph file %s does not exist", __func__, filename);
        iftFLIMGraph *graph = iftReadFLIMGraph(filename);

        // convert image seeds to graph seeds
        iftLabeledSet *graph_seeds = iftGFLIMImageMarkersToGraphMarkers(label_img, S, IFT_GFLIM_NOSTRIDE, label_img->xsize, label_img->ysize);
        iftDestroyLabeledSet(&S);

        // reduce the seed set with a k-means per marker
        Z[i] = iftGFLIMSelectClusterCenters_perGraph(graph, graph_seeds, kernel_size, nfpts_per_marker, attention_opt, aggregation_opt);
        iftDestroyLabeledSet(&graph_seeds);
        
        iftFree(basename);
        iftDestroyImage(&label_img);
        iftDestroyFLIMGraph(&graph);
    }

    // perform a k-means per label on Z and select the seeds closest to the cluster centers
    iftLabeledSet **T = iftGFLIMSelectClusterCenters_allGraphs(Z, fs->n, nfpts_per_label);

    // write the graph-based seeds
    for (int i = 0; i < fs->n; i++)
    {
      char filename[512];
      char *basename = iftFilename(fs->files[i]->path, "-seeds.txt");
      sprintf(filename, "%s/%s-seeds_graph.txt", out_dir, basename);
      iftWriteSeedsGraph(T[i], filename);
      iftDestroyLabeledSet(&T[i]);
      iftDestroyDataSet(&Z[i]);
      iftFree(basename);
    }
    iftFree(T);
    iftFree(Z);
    iftDestroyFileSet(&fs);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
