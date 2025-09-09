/**
 * \file iftWriteSeedsVectors.c
 * \brief Write in txt files the feature vectors of the graph markers and the neighbors used
 * It does the same operations as in iftGFLIMConvertSeeds.c, but it only writes txt files with the feature vectors and neighbors.
 * \warning The image markers coordinates must be in "*-seeds.txt" files.
 */

#include "ift.h"

iftDataSet *iftGFLIMExtractMarkerDataSet_local(iftGFLIMGraph *graph,
                                         iftLabeledSet *S,
                                         int kernel_size,
                                         iftGFLIMGraphAttention attention_opt,
                                         iftGFLIMAggregationType aggregation_opt,
                                         iftMatrix *neighbor_matrix)
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
    printf("\t most_similar: %s\n", most_similar ? "true" : "false");
    printf("\t attention_opt: %d\n", attention_opt);
    printf("\t aggregation_opt: %d\n", aggregation_opt);
#endif

    if(neighbor_matrix == NULL)
        iftError("Neighbor matrix is NULL", __func__);

    int nsamples = iftLabeledSetSize(S);
    int kernel_neighbors = iftMin(graph->num_nodes, kernel_size);
    
    if(neighbor_matrix->nrows != nsamples)
        iftError("Neighbor matrix rows (%d) must match the number of nodes in the seed set (%d)", __func__, neighbor_matrix->nrows, nsamples);

    if(neighbor_matrix->ncols != kernel_neighbors)
        iftError("Neighbor matrix columns (%d) must match the number of neighbors (%d)", __func__, neighbor_matrix->ncols, kernel_neighbors);

    int tensor_size = aggregation_opt == IFT_GFLIM_AGGREGATION_CONCAT ? graph->num_feats * kernel_neighbors : 2 * graph->num_feats;
    iftDataSet *Z = iftCreateDataSet(nsamples, tensor_size);

    iftLabeledSet *seed;
    Z->nclasses = 0;
    int s = 0;
    seed = S;

    while (seed != NULL)
    {
        Z->sample[s].id = seed->elem;
        Z->sample[s].truelabel = seed->marker;
        if (Z->sample[s].truelabel > Z->nclasses)
            Z->nclasses = Z->sample[s].truelabel;

        iftIntArray *neighbors = iftGetKNeighborsFLIMGraph(seed->elem, graph, kernel_neighbors, 1, true, attention_opt);
        
        for(int i = 0; i < neighbors->n; i++) {
            iftMatrixElem(neighbor_matrix, i, s) = neighbors->val[i];
        }
        
        double *feature_vector = iftGFLIMGraphAggregateFeatures(seed->elem, neighbors, graph, kernel_size, aggregation_opt);

        for (int i = 0; i < tensor_size; i++)
            Z->sample[s].feat[i] = (float)feature_vector[i];

        iftFree(feature_vector);
        iftDestroyIntArray(&neighbors);

        s++;
        seed = seed->next;
    }

    iftSetStatus(Z, IFT_TRAIN);
    iftAddStatus(Z, IFT_SUPERVISED);
    return (Z);
}

int main(int argc, char *argv[]) {
    timer *tstart;
    
    if (argc != 7)
      iftError("Usage: iftReduceSeedSet <P1> <P2> <P3> <P4> <P5> <P6> \n"
	       "P1: model folder\n"
	       "P2: image markers folder \n"
         "P3: number of feature points per marker \n"
         "P4: kernel size \n"
         "P5: attention option (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE, ...) \n"
         "P6: aggregation strategy (CONCAT, MEAN, WMEAN) \n"
	       ,"main");

    tstart = iftTic();

    char *model_dir = argv[1];
    char *image_markers_dir = argv[2];
    int nfpts_per_marker = atoi(argv[3]);  
    int kernel_size = 0;
    iftGFLIMGraphAttention attention_opt = IFT_FLIMGRAPH_ATTENTION_NONE;
    iftGFLIMAggregationType aggregation_opt = IFT_GFLIM_AGGREGATION_CONCAT;
    
    if(nfpts_per_marker < 0)
      iftError("Kernel size must be greater than 0: %d", __func__, nfpts_per_marker);
    
    kernel_size = atoi(argv[4]);
    attention_opt = iftGLIMGraphGetAttentionType(argv[5]);
    aggregation_opt = iftGLIMGraphGetAggregationType(argv[6]);
    
    iftGFLIMDimReductionType dimReduction = IFT_GFLIM_DIMREDUCTION_NONE;
    int ncomponents = 1;

    if(!iftDirExists(model_dir))
      iftError("Model directory does not exist: %s", "main", model_dir);
    if(!iftDirExists(image_markers_dir))
      iftError("Image markers directory does not exist: %s", "main", image_markers_dir);

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(image_markers_dir,"-seeds.txt", 1);
    if(fs->n < 1)
      iftError("No files found in the markers directory: %s", "main", image_markers_dir);
    
    char graph_dir[200], labels_dir[200], out_dir[200];
    
    if(nfpts_per_marker != -1){
      sprintf(graph_dir, "%s/layer0", model_dir);
      if(!iftDirExists(graph_dir))
        iftError("Graph directory does not exist: %s", "main", graph_dir);
    }

    sprintf(labels_dir, "%s/labels0", model_dir);
    if(!iftDirExists(labels_dir))
      iftError("Labels directory does not exist: %s", "main", labels_dir);

    sprintf(out_dir, "%s/features/", model_dir);
    iftMakeDir(out_dir);

    char out_dir2[512];
    sprintf(out_dir2, "%s/neighbors/", model_dir);
    iftMakeDir(out_dir2);
    
    for (int i = 0; i < fs->n; i++)
    {
        char filename[512];
        char *basename = iftFilename(fs->files[i]->path, "-seeds.txt");
        
        // read the superpixel image
        sprintf(filename, "%s/%s%s", labels_dir, basename, ".pgm");
        if(!iftFileExists(filename)) iftError("Labels file %s does not exist", __func__, filename);
        iftImage *label_img = iftReadImageByExt(filename);
        
        // read the seed set
        iftLabeledSet *S = iftReadSeeds(label_img, fs->files[i]->path);
        
        // convert image seeds to graph seeds
        iftLabeledSet *graph_seeds = iftGFLIMImageMarkersToGraphMarkers(label_img, S, IFT_GFLIM_NOSTRIDE, label_img->xsize, label_img->ysize);
        iftDestroyLabeledSet(&S);
        iftDestroyImage(&label_img);

        // read the graph
        sprintf(filename, "%s/%s.json", graph_dir, basename);
        if(!iftFileExists(filename)) iftError("Graph file %s does not exist", __func__, filename);
        iftGFLIMGraph *graph = iftReadFLIMGraph(filename);
        
        int obj_label_map[graph->num_nodes];
        iftGFLIMRelabelConnectedComponentsInPlace(graph, graph_seeds, obj_label_map);

        // extract neighbor information and compute the iftDataSet
        iftMatrix *neighbor_matrix = iftCreateMatrix(kernel_size, iftLabeledSetSize(graph_seeds));
        iftDataSet *Z = iftGFLIMExtractMarkerDataSet_local(graph, graph_seeds, kernel_size, attention_opt, aggregation_opt, neighbor_matrix);

        iftDestroyLabeledSet(&graph_seeds);
        iftDestroyFLIMGraph(&graph);
        
        // write the dataset
        sprintf(filename, "%s/%s-seeds_vectors.txt", out_dir, basename);
        FILE *fptr = fopen(filename, "w"); 
        for(int j = 0; j < Z->nsamples; j++) {
          fprintf(fptr, "%d %d ", obj_label_map[Z->sample[j].id], Z->sample[j].truelabel);
          for(int k = 0; k < Z->nfeats; k++) {
            fprintf(fptr, "%f ", Z->sample[j].feat[k]);
          }
          fprintf(fptr, "\n");
        }
        fclose(fptr);
        iftDestroyDataSet(&Z);

        // write the neighbor matrix
        sprintf(filename, "%s/%s-neighbors.txt", out_dir2, basename);
        fptr = fopen(filename, "w"); 
        for(int j = 0; j < neighbor_matrix->nrows; j++) {
          for(int k = 0; k < neighbor_matrix->ncols; k++) {
            fprintf(fptr, "%d ", (int)iftMatrixElem(neighbor_matrix, k, j));
          }
          fprintf(fptr, "\n");
        }
        fclose(fptr);
        iftDestroyMatrix(&neighbor_matrix);

        iftFree(basename);
    }
    iftDestroyFileSet(&fs);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
