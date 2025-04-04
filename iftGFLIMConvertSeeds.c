/**
 * \file iftGFLIMConvertSeeds.c
 * \brief Convert image markers to graph markers. 
 * It also performs k-means on the graph markers if a number of feature points per marker > 0 is defined.
 */

#include "ift.h"

int main(int argc, char *argv[]) {
    timer *tstart;
    
    if (argc != 7 && argc != 4)
      iftError("Usage: iftReduceSeedSet <P1> <P2> <P3> <P4> <P5> <P6> \n"
	       "P1: model folder\n"
	       "P2: image markers folder \n"
         "P3: number of feature points per marker (if -1, do not perform k-means and P4, P5, and P6 are not required) \n"
         "P4: kernel size \n"
         "P5: attention option (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE, ...) \n"
         "P6: aggregation strategy (CONCAT, MEAN, WMEAN) \n"
         //"P7: dimensionality reduction option (none, pca, tsne) \n"
         //"P8: number of components for dimensionality reduction \n"
	       ,"main");

    tstart = iftTic();

    char *model_dir = argv[1];
    char *image_markers_dir = argv[2];
    int nfpts_per_marker = atoi(argv[3]);  
    int kernel_size = 0;
    iftFLIMGraphAttention attention_opt = IFT_FLIMGRAPH_ATTENTION_NONE;
    iftGFLIMAggregationType aggregation_opt = IFT_GFLIM_AGGREGATION_CONCAT;
    
    if(nfpts_per_marker == 0 || nfpts_per_marker < -1)
      iftError("Kernel size must be greater than 0: %d", __func__, nfpts_per_marker);
    
    if(nfpts_per_marker != -1)
    {
      if(argc != 7) iftError("Invalid number of arguments: %d", __func__, argc);
      else{
        kernel_size = atoi(argv[4]);
        attention_opt = iftGLIMGraphGetAttentionType(argv[5]);
        aggregation_opt = iftGLIMGraphGetAggregationType(argv[6]);
      }
    }

    iftGFLIMDimReductionType dimReduction = IFT_GFLIM_DIMREDUCTION_NONE;
    int ncomponents = 1;
    /*if(argc > 7) {
      dimReduction = iftGLFLIMGetDimReductionType(argv[7]);
      if(dimReduction != IFT_GFLIM_DIMREDUCTION_NONE)
      {
        if(argc < 9)
          iftError("Number of components must be provided for dimensionality reduction", "main");
        ncomponents = atoi(argv[8]);
        if(ncomponents < 1)
          iftError("Number of components must be greater than 0", "main");
      }
    }*/

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

    sprintf(out_dir, "%s/markers/", model_dir);
    iftMakeDir(out_dir);

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

        if(nfpts_per_marker != -1) // perform a k-means per marker
        { 
          // read the graph
          sprintf(filename, "%s/%s.json", graph_dir, basename);
          if(!iftFileExists(filename)) iftError("Graph file %s does not exist", __func__, filename);
          iftFLIMGraph *graph = iftReadFLIMGraph(filename);

          // perform a k-means per marker and reduce the number of seeds
          iftLabeledSet *T = iftGFLIMSelectClusterCenters(graph, graph_seeds, kernel_size, nfpts_per_marker, attention_opt, dimReduction, ncomponents, aggregation_opt);
          iftDestroyLabeledSet(&graph_seeds);
          graph_seeds = T;
          T = NULL;
          iftDestroyFLIMGraph(&graph);
        }
        
        // write the graph-based seeds
        sprintf(filename, "%s/%s-seeds_graph.txt", out_dir, basename);
        iftWriteSeedsGraph(graph_seeds, filename);
        iftDestroyLabeledSet(&graph_seeds);

        iftFree(basename);
        iftDestroyImage(&label_img);
    }
    iftDestroyFileSet(&fs);

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
    return (0);
}
