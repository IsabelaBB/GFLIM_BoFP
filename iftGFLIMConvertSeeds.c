/**
 * \file iftGFLIMConvertSeeds.c
 * \brief Convert image-based markers to graph-based markers. 
 * Optionally, it also performs k-means on the graph markers if a number of feature points per marker > 0 is defined. 
 * \warning The markers coordinates must be in "*-seeds.txt" files.
 */

#include "ift.h"

int main(int argc, char *argv[]) {
    timer *tstart;
    //printf("Converting image markers to graph markers ...\n");
    
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
      
    //printf("Reading the parameters ...\n");
    tstart = iftTic();

    //printf("Number of arguments: %d\n", argc);
    char *model_dir = argv[1];
    //printf("Model directory: %s\n", model_dir);
    char *image_markers_dir = argv[2];
    //printf("Image markers directory: %s\n", image_markers_dir);
    int nfpts_per_marker = atoi(argv[3]);  
    //printf("Number of feature points per marker: %d\n", nfpts_per_marker);
    int kernel_size = 0;
    //printf("Kernel size: %d\n", kernel_size);
    iftGFLIMGraphAttention attention_opt = IFT_FLIMGRAPH_ATTENTION_NONE;
    iftGFLIMAggregation aggregation_opt = IFT_GFLIM_AGGREGATION_CONCAT;
    
    if(nfpts_per_marker == 0 || nfpts_per_marker < -1)
      iftError("Kernel size must be greater than 0: %d", __func__, nfpts_per_marker);
    
    if(nfpts_per_marker != -1)
    {
      if(argc != 7) iftError("Invalid number of arguments: %d", __func__, argc);
      else{
        kernel_size = atoi(argv[4]);
        //printf("Kernel size: %d\n", kernel_size);
        attention_opt = iftGLIMGraphGetAttention(argv[5]);
        //printf("Attention option: %s\n", argv[5]);
        aggregation_opt = iftGLIMGraphGetAggregation(argv[6]);
        //printf("Aggregation option: %s\n", argv[6]);
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

    //printf("Loading the image markers files ...\n");
    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(image_markers_dir,"-seeds.txt", 1);
    //printf("Number of files found: %ld\n", fs->n);
    if(fs->n < 1)
      iftError("No files found in the markers directory: %s", "main", image_markers_dir);
    
    //printf("Processing %ld files ...\n", fs->n);
    //char graph_dir[200], labels_dir[200], out_dir[200];
    char filename[1024];
    
    if(nfpts_per_marker != -1){
      sprintf(filename, "%s/layer0", model_dir);
      if(!iftDirExists(filename))
        iftError("Graph directory does not exist: %s", "main", filename);
      //printf("Graph directory: %s\n", filename);
    }

    sprintf(filename, "%s/labels0", model_dir);
    if(!iftDirExists(filename))
      iftError("Labels directory does not exist: %s", "main", filename);
    //printf("Labels directory: %s\n", filename);

    sprintf(filename, "%s/markers/", model_dir);
    iftMakeDir(filename);
    //printf("Output directory: %s\n", filename);

    //printf("Converting image markers to graph markers ...\n");

    for (int i = 0; i < fs->n; i++)
    {
        //char filename[512];
        char *basename = iftFilename(fs->files[i]->path, "-seeds.txt");

        //printf("Processing %s ...\n", basename);
        
        // read the superpixel image
        sprintf(filename, "%s/labels0/%s%s", model_dir, basename, ".pgm");
        if(!iftFileExists(filename)) iftError("Labels file %s does not exist", __func__, filename);
        iftImage *label_img = iftReadImageByExt(filename);
        //printf("Label image: %s\n", filename);

        // read the seed set
        iftLabeledSet *S = iftReadSeeds(label_img, fs->files[i]->path);
        //printf("Number of image markers: %d\n", iftLabeledSetSize(S));
        
        // convert image seeds to graph seeds
        iftLabeledSet *graph_seeds = iftGFLIMImageMarkersToGraphMarkers(label_img, S, IFT_GFLIM_NOSTRIDE, label_img->xsize, label_img->ysize);
        iftDestroyLabeledSet(&S);
        //printf("Number of graph seeds: %d\n", iftLabeledSetSize(graph_seeds));

        if(nfpts_per_marker != -1) // perform a k-means per marker
        { 
          //printf("Reducing the number of seeds per marker to %d ...\n", nfpts_per_marker);
          // read the graph
          sprintf(filename, "%s/layer0/%s.json", model_dir, basename);
          if(!iftFileExists(filename)) iftError("Graph file %s does not exist", __func__, filename);
          iftGFLIMGraph *graph = iftReadGFLIMGraph(filename);
          
          //printf("Number of graph nodes and features: %lu %lu\n", graph->num_nodes, graph->num_feats);
          // perform a k-means per marker and reduce the number of seeds
          iftLabeledSet *T = iftGFLIMSelectClusterCenters(graph, graph_seeds, kernel_size, nfpts_per_marker, attention_opt, dimReduction, ncomponents, aggregation_opt);
          iftDestroyLabeledSet(&graph_seeds);
          graph_seeds = T;
          T = NULL;
          iftDestroyGFLIMGraph(&graph);
          //printf("Number of graph markers after k-means: %d\n", iftLabeledSetSize(graph_seeds));
        }
        
        //printf("Writing the seeds ...\n");
        // write the graph-based seeds
        sprintf(filename, "%s/markers/%s-seeds_graph.txt", model_dir, basename);
        iftWriteSeedsGraph(graph_seeds, filename);
        iftDestroyLabeledSet(&graph_seeds);
        //printf("Finished writing the seeds to %s\n", filename);

        iftFree(basename);
        iftDestroyImage(&label_img);
    }
    iftDestroyFileSet(&fs);
    //printf("Finished converting image markers to graph markers.\n");

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    //printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
