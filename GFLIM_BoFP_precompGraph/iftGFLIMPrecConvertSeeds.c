/**
 * \file iftGFLIMConvertSeeds.c
 * \brief Convert image-based markers to graph-based markers.
 * Optionally, it also performs k-means on the graph markers if a number of feature points per marker > 0 is defined.
 * \warning The markers coordinates must be in "*-seeds.txt" files.
 */

// usado quando kernel_neighbors == "color"

#include "ift.h"

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 6)
        iftError("Usage: iftReduceSeedSet <P1> <P2> <P3> <P4> <P5> \n"
                 "P1: model folder\n"
                 "P2: image markers folder \n"
                 "P3: number of feature points per marker (if -1, do not perform k-means and P4 value is ignored) \n"
                 "P4: kernel size \n"
                 "P5: aggregation option \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    char *image_markers_dir = argv[2];
    int nfpts_per_marker = atoi(argv[3]);
    int kernel_size = atoi(argv[4]);
    iftGFLIMAggregation aggregation_opt = iftGFLIMGraphGetAggregation(argv[5]);

    if (!iftDirExists(model_dir))
        iftError("Model directory does not exist: %s", "main", model_dir);

    iftFileSet *fs = NULL;
    if (iftDirExists(image_markers_dir))
        fs = iftLoadFileSetFromDirBySuffix(image_markers_dir, "-seeds.txt", 1);
    else
    {
        if (!iftFileExists(image_markers_dir))
            iftError("Markers file/folder %s does not exist", __func__, image_markers_dir);
        fs = iftLoadFileSetFromCSV(image_markers_dir, false);
    }

    if (fs == NULL || fs->n < 1)
        iftError("No files found in the markers directory: %s", "main", image_markers_dir);

    char filename[1024];

    if (nfpts_per_marker != -1)
    {
        sprintf(filename, "%s/layer0", model_dir);
        if (!iftDirExists(filename))
            iftError("Graph directory does not exist: %s", "main", filename);
    }

    sprintf(filename, "%s/labels0", model_dir);
    if (!iftDirExists(filename))
        iftError("Labels directory does not exist: %s", "main", filename);

    sprintf(filename, "%s/markers/", model_dir);
    iftMakeDir(filename);

    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, "-seeds.txt");

        // read the superpixel image
        sprintf(filename, "%s/labels0/%s%s", model_dir, basename, ".pgm");
        if (!iftFileExists(filename))
            iftError("Labels file %s does not exist", __func__, filename);
        iftImage *label_img = iftReadImageByExt(filename);

        // read the seed set
        iftLabeledSet *S = iftReadSeeds(label_img, fs->files[i]->path);

        // convert image seeds to graph seeds
        iftLabeledSet *graph_seeds = iftGFLIMImageMarkersToGraphMarkers(label_img, S,
                                                                        IFT_GFLIM_NOSTRIDE,
                                                                        label_img->xsize,
                                                                        label_img->ysize);
        iftDestroyLabeledSet(&S);

        if (nfpts_per_marker != -1) // perform a k-means per marker
        {
            // read the graph
            sprintf(filename, "%s/layer0/%s.json", model_dir, basename);
            if (!iftFileExists(filename))
                iftError("Graph file %s does not exist", __func__, filename);
            iftGFLIMGraph *graph = iftReadGFLIMGraph(filename);

            // read the neighbors matrix
            iftMatrix *neighbors = NULL;
            sprintf(filename, "%s/NeighborsMatrices/%s_0.npy", model_dir, basename);
            if (iftFileExists(filename))
                neighbors = iftReadMatrix(filename);
            else{
                neighbors = iftCreateMatrix(kernel_size, graph->num_nodes);
                for(int p = 0; p < graph->num_nodes; p++)
                {
                    iftIntArray *neighb_arr = iftFLIMGraphGetKNeighborsAggregation(p, graph, kernel_size, 
                                                                                1, true, 
                                                                                IFT_FLIMGRAPH_ATTENTION_NONE);
                    
                    for(int n = 0; n < kernel_size; n++)
                        iftMatrixElem(neighbors, n, p) = neighb_arr->val[n];
                    iftDestroyIntArray(&neighb_arr);
                }
            }
            // perform a k-means per marker and reduce the number of seeds
            iftLabeledSet *T = iftGFLIMSelectClusterCentersV3(graph, graph_seeds,
                                                              kernel_size, nfpts_per_marker,
                                                              neighbors, aggregation_opt);
            iftDestroyLabeledSet(&graph_seeds);
            graph_seeds = T;
            T = NULL;
            iftDestroyGFLIMGraph(&graph);
            iftDestroyMatrix(&neighbors);
        }

        // write the graph-based seeds
        sprintf(filename, "%s/markers/%s-seeds_graph.txt", model_dir, basename);
        iftWriteSeedsGraph(graph_seeds, filename);
        iftDestroyLabeledSet(&graph_seeds);

        iftFree(basename);
        iftDestroyImage(&label_img);
    }
    iftDestroyFileSet(&fs);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
