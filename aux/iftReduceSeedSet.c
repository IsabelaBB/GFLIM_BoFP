#include "ift.h"

#define color_space LABNorm2_CSPACE


iftLabeledSet *imageMarkersToGraphMarkers(iftImage *label_img, iftLabeledSet *S, int num_nodes)
{
    iftLabeledSet *graph_seeds = NULL;
    int nclasses = 0, minclass = IFT_INFINITY_INT;
    iftLabeledSet *tmp;

    tmp = S;
    while (tmp != NULL)
    {
        if(tmp->label > nclasses)
            nclasses = tmp->label;
        if(tmp->label < minclass)
            minclass = tmp->label;
        tmp = tmp->next;
    }

    if(minclass == 0) nclasses++;

    int **classes_count = iftAlloc(num_nodes, sizeof(int*));
    for(int i=0; i < num_nodes; i++)
        classes_count[i] = iftAlloc(nclasses, sizeof(int*));
    
    tmp = S;
    // count the number of pixels per graph node and per class (label)
    while (tmp != NULL)
    {
        classes_count[label_img->val[tmp->elem] - 1][tmp->label - minclass]++;
        tmp = tmp->next;
    }

    for (int p = 0; p < num_nodes; p++)
    {
        int max_class = 0;
        for(int c = 1; c < nclasses; c++)
        {
            if(classes_count[p][c] > classes_count[p][max_class])
                max_class = c;
        }
        iftUnionLabeledSetElemMarkerAndHandicap(&graph_seeds, p, max_class, max_class, 0);
    }
    
    for(int i=0; i < num_nodes; i++)
        iftFree(classes_count[i]);
    iftFree(classes_count);

    return graph_seeds;
}

/**
 * @brief Relabels the marker value of seeds in the iftLabeledSet.
 * @param graph The input graph where the seeds are defined.
 * @param S The set of seeds.
 * @param marker_map The map of markers.
 * @param obj_label_map The map of object labels.
 * @note The function changes the marker value of the seeds in the iftLabeledSet S to the new marker value.
 * @note The function also changes the marker_map and obj_label_map to map from the graph nodes to the marker values and object labels, respectively.
 * @warning marker_map and obj_label_map must be allocated before calling this function. Their size must be equal to the number of nodes in the graph and their initial values does not matter.
 */
void ReLabelSeedsGraphMarkersInPlace(iftFLIMGraph *graph, iftLabeledSet *S, int *marker_map, int *obj_label_map)
{
    iftLabeledSet *seed;
    iftDHeap *queue;
    double *markers = (double *)malloc((int)(graph->num_nodes) * sizeof(double));
    double max_label = (double)(graph->num_nodes) + 1.0; // max val in label_img
    queue = iftCreateDHeap((int)(graph->num_nodes), markers);

    if (markers == NULL)
        iftError("Error to allocate memory for markers", __func__);

    for (int i = 0; i < graph->num_nodes; i++){
        obj_label_map[i] = -1; // none
        markers[i] = -1;
        marker_map[i] = -1;
    }

    seed = S;
    while (seed != NULL)
    {
        int p = seed->elem;
        markers[p] = max_label+1;
        obj_label_map[p] = seed->label;
        iftInsertDHeap(queue, p);
        seed = seed->next;
    }

    double curr_label = 1.0;
    while (!iftEmptyDHeap(queue))
    {
        int node_index = iftRemoveDHeap(queue);

        if (markers[node_index] == max_label+1)
        { // unconnected seed
            markers[node_index] = curr_label;
            curr_label++;
        }
        double node_label = markers[node_index];

        iftSet *neigh = graph->nodes[node_index].adj;
        while (neigh != NULL)
        {
            // insert in heap neighbor nodes that are seeds 
            // and were not visited yet and 
            // and have the same label (foreground or background) as the current node
            if (markers[neigh->elem] == max_label+1 && 
                queue->color[neigh->elem] == IFT_GRAY && 
                obj_label_map[node_index] == obj_label_map[neigh->elem])
            {
                markers[neigh->elem] = node_label; // indicate connectivity - they belong to the same marker
                iftGoUpDHeap(queue, queue->pos[neigh->elem]);
            }
            neigh = neigh->next;
        }
    }

    /*
    Constroi um conjunto de sementes rotuladas
    */
    seed = S;
    while (seed != NULL)
    {
        seed->marker = (int)(markers[seed->elem]);
        marker_map[seed->elem] = seed->marker;
        seed = seed->next;
    }

    iftFree(markers);
    iftDestroyDHeap(&queue);
}

iftDataSet *ExtractMarkerDataSet(iftFLIMGraph *graph,
                                    iftLabeledSet *S,
                                    int kernel_size,
                                    bool most_similar)
{
    int nsamples = iftLabeledSetSize(S);
    int kernel_neighbors = iftMin(graph->num_nodes, kernel_size);
    int tensor_size = graph->num_feats * kernel_neighbors;
    iftDataSet *Z = iftCreateDataSet(nsamples, tensor_size);

    Z->nclasses = 0;
    int s = 0;
    iftLabeledSet *seed = S;
    while (seed != NULL)
    {
        int p = seed->elem;
        Z->sample[s].id = p;
        Z->sample[s].truelabel = seed->marker;
        if (Z->sample[s].truelabel > Z->nclasses)
            Z->nclasses = Z->sample[s].truelabel;

        iftIntArray *neighbors = iftGetKNeighborsFLIMGraph(p, graph, kernel_neighbors,
                                                        1, most_similar);

        int j = 0, x = 0;
        while (x < neighbors->n)
        {
            for (int i = 0; i < graph->num_feats; i++)
            {
                Z->sample[s].feat[j] = graph->nodes[neighbors->val[x]].feats[i];
                j++;
            }
            x++;
        }
        iftDestroyIntArray(&neighbors);

        // caso não tenha vizinhos suficientes, completa o kernel com zeros
        while (x < kernel_neighbors)
        {
            for (int i = 0; i < graph->num_feats; i++)
            {
                Z->sample[s].feat[j] = 0.0;
                j++;
            }
            x++;
        }
        s++;
        seed = seed->next;
    }

    iftSetStatus(Z, IFT_TRAIN);
    iftAddStatus(Z, IFT_SUPERVISED);
    return (Z);
}

/**
 * @brief Selects the cluster centers from the seeds in the image.
 * @param graph The input graph where the seeds are defined.
 * @param S The set of graph seeds.
 * @param kernel_size The kernel size for the feature extraction.
 * @param nclusters_per_bkg_marker The number of cluster centers per background marker.
 * @param nclusters_per_obj_marker The number of cluster centers per object marker.
 * @return The set of cluster centers.
 * @note The function assign a value to each connected set of labeled nodes in the marker field of the iftLabeledSet S.
 */
iftLabeledSet *SelectClusterCenters(iftFLIMGraph *graph, iftLabeledSet *S, 
                                    int kernel_size, int nclusters_per_bkg_marker, int nclusters_per_obj_marker)
{
    iftLabeledSet *T = NULL;
    int markers_map[graph->num_nodes];
    int obj_label_map[graph->num_nodes];
    ReLabelSeedsGraphMarkersInPlace(graph, S, markers_map, obj_label_map);

    // extract neighbor information and compute the iftDataSet
    iftDataSet *Z = ExtractMarkerDataSet(graph, S, kernel_size, true);

    /* Estimate the given number of cluster centers per marker */
    for (int c = 1; c <= Z->nclasses; c++)
    {
        iftDataSet *Z1 = iftExtractSamplesFromClass(Z, c);
        int nclusters_per_marker;
        /* verify the label of a first pixel from the current marker */
        if (obj_label_map[Z1->sample[0].id] == 0)
        { /* background marker */
            nclusters_per_marker = nclusters_per_bkg_marker;
        }
        else
        { /* object marker */
            nclusters_per_marker = nclusters_per_obj_marker;
        }
        int ngroups = iftMin(Z1->nsamples, nclusters_per_marker);
        iftKMeans(Z1, ngroups, 100, 0.001, NULL, NULL, iftEuclideanDistance);
        for (int s = 0; s < Z1->nsamples; s++)
        {
            if (iftHasSampleStatus(Z1->sample[s], IFT_PROTOTYPE))
            {
                int p = Z1->sample[s].id;
                iftInsertLabeledSetMarkerAndHandicap(&T, p,
                                                     obj_label_map[p],
                                                     markers_map[p],
                                                     0);
            }
        }
        iftDestroyDataSet(&Z1);
    }
    iftDestroyDataSet(&Z);
    return (T);
}

iftLabeledSet *reduceImageSeeds(iftLabeledSet *img_seeds, iftLabeledSet *graph_seeds, iftImage *label_img, int num_nodes)
{
    iftLabeledSet *reduced_seeds = NULL;
    iftLabeledSet *s;
    int obj_label_map[num_nodes];
    int marker_map[num_nodes];

    for(int i = 0; i < num_nodes; i++)
    {
        obj_label_map[i] = -1; // none
        marker_map[i] = -1;
    }

    // indicate which nodes are seeds
    s = graph_seeds;
    while (s != NULL)
    {
        int p = s->elem;
        obj_label_map[p] = s->label;
        marker_map[p] = s->marker;
        s = s->next;
    }

    // insert in the reduced set of image seeds only those in the reduced set of graph seeds
    s = img_seeds;
    while (s != NULL)
    {
        int p = s->elem;
        int node = label_img->val[p]-1;
        if (obj_label_map[node] != -1)
        {
            iftInsertLabeledSetMarkerAndHandicap(&reduced_seeds, p, obj_label_map[node], marker_map[node], 0);
        }
        s = s->next;
    }
    return reduced_seeds;
}

int main(int argc, char *argv[]) {
    timer *tstart;
    
    if (argc!=6)
      iftError("Usage: iftReduceSeedSet <P1> <P2> <P3> <P4> <P5>\n"
	       "P1: model folder\n"
	       "P2: image markers folder \n"
           "P3: kernel size \n"
	       "P4: number of feature points on background markers \n"
	       "P5: number of feature points on object markers \n",
	       "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    char *image_markers_dir = argv[2];
    int kernel_size = atoi(argv[3]);
    int nfpts_per_bkg_marker = atoi(argv[4]);
    int nfpts_per_obj_marker = atoi(argv[5]);    

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

    for (int i = 0; i < fs->n; i++)
    {
        char filename[512];
        char *basename = iftFilename(fs->files[i]->path, "-seeds.txt");

        sprintf(filename, "%s/%s%s", labels_dir, basename, ".pgm");
        iftImage *label_img = iftReadImageByExt(filename);
        
        iftLabeledSet *S = iftReadSeeds(label_img, fs->files[i]->path);
        
        sprintf(filename, "%s/%s.json", graph_dir, basename);
        iftFLIMGraph *graph = iftReadFLIMGraph(filename);

        // convert image seeds to graph seeds
        iftLabeledSet *graph_seeds = imageMarkersToGraphMarkers(label_img, S, graph->num_nodes);
        
        // outputs a set of seeds in the graph space
        iftLabeledSet *T = SelectClusterCenters(graph, graph_seeds, kernel_size, nfpts_per_bkg_marker, nfpts_per_obj_marker);
        iftDestroyLabeledSet(&graph_seeds);
        
        // Convert the seeds in T back to the image space. Therefore, outputs a subset of S
        iftLabeledSet *final_seeds = reduceImageSeeds(S, T, label_img, graph->num_nodes);
        sprintf(filename, "%s/%s-seeds.txt", out_dir, basename);
        iftWriteLabeledSet(final_seeds, label_img->xsize, label_img->ysize, label_img->zsize, filename);
        
        iftFree(basename);
        iftDestroyLabeledSet(&final_seeds);
        iftDestroyLabeledSet(&S);
        iftDestroyLabeledSet(&T);
        iftDestroyImage(&label_img);
        iftDestroyFLIMGraph(&graph);
    }
    iftDestroyFileSet(&fs);

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
    return (0);
}
