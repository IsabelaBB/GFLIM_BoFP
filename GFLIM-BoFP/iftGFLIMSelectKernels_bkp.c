/**
 * \file iftGFLIMConvertSeeds.c
 * \brief Convert image-based markers to graph-based markers.
 * Optionally, it also performs k-means on the graph markers if a number of feature points per marker > 0 is defined.
 * \warning The markers coordinates must be in "*-seeds.txt" files.
 */

#include "ift.h"

int getNextNumber(char *str, int *start)
{
    char *p = str + *start;
    while (*p != '\0' && !isdigit(*p))
        p++;
    if (*p == '\0')
        return -1;
    int num = atoi(p);
    while (isdigit(*p))
        p++;
    *start = p - str;
    return num;
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 6 && argc != 9)
        iftError("Usage: iftReduceSeedSet <P1> <P2> <P3> <P4> <P5> <P6> \n"
                 "P1: model folder\n"
                 "P2: image markers folder \n"
                 "P3: number of feature points per marker (if -1, do not perform k-means and P6, P7, and P8 are not required) \n"
                 "P4: list of kernels to be removed \n"
                 "P5: image name \n",
                 "P6: kernel size \n"
                 "P7: attention option (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE, ...) \n"
                 "P8: aggregation strategy (CONCAT, MEAN, WMEAN) \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    char *image_markers_dir = argv[2];
    int nfpts_per_marker = atoi(argv[3]);
    char *rem_kernel_list = argv[4];
    char *image_name = argv[5];
    int kernel_size = 0;
    iftFLIMGraphAttention attention_opt = IFT_FLIMGRAPH_ATTENTION_NONE;
    iftGFLIMAggregationType aggregation_opt = IFT_GFLIM_AGGREGATION_CONCAT;
    iftGFLIMDimReductionType dimReduction = IFT_GFLIM_DIMREDUCTION_NONE;

    if (nfpts_per_marker == 0 || nfpts_per_marker < -1)
        iftError("Kernel size must be greater than 0: %d", __func__, nfpts_per_marker);

    if (nfpts_per_marker != -1)
    {
        if (argc != 9)
            iftError("Invalid number of arguments: %d", __func__, argc);
        else
        {
            kernel_size = atoi(argv[6]);
            attention_opt = iftGLIMGraphGetAttentionType(argv[7]);
            aggregation_opt = iftGLIMGraphGetAggregationType(argv[8]);
        }
    }

    if (!iftDirExists(model_dir))
        iftError("Model directory does not exist: %s", "main", model_dir);
    if (!iftDirExists(image_markers_dir))
        iftError("Image markers directory does not exist: %s", "main", image_markers_dir);

    char graph_dir[200], labels_dir[200], out_dir[200];

    if (nfpts_per_marker != -1)
    {
        sprintf(graph_dir, "%s/layer0", model_dir);
        if (!iftDirExists(graph_dir))
            iftError("Graph directory does not exist: %s", "main", graph_dir);
    }

    sprintf(labels_dir, "%s/labels0", model_dir);
    if (!iftDirExists(labels_dir))
        iftError("Labels directory does not exist: %s", "main", labels_dir);

    sprintf(out_dir, "%s/markers/", model_dir);
    iftMakeDir(out_dir);

    char filename[512];

    // read the superpixel image
    sprintf(filename, "%s/%s%s", labels_dir, image_name, ".pgm");
    if (!iftFileExists(filename))
        iftError("Labels file %s does not exist", __func__, filename);
    iftImage *label_img = iftReadImageByExt(filename);

    // read the seed set
    sprintf(filename, "%s/%s%s", image_markers_dir, image_name, "-seeds.txt");
    iftLabeledSet *S = iftReadSeeds(label_img, filename);

    // convert image seeds to graph seeds
    iftLabeledSet *graph_seeds = iftGFLIMImageMarkersToGraphMarkers(label_img, S, IFT_GFLIM_NOSTRIDE, label_img->xsize, label_img->ysize);

    if (nfpts_per_marker != -1) // perform a k-means per marker
    {
        // read the graph
        sprintf(filename, "%s/%s.json", graph_dir, image_name);
        if (!iftFileExists(filename))
            iftError("Graph file %s does not exist", __func__, filename);
        iftFLIMGraph *graph = iftReadFLIMGraph(filename);

        // perform a k-means per marker and reduce the number of seeds
        iftLabeledSet *T = iftGFLIMSelectClusterCenters(graph, graph_seeds, kernel_size, nfpts_per_marker, attention_opt, dimReduction, 1, aggregation_opt);
        iftDestroyLabeledSet(&graph_seeds);
        graph_seeds = T;
        T = NULL;
        iftDestroyFLIMGraph(&graph);
    }

    // remove the kernels from the graph_seeds
    int ngraph_seeds = iftLabeledSetSize(graph_seeds);
    bool *rem_kernels_set = (bool *)iftAlloc(ngraph_seeds, sizeof(bool));
    int kernel, start = 0;
    do{
        kernel = getNextNumber(rem_kernel_list, &start);
        if (kernel != -1)
            rem_kernels_set[kernel] = true;
        
    }while(kernel != -1);
    
    iftLabeledSet *prev = NULL;
    int i = ngraph_seeds; // seed_index
    bool *rem_spx = (bool *)iftAlloc(iftMaximumValue(label_img), sizeof(bool));
    bool *rem_marker = (bool *)iftAlloc(iftMaximumValue(label_img), sizeof(bool));
    iftLabeledSet *s = graph_seeds;
    while (s != NULL)
    {
        if (rem_kernels_set[i])
            rem_marker[s->marker] = true;
        s = s->next;
        i--;
    }

    s = graph_seeds;
    while (s != NULL)
    {
        if (rem_marker[s->marker])
        {
            rem_spx[s->elem] = true;
            if (prev != NULL)
            {
                prev->next = s->next;
                s->next = NULL;
                iftFree(s);
                s = prev->next;
            }
            else
            {
                prev = s->next;
                s->next = NULL;
                iftFree(s);
                s = prev;
                prev = NULL;
                graph_seeds = s;
            }
        }else
        {
            prev = s;
            s = s->next;
        }
        i--;
    }
    
    iftFree(rem_kernels_set);

    // remove the kernels from S
    s = S;
    prev = NULL;
    while (s != NULL)
    {
        if (rem_spx[label_img->val[s->elem] - 1])
        {
            if (prev != NULL)
            {
                prev->next = s->next;
                s->next = NULL;
                iftFree(s);
                s = prev->next;
            }
            else
            {
                prev = s->next;
                s->next = NULL;
                iftFree(s);
                s = prev;
                prev = NULL;
                S = s;
            }
        }
        else{
            prev = s;
            s = s->next;
        }
    }
    prev = NULL;
    iftFree(rem_spx);

    // save image seeds after removing kernels
    sprintf(filename, "%s/markers_red", model_dir);
    iftMakeDir(filename);
    sprintf(filename, "%s/markers_red/%s-seeds.txt", model_dir, image_name);
    iftWriteSeeds(S, label_img, filename);
    iftDestroyLabeledSet(&S);

    // write the graph-based seeds
    sprintf(filename, "%s/%s-seeds_graph.txt", out_dir, image_name);
    iftWriteSeedsGraph(graph_seeds, filename);
    iftDestroyLabeledSet(&graph_seeds);

    iftDestroyImage(&label_img);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
