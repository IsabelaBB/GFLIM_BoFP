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

    if (argc != 5)
        iftError("Usage: iftGFLIMSelectKernels <P1> <P2> <P3> <P4> \n"
                 "P1: model folder\n"
                 "P2: image markers folder \n"
                 "P3: list of kernels to maintain \n"
                 "P4: image name \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    char *image_markers_dir = argv[2];
    char *maintain_kernel_list = argv[3];
    char *image_name = argv[4];

    char filename[1024];

    if (!iftDirExists(model_dir))
        iftError("Model directory does not exist: %s", "main", model_dir);

    sprintf(filename, "%s/markers/", model_dir);
    if (!iftDirExists(filename))
        iftError("Graph markers directory does not exist: %s", "main", filename);

    if (!iftDirExists(image_markers_dir))
        iftError("Image markers directory does not exist: %s", "main", image_markers_dir);

    // read the seed set
    sprintf(filename, "%s/markers/%s-seeds_graph.txt", model_dir, image_name);
    if (!iftFileExists(filename))
        iftError("Graph seeds file %s does not exist", __func__, filename);
    iftLabeledSet *graph_seeds = iftReadSeedsGraph(filename);

    // get kernel values to be removed (they may be unsorted)
    int ngraph_seeds = iftLabeledSetSize(graph_seeds);
    printf("Number of graph seeds: %d\n", ngraph_seeds);
    bool *maintain_kernels_set = (bool *)iftAlloc(ngraph_seeds, sizeof(bool));
    printf("Kernels to be maintained: ");
    int kernel, start = 0;
    do{
        kernel = getNextNumber(maintain_kernel_list, &start);
        if (kernel > 0){
            maintain_kernels_set[kernel-1] = true;
            if(kernel > ngraph_seeds)
                iftError("Kernel %d is greater than the number of graph seeds %d", __func__, kernel, ngraph_seeds);
            printf("%d ", kernel);
        }
    }while(kernel > 0);
    printf("\n");

    // read the superpixel image
    sprintf(filename, "%s/labels0/%s.pgm", model_dir, image_name);
    if (!iftFileExists(filename))
        iftError("Labels file %s does not exist", __func__, filename);
    iftImage *label_img = iftReadImageByExt(filename);

    // remove graph seeds and mark labeled seeds
    iftLabeledSet *prev = NULL;
    iftLabeledSet *s = graph_seeds;
    int nnodes = iftMaximumValue(label_img);
    int *labeled_superpixels = (int *)iftAlloc(nnodes, sizeof(int));
    int i = 0;
    iftLabeledSet *new_graph_seeds = NULL;
    while (s != NULL)
    {
        if (maintain_kernels_set[i])
        {
            labeled_superpixels[s->elem] = s->label+1;
            iftInsertLabeledSetMarkerAndHandicap(&new_graph_seeds, s->elem, s->label, s->marker, 0);
        }
        s = s->next;
        i++;
    }
    /*while (s != NULL)
    {
        if (!maintain_kernels_set[i])
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
                graph_seeds = s;
            }
        }else
        {
            labeled_superpixels[s->elem] = s->label+1;
            prev = s;
            s = s->next;
        }
        i++;
    }*/
    iftFree(maintain_kernels_set);
    iftDestroyLabeledSet(&graph_seeds);
    graph_seeds = new_graph_seeds;

    // ---> create a set of image markers, labeling all pixels from marked nodes (superpixels)

    // create a set of image markers
    iftLabeledSet *image_seeds = NULL;
    for(int p = 0; p < label_img->n; p++)
    {
        if(label_img->val[p] < 1) continue;
        int label = labeled_superpixels[label_img->val[p] - 1];
        if(label > 0)
            iftInsertLabeledSetMarkerAndHandicap(&image_seeds, p, label-1, -1, 0);
    }
    iftFree(labeled_superpixels);

    // save image seeds 
    sprintf(filename, "%s/markers_img", model_dir);
    iftMakeDir(filename);
    sprintf(filename, "%s/markers_img/%s-seeds.txt", model_dir, image_name);
    iftWriteSeeds(image_seeds, label_img, filename);
    iftDestroyLabeledSet(&image_seeds);

    // write the graph-based seeds
    sprintf(filename, "%s/markers/%s-seeds_graph.txt", model_dir, image_name);
    iftWriteSeedsGraph(graph_seeds, filename);
    iftDestroyLabeledSet(&graph_seeds);

    iftDestroyImage(&label_img);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
