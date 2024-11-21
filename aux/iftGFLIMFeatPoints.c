#include "iftCommon.h"
#include "iftGFLIM.h"
// cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug ..

#define color_space LABNorm2_CSPACE

void usage()
{
    printf(" Usage: \n");
    printf("iftGFLIMFeatPoints <P1> <P2> \n");
    printf("P1: image markers directory \n");
    printf("P2: model directory \n");
}

void imageMarkersToGraphMarkers(char *labels_file, char *seeds_file, char *output_file)
{
    iftImage *label_img = iftReadImageByExt(labels_file);
    iftLabeledSet *S = iftReadSeeds(label_img, seeds_file);
    iftLabeledSet *graph_seeds = NULL;
    
    int num_nodes = iftMaximumValue(label_img);
    if(num_nodes < 1) iftError("Number of nodes must be positive: %d", __func__, num_nodes);

    int *foreground_count = (int *)calloc(num_nodes, sizeof(int));
    int *background_count = (int *)calloc(num_nodes, sizeof(int));

    iftLabeledSet *tmp = S;
    while (tmp != NULL)
    {
        int l = tmp->label;
        int p = tmp->elem;

        // define whether the superpixel is foreground or background
        if (l == 1)
            foreground_count[label_img->val[p] - 1]++;
        else
            background_count[label_img->val[p] - 1]++;

        tmp = tmp->next;
    }

    iftDestroyImage(&label_img);
    iftDestroyLabeledSet(&S);

    for (int p = 0; p < num_nodes; p++)
    {
        if (foreground_count[p] > 0 || background_count[p] > 0)
        {
            if (foreground_count[p] > background_count[p])
                iftUnionLabeledSetElemMarkerAndHandicap(&graph_seeds, p, 1, 1, 0);
            else
                iftUnionLabeledSetElemMarkerAndHandicap(&graph_seeds, p, 0, 0, 0);
        }
    }

    iftWriteSeedsGraph(graph_seeds, output_file);
    iftDestroyLabeledSet(&graph_seeds);
    free(foreground_count);
    free(background_count);
}

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        usage();
        return 1;
    }

    timer *tstart = iftTic();
    char *img_markers_dir = argv[1];
    char *model_dir = argv[2];    

    char labels_dir[200], graph_markers_dir[200];
    sprintf(labels_dir, "%s/labels0", model_dir);
    sprintf(graph_markers_dir, "%s/markers", model_dir);

    if (!iftDirExists(img_markers_dir))
        iftError("Image markers directory does not exist", __func__);
    if (!iftDirExists(labels_dir))
        iftError("Labels directory does not exist", __func__);
    if (strcmp(graph_markers_dir, "") == 0)
        iftError("Graph markers directory is NULL", __func__);
    
    iftMakeDir(graph_markers_dir);

    iftFileSet *fs_seeds = iftLoadFileSetFromDirBySuffix(img_markers_dir, "-seeds.txt", 1);
    if (fs_seeds->n < 1){
        iftError("Image markers directory is empty", __func__);
    }
    
    for (int i = 0; i < fs_seeds->n; i++)
    {
        char output_file[255], labels_file[255];
        char *basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        
        sprintf(labels_file, "%s/%s.pgm", labels_dir, basename);
        if(!iftFileExists(labels_file))
            iftError("Labels file %s does not exist", __func__, labels_file);
        sprintf(output_file, "%s/%s-seeds_graph.txt", graph_markers_dir, basename);

        imageMarkersToGraphMarkers(labels_file, fs_seeds->files[i]->path, output_file);
        iftFree(basename);
    }
    iftDestroyFileSet(&fs_seeds);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return 0;
}
