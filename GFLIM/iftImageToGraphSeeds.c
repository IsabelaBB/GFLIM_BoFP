#include "ift.h"

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 3)
        iftError("Usage: iftCreateLayerModel P1 P2 \n"
                 "P1: input model directory\n"
                 "P2: image markers folder \n",
                 "main");

    tstart = iftTic();
    
    char *model_dir = argv[1];
    char *img_seeds_dir = argv[2];
    char graph_seeds_dir[200], labels_dir[200];

    if(!iftDirExists(model_dir)){
        iftError("Model directory missing or does not exist: %s", "main", model_dir);
    }

    if(!iftDirExists(img_seeds_dir)){
        iftError("Image seeds directory missing or does not exist: %s", "main", img_seeds_dir);
    }

    sprintf(labels_dir, "%s/labels0", model_dir);
    if(!iftDirExists(labels_dir)){
        iftError("Labels directory missing or does not exist: %s", "main", labels_dir);
    }

    sprintf(graph_seeds_dir, "%s/markers", model_dir);
    iftMakeDir(graph_seeds_dir);

    // convert seeds pixel to seeds graph and write in "model_dir/markers_graph1"
    iftSeedsPixelToGraphDir(labels_dir, img_seeds_dir, graph_seeds_dir);

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
