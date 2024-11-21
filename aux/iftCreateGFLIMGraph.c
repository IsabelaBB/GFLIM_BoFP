#include "iftCommon.h"
#include "iftFLIMGraph.h"
// cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug ..

void usage()
{
    printf(" Usage: \n");
    printf("iftCreateGFLIMGraph <P1> <P2> <P3> <P4> \n");
    printf("P1: images diretory \n");
    printf("P2: number of initial seeds \n");
    printf("P3: number of superpixels \n");
    printf("P4: model folder \n");
}

int main(int argc, char *argv[])
{
#ifdef IFT_DEBUG
   printf("function %s \n", __func__);
#endif

    if(argc != 5)
    {
        usage();
        return 1;
    }

    timer *tstart = iftTic();
    char *images_dir = argv[1];
    int initial_seeds = atoi(argv[2]);
    int final_spx = atoi(argv[3]);
    char *model_dir = argv[4];

    char graphs_dir[200], labels_dir[200];
    sprintf(graphs_dir, "%s/layer0", model_dir);
    sprintf(labels_dir, "%s/labels0", model_dir);
    
    iftImageToFLIMGraphDISFDir(images_dir, labels_dir, graphs_dir, initial_seeds, final_spx);

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

   return 0;
}
