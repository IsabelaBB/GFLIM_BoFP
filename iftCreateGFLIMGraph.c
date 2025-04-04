/**
 * \file iftCreateGFLIMGraph.c
 * \brief Create graphs from images
 */

#include "iftCommon.h"
#include "iftFLIMGraph.h"

void usage()
{
    printf(" Usage: \n");
    printf("iftCreateGFLIMGraph <P1> <P2> <P3> <P4> <P5> [<P6>] \n");
    printf("P1: images diretory \n");
    printf("P2: number of superpixels \n");
    printf("P3: model folder \n");
    printf("P4: Graph feature (MEAN, SUM, SUM_MULT_RELSIZE, SUM_CONCAT_RELSIZE). See iftFLIMGraphFeature at iftGFLIMGraph.h for details \n");
    printf("P5: superpixel method (DISF, ERGC). See iftFLIMGraphType at iftGFLIMGraph.h for details \n");
    printf("P6: number of initial seeds (only used when P5=DISF) \n");
}

int main(int argc, char *argv[])
{
#ifdef IFT_DEBUG
   printf("function %s \n", __func__);
#endif

    if(argc != 7 && argc != 6)
    {
        usage();
        return 1;
    }

    timer *tstart = iftTic();
    char *images_dir = argv[1];
    int final_spx = atoi(argv[2]);
    char *model_dir = argv[3];
    iftFLIMGraphFeature graph_feat = iftGFLIMGetGraphFeature(argv[4]);
    iftFLIMGraphType spx_method = iftGFLIMGetGraphType(argv[5]);

    int initial_seeds = 0;
    if(spx_method == IFT_FLIMGRAPH_TYPE_DISF)
    {
        if(argc != 7)
            iftError("Number of initial seeds is required for DISF method", __func__);
        else initial_seeds = atoi(argv[6]);
    }

#ifdef IFT_DEBUG
   printArgs(images_dir, final_spx, model_dir, graph_feat, spx_method, initial_seeds);
#endif

    char graphs_dir[200], labels_dir[200];
    sprintf(graphs_dir, "%s/layer0", model_dir);
    sprintf(labels_dir, "%s/labels0", model_dir);
    
    switch (spx_method)
    {
    case IFT_FLIMGRAPH_TYPE_DISF: // segmentation using DISF
        iftImageToFLIMGraphDISFDir(images_dir, labels_dir, graphs_dir, initial_seeds, final_spx, graph_feat);
        break;
    
    case IFT_FLIMGRAPH_TYPE_ERGC: // segmentation using ERGC
        iftImageToFLIMGraphERGCDir(images_dir, labels_dir, graphs_dir, final_spx, 0, graph_feat);
        break;

    default:
        iftError("Invalid superpixel method %d", __func__, spx_method);
        break;
    }

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

   return 0;
}
