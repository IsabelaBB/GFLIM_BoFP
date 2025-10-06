/**
 * \file iftCreateGFLIMGraph.c
 * \brief Create graphs from images
 */

#include "iftCommon.h"
#include "iftGFLIMGraph.h"

void usage()
{
    printf(" Usage: \n");
    printf("iftCreateGFLIMGraph <P1> <P2> <P3> <P4> <P5> [<P6>] \n");
    printf("P1: images diretory \n");
    printf("P2: number of superpixels \n");
    printf("P3: model folder \n");
    printf("P4: Graph feature (MEAN, SUM, SUM_MULT_RELSIZE, SUM_CONCAT_RELSIZE). See iftGFLIMGraphNodeFeature at iftGFLIMGraph.h for details \n");
    printf("P5: superpixel method (DISF, ERGC). See iftGFLIMGraphSpxMethod at iftGFLIMGraph.h for details \n");
    printf("P6: number of initial seeds (only used when P5=DISF) \n");
}

int main(int argc, char *argv[])
{
#ifdef IFT_DEBUG
    printf("function %s \n", __func__);
#endif

    if (argc != 7 && argc != 6)
    {
        usage();
        return 1;
    }

    timer *tstart = iftTic();
    char *images_dir = argv[1];
    int final_spx = atoi(argv[2]);
    char *model_dir = argv[3];
    iftGFLIMGraphNodeFeature graph_feat = iftGFLIMGraphGetNodeFeature(argv[4]);
    iftGFLIMGraphSpxMethod spx_method = iftGFLIMGraphGetSpxMethod(argv[5]);

    int initial_seeds = 0;
    if (spx_method == IFT_FLIMGRAPH_TYPE_DISF)
    {
        if (argc != 7)
            iftError("Number of initial seeds is required for DISF method", __func__);
        else
            initial_seeds = atoi(argv[6]);
    }

#ifdef IFT_DEBUG
    printArgs(images_dir, final_spx, model_dir, graph_feat, spx_method, initial_seeds);
#endif

    char graphs_dir[200], labels_dir[200];
    sprintf(graphs_dir, "%s/layer0", model_dir);
    sprintf(labels_dir, "%s/labels0", model_dir);

    if (images_dir == NULL || !iftDirExists(images_dir))
        iftError("Image directory does not exist", __func__);
    if (final_spx < 1)
        iftError("Number of superpixels must be positive", __func__);

    iftFileSet *fs_imgs = iftLoadFileSetFromDir(images_dir, 1);
    if (fs_imgs->n < 1)
        iftError("No images found in the directory", __func__);

    if (!iftDirExists(graphs_dir))
        iftMakeDir(graphs_dir);
    if (!iftDirExists(labels_dir))
        iftMakeDir(labels_dir);

    for (int i = 0; i < fs_imgs->n; i++)
    {
        char *basename = NULL;
        char filename[512], ext[10];

        sprintf(ext, "%s", iftFileExt(fs_imgs->files[i]->path));
        basename = iftFilename(fs_imgs->files[i]->path, ext);

        iftImage *label_img;
        iftImage *img = iftReadImageByExt(fs_imgs->files[i]->path);

        iftGFLIMGraph *graph = iftImageToGFLIMGraph(img, final_spx, initial_seeds, &label_img, graph_feat, spx_method);

        sprintf(filename, "%s/%s.pgm", labels_dir, basename);
        iftWriteImageByExt(label_img, filename);

        sprintf(filename, "%s/%s.json", graphs_dir, basename);
        iftWriteGFLIMGraph(graph, filename);

        iftDestroyImage(&label_img);
        iftDestroyGFLIMGraph(&graph);
        iftDestroyImage(&img);

        iftFree(basename);
    }
    iftDestroyFileSet(&fs_imgs);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return 0;
}
