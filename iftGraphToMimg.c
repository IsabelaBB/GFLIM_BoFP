/**
 * \file iftGraphToMimg.c
 * \brief Convert GFLIM graphs to MImages (.mimg files)
 * \author Isabela Borlido
 */

#include "iftCommon.h"
#include "iftGFLIMGraph.h"


void usage()
{
    printf(" Usage: \n");
    printf("iftGraphToMimg P1 P2 P3 \n");
    printf("P1: graphs directory \n");
    printf("P2: superpixel images directory \n");
    printf("P3: output directory \n");
}

int main(int argc, char *argv[])
{
#ifdef IFT_DEBUG
    printf("function %s \n", __func__);
#endif

    srand(42);

    if (argc != 4)
    {
        usage();
        return 1;
    }

    timer *tstart = iftTic();
    char *graphs_dir = argv[1];
    char *spx_images_dir = argv[2];
    char *output_dir = argv[3];

    if (graphs_dir == NULL || (!iftDirExists(graphs_dir) && !iftFileExists(graphs_dir)))
        iftError("Graphs directory does not exist", __func__);
    if (spx_images_dir == NULL || (!iftDirExists(spx_images_dir) && !iftFileExists(spx_images_dir)))
        iftError("Superpixel images directory does not exist", __func__);

    iftFileSet *fs_graphs = iftLoadFileSetFromDirOrCSV(graphs_dir, 1, false);
    if (fs_graphs->n < 1)
        iftError("No graphs found in the directory", __func__);

    if (!iftDirExists(output_dir))
        iftMakeDir(output_dir);

    for (int i = 0; i < fs_graphs->n; i++)
    {
        char *basename = NULL;
        char filename[1024], ext[10];

        sprintf(ext, "%s", iftFileExt(fs_graphs->files[i]->path));
        basename = iftFilename(fs_graphs->files[i]->path, ext);

        iftGFLIMGraph *graph = iftReadGFLIMGraph(fs_graphs->files[i]->path);
        sprintf(filename, "%s/%s.pgm", spx_images_dir, basename);
        iftImage *spx_img = iftReadImageByExt(filename);

        iftMImage *mimg = iftGFLIMGraphToMImage(graph, spx_img);

        iftDestroyImage(&spx_img);
        iftDestroyGFLIMGraph(&graph);

        sprintf(filename, "%s/%s.mimg", output_dir, basename);
        iftWriteMImage(mimg, filename);
        iftDestroyMImage(&mimg);

        iftFree(basename);
    }
    iftDestroyFileSet(&fs_graphs);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return 0;
}
