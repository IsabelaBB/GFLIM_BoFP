/**
 * \file iftCreateGFLIMGraph.c
 * \brief Create graphs from images
 * \details Given a directory with images, create graphs using superpixels
 *          and save them in a specified model directory.
 * \author Isabela Borlido
 */

#include "iftCommon.h"
#include "iftGFLIMGraph.h"
#include "iftGFLIM.h"

void usage()
{
    printf(" Usage: \n");
    printf("iftCreateGFLIMGraph <P1> <P2> <P3> <P4> <P5> [<P6> <P7>] \n");
    printf("P1: images diretory \n");
    printf("P2: number of superpixels \n");
    printf("P3: model folder \n");
    printf("P4: Graph feature (MEAN, SUM, SUM_MULT_RELSIZE, SUM_CONCAT_RELSIZE). See iftGFLIMGraphNodeFeature at iftGFLIMGraph.h for details \n");
    printf("P5: superpixel method (DISF, ERGC). See iftGFLIMGraphSpxMethod at iftGFLIMGraph.h for details \n");
    printf("P6: number of initial seeds (only used when P5=DISF) \n");
    printf("P7: bool use mask (optional) --- used only if P5=DISF \n");
}

int main(int argc, char *argv[])
{
#ifdef IFT_DEBUG
    printf("function %s \n", __func__);
#endif

    srand(42);

    if (argc != 7 && argc != 6 && argc != 8)
    {
        usage();
        return 1;
    }

    timer *tstart = iftTic();
    char *images_dir = argv[1];
    float final_spx = atof(argv[2]);
    char *model_dir = argv[3];
    iftGFLIMGraphNodeFeature graph_feat = iftGFLIMGraphGetNodeFeature(argv[4]);
    iftGFLIMGraphSpxMethod spx_method = iftGFLIMGraphGetSpxMethod(argv[5]);
    bool use_mask = false;

    float initial_seeds = 0;
    if (spx_method == IFT_FLIMGRAPH_TYPE_DISF)
    {
        if (argc < 7)
            iftError("Number of initial seeds is required for DISF method", __func__);
        else
            initial_seeds = atof(argv[6]);
    }

    if(argc == 8)
    {
        if (spx_method != IFT_FLIMGRAPH_TYPE_DISF)
            iftError("Mask image is only supported for DISF method", __func__);
        use_mask = atoi(argv[7]);
    }

#ifdef IFT_DEBUG
    printArgs(images_dir, final_spx, model_dir, graph_feat, spx_method, initial_seeds);
#endif

    char graphs_dir[1024], labels_dir[1024];
    sprintf(graphs_dir, "%s/layer0", model_dir);
    sprintf(labels_dir, "%s/labels0", model_dir);

    if (images_dir == NULL || (!iftDirExists(images_dir) && !iftFileExists(images_dir)))
        iftError("Image directory does not exist", __func__);
    if (final_spx < 1)
        iftError("Number of superpixels must be positive", __func__);

    iftFileSet *fs_imgs = iftLoadFileSetFromDirOrCSV(images_dir, 1, false);
    if (fs_imgs->n < 1)
        iftError("No images found in the directory", __func__);

    if (!iftDirExists(graphs_dir))
        iftMakeDir(graphs_dir);
    if (!iftDirExists(labels_dir))
        iftMakeDir(labels_dir);

    for (int i = 0; i < fs_imgs->n; i++)
    {
        char *basename = NULL;
        char filename[1024], ext[10];
        int final_spx_img = (int)final_spx;
        int initial_seeds_img = (int)initial_seeds;

        sprintf(ext, "%s", iftFileExt(fs_imgs->files[i]->path));
        basename = iftFilename(fs_imgs->files[i]->path, ext);

        iftImage *img = iftReadImageByExt(fs_imgs->files[i]->path);

        iftImage *mask = NULL;
        int npixels = img->n;
        if(use_mask)
        {
            strcpy(filename, fs_imgs->files[i]->path);
            GetMaskDirInplace(filename);
            mask = iftReadImageByExt(filename);

            npixels = 0;
            for(int p = 0; p < mask->n; p++)
            {
                if(mask->val[p] > 0)
                    npixels++; 
            }
        }
        
        if(final_spx < 1)
            final_spx_img = (int)(npixels * final_spx);
        if(initial_seeds < 1)
            initial_seeds_img = (int)(npixels * initial_seeds);

        //printf("Train img: %s, spx: %d, seeds: %d\n", basename, final_spx_img, initial_seeds_img);
        iftImage *label_img = NULL;
        iftGFLIMGraph *graph = iftImageToGFLIMGraph(img, final_spx_img, initial_seeds_img, 
                                                    &label_img, graph_feat, spx_method,
                                                    mask);
        
        /*if(strcmp(basename, "000002_00000014") == 0)
        {
            printf("Image %s, num nodes: %d, num edges: %d, num feats: %d\n", 
                basename, graph->num_nodes, graph->num_edges, graph->num_feats);
            printf("Node 0: ");
            for(int f = 0; f < graph->num_feats; f++)
                printf("%f ", graph->nodes[0].feats[f]);
            printf("\n");
        }*/

        if(use_mask) iftDestroyImage(&mask);
        
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
