#include "ift.h"

/**
 * \file iftGFLIMCreateNeighbMatricesAndGraphs.c
 * \brief Create graphs from images and their neighborhood matrices for all layers
 * \note Used when stride_neighbors == "color", since the neighbors are computed 
 *      based on the color similarity of the superpixels, which does not change with the architecture. 
 *      In this case, we compute the neighbors matrices for all layers and save them in files, 
 *      so they can be reused during training and inference.
 * \author Isabela Borlido
 */

void usage()
{
    printf(" Usage: \n");
    printf("iftGFLIMCreateNeighbMatricesAndGraphs P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 \n");
    printf("P1: images diretory \n");
    printf("P2: number of superpixels \n");
    printf("P3: initial number of seeds \n");
    printf("P4: model folder \n");
    printf("P5: Graph feature. See iftGFLIMGraphNodeFeature at iftGFLIMGraph.h for details \n");
    printf("P6: Architecture file (json file) \n");
    printf("P7: use mask (boolean) \n");
    printf("P8: fixed number of superpixels (boolean) \n");
    printf("P9: kernel neighbors type (color, act) \n");
    printf("P10: pool neighbors type (color, act) \n");
    printf("P11: stride neighbors type (color, act) \n");
}

int main(int argc, char *argv[])
{
#ifdef IFT_DEBUG
    printf("function %s \n", __func__);
#endif

    srand(42);

    if (argc != 12)
    {
        usage();
        return 1;
    }

    timer *tstart = iftTic();
    char *images_dir = argv[1];
    float final_spx = atof(argv[2]);
    float initial_seeds = atof(argv[3]);
    char *model_dir = argv[4];
    iftGFLIMGraphNodeFeature graph_feat = iftGFLIMGraphGetNodeFeature(argv[5]);
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[6]);
    bool use_mask = atoi(argv[7]);
    bool fixed_spx = atoi(argv[8]);
    char *kernel_neighbors = argv[9];
    char *pool_neighbors = argv[10];
    char *stride_neighbors = argv[11];

    if (images_dir == NULL || (!iftDirExists(images_dir) && !iftFileExists(images_dir)))
        iftError("Image directory does not exist", __func__);
    if (final_spx <= 0)
        iftError("Number of superpixels must be positive", __func__);

    iftFileSet *fs_imgs = iftLoadFileSetFromDirOrCSV(images_dir, 1, false);
    if (fs_imgs->n < 1)
        iftError("No images found in the directory", __func__);

    char neighbors_path[1024];
    sprintf(neighbors_path, "%s/NeighborsMatrices", model_dir);
    if(iftDirExists(neighbors_path) == false)
        iftMakeDir(neighbors_path);

    char dir_name[1024];
    for(int l = 0; l <= arch->nlayers; l++)
    {
        sprintf(dir_name, "%s/layer%d", model_dir, l);
        if(!iftDirExists(dir_name))
            iftMakeDir(dir_name);
        sprintf(dir_name, "%s/labels%d", model_dir, l);
        if(!iftDirExists(dir_name))
            iftMakeDir(dir_name);
    }

    bool has_stride = false;
    if(strcmp(stride_neighbors, "color") == 0)
    {
        for(int l = 0; l < arch->nlayers; l++)
        {
            if(arch->layer[l].pool_stride > 1)
            {
                has_stride = true;
                break;
            }
        }
    }

    for (int i = 0; i < fs_imgs->n; i++)
    {
        char *basename = NULL;
        char filename[1024], ext[10];
        int final_spx_img = (int)final_spx;
        int initial_seeds_img = (int)initial_seeds;

        sprintf(ext, "%s", iftFileExt(fs_imgs->files[i]->path));
        basename = iftFilename(fs_imgs->files[i]->path, ext);

        iftImage *orig_image = iftReadImageByExt(fs_imgs->files[i]->path);

        iftImage *mask_img = NULL;
        int npixels = orig_image->n;
        if(use_mask)
        {
            strcpy(filename, fs_imgs->files[i]->path);
            GetMaskDirInplace(filename);
            mask_img = iftReadImageByExt(filename);

            npixels = 0;
            for(int p = 0; p < mask_img->n; p++)
            {
                if(mask_img->val[p] > 0)
                    npixels++; 
            }
        }

        if(initial_seeds < 1)
            initial_seeds_img = initial_seeds * npixels;
        if(final_spx < 1)
            final_spx_img = final_spx * npixels;

        float scale[3];
        scale[0] = 1.0; scale[1] = 1.0; scale[2] = 1.0;
        for(int l = 0; l < arch->nlayers; l++)
        {
            iftImage *label_img = NULL;
            iftGFLIMGraph *graph = iftImageToGFLIMGraph(orig_image, final_spx_img, initial_seeds_img, 
                                                    &label_img, graph_feat, IFT_FLIMGRAPH_TYPE_DISF,
                                                    mask_img);

            if(strcmp(kernel_neighbors, "color") == 0 || strcmp(pool_neighbors, "color") == 0)
            {
                int num_neighbors = iftMax(arch->layer[l].kernel_size, arch->layer[l].pool_size);
                iftMatrix *neighbors = iftGFLIMConstructGraphRelation(orig_image, label_img, graph_feat, num_neighbors);

                // write neighbors matrix in a file
                sprintf(filename, "%s/%s_%d.npy", neighbors_path, basename, l);
                iftWriteMatrix(neighbors, filename);
                iftDestroyMatrix(&neighbors);
            }

            // write the graph and label img
            sprintf(filename, "%s/layer%d/%s.json", model_dir, l, basename);
            iftWriteGFLIMGraph(graph, filename);
            sprintf(filename, "%s/labels%d/%s.pgm", model_dir, l, basename);
            iftWriteImageByExt(label_img, filename);
            iftDestroyImage(&label_img);
            iftDestroyGFLIMGraph(&graph);

            if(!has_stride)
                break;

            // if stride > 1, resize the image and mask for the next layer
            if (arch->layer[l].pool_stride > 1)
            {
                scale[0] *= (1.0f / (float)arch->layer[l].pool_stride);
                scale[1] *= (1.0f / (float)arch->layer[l].pool_stride);
                scale[2] *= (1.0f / (float)arch->layer[l].pool_stride);

                iftImage *orig_img_resized = iftInterp2D(orig_image, scale[0], scale[1]);
                iftDestroyImage(&orig_image);
                orig_image = orig_img_resized;
                orig_img_resized = NULL;

                int npixels_prev = npixels;
                npixels = orig_image->n;
                if(use_mask)
                {
                    iftImage *mask_img_resized = iftInterp2D(mask_img, scale[0], scale[1]);
                    iftDestroyImage(&mask_img);
                    mask_img = mask_img_resized;
                    mask_img_resized = NULL;

                    npixels = 0;
                    for(int p = 0; p < mask_img->n; p++)
                    {
                        if(mask_img->val[p] > 0)
                            npixels++; 
                    }
                }
                
                if(!fixed_spx)
                {
                    if(final_spx < 1)
                        final_spx_img = final_spx * npixels;
                    else
                        final_spx_img = (final_spx_img * npixels) / npixels_prev;
                    final_spx_img = iftMin(final_spx_img, npixels);

                    if(initial_seeds < 1)
                        initial_seeds_img = initial_seeds * npixels;
                    else
                        initial_seeds_img = (initial_seeds_img * npixels) / npixels_prev;
                    initial_seeds_img = iftMin(initial_seeds_img, npixels);
                }
            }
        }

        if(has_stride)
        {
            iftImage *label_img = NULL;
            iftGFLIMGraph *graph = iftImageToGFLIMGraph(orig_image, final_spx_img, initial_seeds_img, 
                                                        &label_img, graph_feat, IFT_FLIMGRAPH_TYPE_DISF,
                                                        mask_img);

            // write the graph and label img
            sprintf(filename, "%s/layer%d/%s.json", model_dir, arch->nlayers, basename);
            iftWriteGFLIMGraph(graph, filename);
            sprintf(filename, "%s/labels%d/%s.pgm", model_dir, arch->nlayers, basename);
            iftWriteImageByExt(label_img, filename);
            iftDestroyImage(&label_img);
            iftDestroyGFLIMGraph(&graph);
        }

        if(use_mask) iftDestroyImage(&mask_img);
        iftDestroyImage(&orig_image);
        iftFree(basename);
    }
    iftDestroyFileSet(&fs_imgs);
    iftDestroyGFLIMArch(&arch);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return 0;
}
