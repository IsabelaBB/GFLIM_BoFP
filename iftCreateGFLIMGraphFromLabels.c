/**
 * \file iftCreateGFLIMGraph.c
 * \brief Create graphs from images
 */

#include "iftCommon.h"
#include "iftGFLIMGraph.h"

void usage()
{
    printf(" Usage: \n");
    printf("iftCreateGFLIMGraph <P1> <P2> <P3> <P4> \n");
    printf("P1: original images directory \n");
    printf("P2: superpixel images diretory \n");
    printf("P3: Graph feature (MEAN, SUM, SUM_MULT_RELSIZE, SUM_CONCAT_RELSIZE). See iftGFLIMGraphFeature at iftGFLIMGraph.h for details \n");
    printf("P4: graphs diretory \n");
    
}

int main(int argc, char *argv[])
{
#ifdef IFT_DEBUG
    printf("function %s \n", __func__);
#endif

    if (argc != 5)
    {
        usage();
        return 1;
    }

    timer *tstart = iftTic();
    char *images_dir = argv[1];
    char *spx_dir = argv[2];
    iftGFLIMGraphFeature graph_feat = iftGFLIMGetGraphFeature(argv[3]);
    char *graphs_dir = argv[4];
    

    if (images_dir == NULL || !iftDirExists(images_dir))
        iftError("Image directory does not exist", __func__);
    
    if (spx_dir == NULL || !iftDirExists(spx_dir))
        iftError("Superpixel directory does not exist", __func__);

    iftFileSet *fs_imgs = iftLoadFileSetFromDir(images_dir, 1);
    if (fs_imgs->n < 1)
        iftError("No images found in the directory", __func__);

    if (!iftDirExists(graphs_dir))
        iftMakeDir(graphs_dir);
    
    for (int i = 0; i < fs_imgs->n; i++)
    {
        char *basename = NULL;
        char filename[512], ext[10];

        sprintf(ext, "%s", iftFileExt(fs_imgs->files[i]->path));
        basename = iftFilename(fs_imgs->files[i]->path, ext);
        //printf("Processing image %s ...\n", basename);

        sprintf(filename, "%s/%s.pgm", spx_dir, basename);
        if (!iftFileExists(filename))
        {
            iftWarning("Superpixel image %s does not exist. Skipping...", __func__, filename);
            iftFree(basename);
            continue;
        }

        //printf("Reading label image %s ...\n", filename);
        iftImage *label_img = iftReadImageByExt(filename);
        //printf("Reading original image %s ...\n", fs_imgs->files[i]->path); 
        iftImage *img = iftReadImageByExt(fs_imgs->files[i]->path);

        iftMImage *mimg;
        // Use sqrt(2.0) for 8-neighborhood adjacency; for ERGC superpixels, use 1.0 for 4-neighborhood adjacency.
        // If processing ERGC superpixels, replace sqrtf(2.0) with 1.0 to ensure correct adjacency.
        iftAdjRel *A = iftCircular(sqrtf(2.0));
        iftImage *gray_img = NULL;
        iftGFLIMGraph *graph;

        //printf("Converting image to MImage ...\n");
        if (iftIsColorImage(img))
            mimg = iftImageToMImage(img, LABNorm_CSPACE);
        else
            mimg = iftImageToMImage(img, GRAYNorm_CSPACE);
        
        //printf("Shifting minimum label value to 1 ...\n");
        int min_node = iftMinimumValue(label_img);
        int max_node = iftMaximumValue(label_img);
        bool *spx_exist = (bool *) calloc(max_node + 1, sizeof(bool));
        //printf("Minimum label value: %d\n", min_node);
        for (int p = 0; p < label_img->n; p++)
        {
            spx_exist[label_img->val[p]-1] = true;
        }
        for (int p = 0; p < max_node+1; p++)
        {
            if (!spx_exist[p])
            {
                printf("Image %s, Superpixel %d does not exist. \n", basename, p);
            }
        }

        if(min_node != 1)
        {
            for (int p = 0; p < label_img->n; p++)
            {
                label_img->val[p] += (-min_node + 1);
            }
            sprintf(filename, "%s/%s.pgm", spx_dir, basename);
            //printf("Writing superpixel image %s ...\n", filename);
            iftWriteImageByExt(label_img, filename);
        }
        int num_superpixels = iftMaximumValue(label_img);
        //printf("Number of superpixels: %d\n", num_superpixels);

        //printf("Creating graph ...\n");
        switch (graph_feat)
        {
        case IFT_FLIMGRAPH_FEATURE_MEAN:
            graph = ComputeGraphFeatureMean(
                        num_superpixels, A, mimg, 
                        label_img);
            break;
        case IFT_FLIMGRAPH_FEATURE_SUM:
            graph = ComputeGraphFeatureSum(
                        num_superpixels, A, mimg,
                        label_img);
            break;
        case IFT_FLIMGRAPH_FEATURE_SUM_MULT_RELSIZE:
            graph = ComputeGraphFeatureSumMultRelsize(
                        num_superpixels, A, mimg,
                        label_img); 
            break;
        case IFT_FLIMGRAPH_FEATURE_MEAN_MULT_RELSIZE:
            graph = ComputeGraphFeatureMeanMultRelsize(
                        num_superpixels, A, mimg,
                        label_img);
            break;
        case IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE:
            graph = ComputeGraphFeatureMeanConcatRelsize(
                        num_superpixels, A, mimg,
                        label_img);
            break;
        case IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_NEIGHB:
            graph = ComputeGraphFeatureMeanConcatRelsizeNeighb(
                        num_superpixels, A, mimg,
                        label_img);
            break;
        case IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_HARALICK:
            gray_img = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    #pragma omp parallel for
            for (int i = 0; i < mimg->n; i++)
                gray_img->val[i] = mimg->val[i][0];
            graph = ComputeGraphFeatureMeanConcatRelsizeHaralick(
                        num_superpixels, A, mimg,
                        label_img, NULL);
            break;
        case IFT_FLIMGRAPH_FEATURE_SUM_MULT_RELSIZE_CONCAT_HARALICK:
            iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    #pragma omp parallel for
            for (int i = 0; i < mimg->n; i++)
                gray_img->val[i] = mimg->val[i][0];
            graph = ComputeGraphFeatureSumMultRelsizeConcatHaralick(
                        num_superpixels, A, mimg,
                        label_img, NULL);
            break;
        case IFT_FLIMGRAPH_FEATURE_SUM_CONCAT_HARALICK:
            iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    #pragma omp parallel for
            for (int i = 0; i < mimg->n; i++)
                gray_img->val[i] = mimg->val[i][0];
            graph = ComputeGraphFeatureSumConcatHaralick(
                        num_superpixels, A, mimg,
                        label_img, graph_feat, NULL);
            break;
        default:
            iftError("Invalid graph feature type %d", __func__, graph_feat);
        }

        sprintf(filename, "%s/%s.json", graphs_dir, basename);
        //printf("Writing graph at %s ...\n", filename);
        iftWriteFLIMGraph(graph, filename);

        if(gray_img != NULL)
        {
            iftDestroyImage(&gray_img);
        }
        iftDestroyAdjRel(&A);
        iftDestroyMImage(&mimg);
        iftDestroyImage(&label_img);
        iftDestroyFLIMGraph(&graph);
        iftDestroyImage(&img);
        
        iftFree(basename);
    }
    iftDestroyFileSet(&fs_imgs);

    char *elapsed_time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", elapsed_time);
    iftFree(elapsed_time);
    return 0;
}
