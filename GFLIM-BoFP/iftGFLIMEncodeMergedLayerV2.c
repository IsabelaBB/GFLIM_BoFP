#include "ift.h"

/**
 * \file iftGFLIMEncodeMergedLayer.c
 * \brief Encode a layer using the merged model created by iftGFLIMMergeLayerModels.
 */

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 8)
        iftError("Usage: iftGFLIMGSEncodeMergedLayer <P1> <P2> <P3> <P4> <P5> <P6> <P7> \n"
                 "[1] model directory \n"
                 "[2] architecture of the network (.json) \n"
                 "[3] layer number (1, 2, 3) \n"
                 "[4] stride type (none, image-based, graph-based) \n"
                 "[5] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE, ...) \n"
                 "[6] graph node feature (MEAN, SUM) \n"
                 "[7] original images path (used to compute node neighbors) --- it must have at least the training images \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[4]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[5]);
    iftGFLIMGraphNodeFeature node_feat = iftGFLIMGraphGetNodeFeature(argv[6]);
    char *orig_images_path = argv[7];
    
    if (layer < 1)
        iftError("Invalid layer number: %d", "main", layer);

    if (layer > arch->nlayers)
        iftError("Invalid layer number: %d", "main", layer);

    char input_dir[1024], labels_dir[1024], output_dir[1024], out_labels_dir[1024];
    char *filename = iftAllocCharArray(1024);

    // set graphs input directory
    sprintf(input_dir, "%s/layer%d", model_dir, layer - 1);
    if (!iftDirExists(input_dir))
        iftError("Graph directory does not exist", "main");

    sprintf(output_dir, "%s/layer%d", model_dir, layer);
    iftMakeDir(output_dir);

    // set labels directory
    if (stride_type != IFT_GFLIM_NOSTRIDE)
    {
        sprintf(labels_dir, "%s/labels%d", model_dir, layer - 1);
        if (!iftDirExists(labels_dir))
            iftError("Labels directory does not exist", "main");
        sprintf(out_labels_dir, "%s/labels%d", model_dir, layer);
        iftMakeDir(out_labels_dir);
    }else{
        sprintf(labels_dir, "%s/labels0", model_dir);
        if (!iftDirExists(labels_dir))
            iftError("Labels directory does not exist", "main");
    }

    iftFileSet *fs = NULL;
    
    // iterate over the training graphs
    if(iftDirExists(input_dir))
        fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", 1);
    else{
        if(!iftFileExists(input_dir))
            iftError("Graph directory or file does not exist: %s", "main", input_dir);
        fs = iftLoadFileSetFromCSV(input_dir, false);
    }
    if (fs == NULL || fs->n < 1)
        iftError("Graph directory or file is empty or does not exist: %s", "main", input_dir);

    iftMatrix *Kmerged = NULL;
    float *bias_merged = NULL;
    int *true_labels = NULL;
    
    // read kernels and bias
    sprintf(filename, "%s/model/conv%d", model_dir, layer);
    iftGFLIMLoadMergedModel(filename, &Kmerged, &bias_merged);

    // read the kernel labels
    if (attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE) // several attention mechanisms use the kernel labels
    {
        sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer);
        if (!iftFileExists(filename))
            iftError("File %s does not exist", "main", filename);
        int nkernels = 0;
        true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
    }
    
    /* Encode layer with merged model */

    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, ".json");

        // read the graph and superpixel image
        iftGFLIMGraph *graph = iftReadGFLIMGraph(fs->files[i]->path);
        iftImage *input_labels = NULL, *output_labels = NULL;
        sprintf(filename, "%s/%s.pgm", labels_dir, basename);
        input_labels = iftReadImageByExt(filename);
        
        // read the original image
        sprintf(filename, "%s/%s.png", orig_images_path, basename);
        if (!iftFileExists(filename))
        {
            sprintf(filename, "%s/%s.jpg", orig_images_path, basename);
            if (!iftFileExists(filename))
            {
                iftDestroyGFLIMGraph(&graph);
                iftDestroyImage(&input_labels);
                iftError("Original image %s/%s.(png/jpg) does not exist", "main", orig_images_path, basename);
                iftFree(basename);
            }
        }
        iftImage *orig_image = iftReadImageByExt(filename);

        // compute the neighbors matrix based on the original image
        iftMatrix *neighbors = iftGFLIMConstructGraphRelation(orig_image, input_labels, 
                                                            node_feat, 
                                                            iftMax(arch->layer[layer - 1].kernel_size, arch->layer[layer - 1].pool_size));
        iftDestroyImage(&orig_image);

        // encode layer
        iftGFLIMGraph *output_graph = iftGFLIMEncodeLayerV2(graph, Kmerged, bias_merged, 
                                                        arch->layer[layer - 1], stride_type, 
                                                        input_labels, &output_labels, 
                                                        attention_opt, true_labels, 
                                                        neighbors);
        iftDestroyMatrix(&neighbors);
        
        iftDestroyImage(&input_labels);
        iftDestroyGFLIMGraph(&graph);

        /* save output */
        if (stride_type != IFT_GFLIM_NOSTRIDE)
        {
            sprintf(filename, "%s/%s.pgm", out_labels_dir, basename);
            iftWriteImageByExt(output_labels, filename);
            iftDestroyImage(&output_labels);
        }

        sprintf(filename, "%s/%s.json", output_dir, basename);
        iftWriteGFLIMGraph(output_graph, filename);
        iftDestroyGFLIMGraph(&output_graph);
        iftFree(basename);
    }

    if(true_labels != NULL) iftFree(true_labels);
    iftDestroyMatrix(&Kmerged);
    iftFree(bias_merged);
    iftFree(filename);
    iftDestroyFileSet(&fs);
    iftDestroyGFLIMArch(&arch);
    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
