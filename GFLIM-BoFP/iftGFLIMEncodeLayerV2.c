#include "ift.h"

/**
 * \file iftGFLIMGSEncodeMergedLayer.c
 * \brief Encode a layer using the non-merged kernels, bias, and labels created by iftGFLIMCreateLayerModel
 * \warning This function only encode feature from the training graphs.
 */

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 8)
        iftError("Usage: iftGFLIMGSEncodeLayer <P1> <P2> <P3> <P4> <P5> <P6> <P7>\n"
                 "[1] model directory \n"
                 "[2] architecture of the network (.json) \n"
                 "[3] layer number (1, 2, 3) \n"
                 "[4] stride type (none, image-based, graph-based) \n"
                 "[5] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE) \n"
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

    // set graph markers directory
    sprintf(filename, "%s/markers", model_dir);
    if (!iftDirExists(filename))
        iftError("Markers directory does not exist", "main");
    
    // set graphs output directory
    sprintf(output_dir, "%s/layer%d", model_dir, layer);
    iftMakeDir(output_dir);

    // iterate over the training samples based on the graph marker files
    iftFileSet *fs = NULL;
    if(iftDirExists(filename))
      fs = iftLoadFileSetFromDirBySuffix(filename, "-seeds_graph.txt", 1);
    else{
        if(!iftFileExists(filename))
            iftError("Graph markers directory or file does not exist: %s", "main", filename);
        fs = iftLoadFileSetFromCSV(filename, false);
    }
    if(fs == NULL || fs->n < 1)
        iftError("Graph markers directory is empty: %s", "main", filename);

    /* Encode layer */
    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, "-seeds_graph.txt");

        // read the graph
        sprintf(filename, "%s/%s.json", input_dir, basename);
        if (!iftFileExists(filename)){
            iftWarning("File %s does not exist", "main", filename);
            iftFree(basename);
            continue;
        }
        iftGFLIMGraph *graph = iftReadGFLIMGraph(filename);

        // check if there are kernels created from the graph
        sprintf(filename, "%s/model/%s-conv%d-kernels.npy", model_dir, basename, layer);
        if (!iftFileExists(filename)){
            iftWarning("File %s does not exist", "main", filename);
            iftFree(basename);
            iftDestroyGFLIMGraph(&graph);
            continue;
        }
        iftMatrix *K = iftReadMatrix(filename);

        // read the bias
        sprintf(filename, "%s/model/%s-conv%d", model_dir, basename, layer);
        float *bias = iftGFLIMLoadBias(filename);
        if(bias == NULL) iftError("Bias is NULL", "main");

        // read the superpixel image
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
                iftDestroyMatrix(&K);
                iftFree(bias);
                iftDestroyGFLIMGraph(&graph);
                iftDestroyImage(&input_labels);
                iftError("Original image %s/%s.(png/jpg) does not exist", "main", orig_images_path, basename);
                iftFree(basename);
            }
        }
        iftImage *orig_image = iftReadImageByExt(filename);

        // compute the neighbors matrix based on the original image
        iftMatrix *neighbors = iftGFLIMConstructGraphRelation(orig_image, input_labels, node_feat, 
                                                    iftMax(arch->layer[layer].kernel_size, arch->layer[layer].pool_size));
        iftDestroyImage(&orig_image);

        // read the kernel labels
        int *true_labels = NULL;
        if (attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE) // several attention mechanisms use the kernel labels
        {
            sprintf(filename, "%s/model/%s-conv%d-labels.txt", model_dir, basename, layer);
            if (!iftFileExists(filename))
                iftError("File %s does not exist", "main", filename);
            int nkernels = 0;
            true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
        }

        // encode the layer
        iftGFLIMGraph *output_graph = iftGFLIMEncodeLayerV2(graph, K, bias, 
                                                        arch->layer[layer - 1], stride_type, 
                                                        input_labels, &output_labels, 
                                                        attention_opt, true_labels, 
                                                        neighbors);
        iftDestroyMatrix(&neighbors);
        iftDestroyImage(&input_labels);

        if(true_labels != NULL) iftFree(true_labels);

        /* save superpixel labels and graph (layer output) */
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
        iftDestroyMatrix(&K);
        iftFree(bias);
        iftDestroyGFLIMGraph(&graph);
    }

    iftFree(filename);
    iftDestroyFileSet(&fs);
    iftDestroyGFLIMArch(&arch);
    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
