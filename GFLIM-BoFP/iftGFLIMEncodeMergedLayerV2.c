#include "ift.h"

/**
 * \file iftGFLIMEncodeMergedLayer.c
 * \brief Encode a layer using the merged model created by iftGFLIMMergeLayerModels.
 */

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 7)
        iftError("Usage: iftGFLIMGSEncodeMergedLayer <P1> <P2> <P3> <P4> <P5> <P6> \n"
                 "[1] model directory \n"
                 "[2] architecture of the network (.json) \n"
                 "[3] layer number (1, 2, 3) \n"
                 "[4] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE, ...) \n"
                 "[5] graph node feature (MEAN, SUM) \n"
                 "[6] original images path (used to compute node neighbors) --- it must have at least the training images \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[4]);
    iftGFLIMGraphNodeFeature node_feat = iftGFLIMGraphGetNodeFeature(argv[5]);
    char *orig_images_path = argv[6];

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
    if (arch->layer[layer - 1].pool_stride > 1)
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
                                                        arch->layer[layer - 1], 
                                                        attention_opt, true_labels, 
                                                        neighbors);
        iftDestroyMatrix(&neighbors);

        // if pooling with stride was applied, generate the output superpixel labels
        if (arch->layer[layer - 1].pool_stride > 1)
        {
            iftGFLIMGraph *pooling_graph = output_graph;
            output_graph = NULL;
            int nodes_map[pooling_graph->num_nodes];
            output_labels = iftCreateImage(input_labels->xsize, input_labels->ysize, input_labels->zsize);
            output_graph = iftGFLIMGraphStride(pooling_graph, arch->layer[layer - 1].pool_stride, nodes_map);
            for (int j = 0; j < input_labels->n; j++)
            {
                int spx_label = input_labels->val[j] - 1;
                if(spx_label < 0)
                    output_labels->val[j] = 0;
                else
                    output_labels->val[j] = nodes_map[spx_label] + 1;
            }
            iftDestroyGFLIMGraph(&pooling_graph);
            
            // save superpixel labels
            sprintf(filename, "%s/%s.pgm", out_labels_dir, basename);
            iftWriteImageByExt(output_labels, filename);
            iftDestroyImage(&output_labels);
        }
        
        iftDestroyImage(&input_labels);
        iftDestroyGFLIMGraph(&graph);

        /* save graph (layer output) */
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
