#include "ift.h"

/**
 * \file iftGFLIMEncodeMergedLayer.c
 * \brief Encode a layer using the merged model created by iftGFLIMMergeLayerModels.
 */

// usado quando kernel_neighbors == "color" ou pool_neighbors == "color" ou stride_neighbors == "color"

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 11)
        iftError("Usage: iftGFLIMGSEncodeMergedLayer <P1> <P2> <P3> <P4> \n"
                 "[1] model directory \n"
                 "[2] architecture of the network (.json) \n"
                 "[3] layer number (1, 2, 3) \n"
                 "[4] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE, ...) \n"
                 "[5] kernel neighbors (color, act) \n"
                 "[6] pool neighbors (color, act) \n"
                 "[7] stride neighbors (color, act) \n"
                 "[8] graph feature (MEAN, SUM) \n"
                 "[9] aggregation strategy \n"
                 "[10] original images path \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[4]);
    char *kernel_neighbors = argv[5];
    char *pool_neighbors = argv[6];
    char *stride_neighbors = argv[7];
    iftGFLIMGraphNodeFeature graph_feat = iftGFLIMGraphGetNodeFeature(argv[8]);
    iftGFLIMAggregation aggregation_opt = iftGFLIMGraphGetAggregation(argv[9]);
    char *orig_images_path = argv[10];

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
        iftGFLIMGraph *input = iftReadGFLIMGraph(fs->files[i]->path);
        iftImage *label_img = NULL;
        sprintf(filename, "%s/%s.pgm", labels_dir, basename);
        label_img = iftReadImageByExt(filename);

        // compute the neighbors matrix based on the original image
        iftMatrix *neighbors = NULL;
        if(strcmp(kernel_neighbors, "color") == 0 || 
            strcmp(pool_neighbors, "color") == 0)
        {
            sprintf(filename, "%s/NeighborsMatrices/%s_%d.npy", model_dir, basename, layer-1);
            if(strcmp(stride_neighbors, "color") == 0 && iftFileExists(filename))
            {   
                neighbors = iftReadMatrix(filename);
            }else{
                int num_neighbors = iftMax(arch->layer[layer - 1].kernel_size, arch->layer[layer - 1].pool_size);
                iftImage *orig_image = NULL;
                sprintf(filename, "%s/%s.png", orig_images_path, basename);
                orig_image = iftReadImageByExt(filename);
                neighbors = iftGFLIMConstructGraphRelation(orig_image, 
                                                        label_img, 
                                                        graph_feat, 
                                                        num_neighbors);
            }
        }

        // encode layer
        iftGFLIMGraph *graph = iftGFLIMEncodeLayerV2(input, Kmerged, bias_merged, 
                                                        arch->layer[layer - 1], 
                                                        attention_opt, true_labels, 
                                                        neighbors, 
                                                        kernel_neighbors, pool_neighbors,
                                                        aggregation_opt);
        iftDestroyGFLIMGraph(&input);
        if(neighbors != NULL)
            iftDestroyMatrix(&neighbors);

        /* stride */
        if (strcmp(stride_neighbors, "color") == 0)
        {
            if (arch->layer[layer - 1].pool_stride > 1)
            {
                // read the precomputed graph
                iftGFLIMGraph *graph_resized = NULL;
                sprintf(filename, "%s/%s.json", output_dir, basename);
                graph_resized = iftReadGFLIMGraph(filename);

                // read the precomputed superpixel images
                iftImage *label_img_resized = NULL;
                sprintf(filename, "%s/%s.pgm", out_labels_dir, basename);
                label_img_resized = iftReadImageByExt(filename);

                float scale[3];
                scale[0] = (float)label_img_resized->xsize / (float)label_img->xsize;
                scale[1] = (float)label_img_resized->ysize / (float)label_img->ysize;
                scale[2] = 1.0;

                // reallocate the node vectors
                for(int j = 0; j < graph_resized->num_nodes; j++)
                {
                    iftFree(graph_resized->nodes[j].feats);
                    graph_resized->nodes[j].feats = (double *)iftAlloc(graph->num_feats, sizeof(double));
                }
                graph_resized->num_feats = graph->num_feats;

                iftGFLIMStridePrecomputed(graph,
                                          graph_resized,
                                          label_img,
                                          label_img_resized,
                                          scale);
                
                // save superpixel labels
                sprintf(filename, "%s/%s.pgm", out_labels_dir, basename);
                iftWriteImageByExt(label_img_resized, filename);
                iftDestroyImage(&label_img_resized);

                iftDestroyGFLIMGraph(&graph);
                graph = graph_resized;
                graph_resized = NULL;
            }
        }else
        {
            iftGFLIMGraph *output_graph = NULL;
            int nodes_map[graph->num_nodes];
            output_graph = iftGFLIMGraphStride(graph, arch->layer[layer - 1].pool_stride, nodes_map);
            for (int j = 0; j < label_img->n; j++)
            {
                int spx_label = label_img->val[j] - 1;
                if (spx_label < 0)
                    label_img->val[j] = 0;
                else
                    label_img->val[j] = nodes_map[spx_label] + 1;
            }
                    
            iftDestroyGFLIMGraph(&graph);
            graph = output_graph;
            output_graph = NULL;

            // save superpixel labels
            sprintf(filename, "%s/%s.pgm", out_labels_dir, basename);
            iftWriteImageByExt(label_img, filename);
            iftDestroyImage(&label_img);
        }
        
        iftDestroyImage(&label_img);
        iftDestroyGFLIMGraph(&graph);

        /* save graph (layer output) */
        sprintf(filename, "%s/%s.json", output_dir, basename);
        iftWriteGFLIMGraph(graph, filename);
        iftDestroyGFLIMGraph(&graph);
        
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
