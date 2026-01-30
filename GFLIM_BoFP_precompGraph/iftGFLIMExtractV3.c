#include "ift.h"

/**
 * \file iftGFLIMExtract.c
 * \brief Extract features from a set of images and decode from a GFLIM or a GFLIM-BoFP trained model.
 */

// usado quando kernel_neighbors == "color" ou pool_neighbors == "color" ou stride_neighbors == "color"

void validateArgs(char *images_dir, char *model_dir, iftGFLIMArch *arch, int nLayers, 
                  int batchsize, 
                  int argc, char *argv[])
{
    if (nLayers < 1)
        iftError("Invalid number of layers: %d", __func__, nLayers);

    if (nLayers > arch->nlayers)
        iftError("Invalid number of layers: %d", __func__, nLayers);

    if (!iftDirExists(images_dir) && !iftFileExists(images_dir))
        iftError("Images directory does not exist", __func__);

    if (!iftDirExists(model_dir))
        iftError("Model directory does not exist", __func__);

    if (batchsize < 1)
        iftError("Invalid batch size: %d", __func__, batchsize);
}

int *SplitIntList(char *str, int *nvalues)
{
    *nvalues = 100;
    int *values = (int *)iftAlloc(*nvalues, sizeof(int));
    char *token = strtok(str, ",");
    int i = 0;
    while (token != NULL && i < *nvalues)
    {
        if(i == *nvalues)
        {
            *nvalues += 100;
            values = (int *)iftRealloc(values, *nvalues * sizeof(int));
        }
        values[i++] = atoi(token);
        token = strtok(NULL, ",");
    }
    if(i != *nvalues)
        values = (int *)iftRealloc(values, i * sizeof(int));
    *nvalues = i;
    return values;
}

iftGFLIMDecoderType *SplitDecoderList(char *str, int *nvalues)
{
    *nvalues = 100;
    iftGFLIMDecoderType *values = (iftGFLIMDecoderType *)iftAlloc(*nvalues, sizeof(iftGFLIMDecoderType));
    char *token = strtok(str, ",");
    int i = 0;
    while (token != NULL && i < *nvalues)
    {
        if(i == *nvalues)
        {
            *nvalues += 100;
            values = (iftGFLIMDecoderType *)iftRealloc(values, *nvalues * sizeof(iftGFLIMDecoderType));
        }
        values[i++] = iftGLFLIMGetDecoderType(token);
        token = strtok(NULL, ",");
    }
    if(i != *nvalues)
        values = (iftGFLIMDecoderType *)iftRealloc(values, i * sizeof(iftGFLIMDecoderType));
    *nvalues = i;
    return values;
}

char *iftGFLIMDecoderToString(iftGFLIMDecoderType decoder)
{
    char *decoder_name = iftAllocCharArray(20);
    switch (decoder)
    {
    case IFT_GFLIM_TRISTATE_DECODER:
        sprintf(decoder_name, "tristate");
        break;

    case IFT_GFLIM_HYBRID_DECODER:
        sprintf(decoder_name, "hybrid");
        break;

    case IFT_GFLIM_GRAPHADAPTIVE_DECODER:
        sprintf(decoder_name, "graph-adaptive");
        break;

    case IFT_GFLIM_GAUSSIAN_DECODER:
        sprintf(decoder_name, "gaussian");
        break;

    case IFT_GFLIM_TEST_DECODER:
        sprintf(decoder_name, "test");
        break;

    case IFT_GFLIM_MEAN_DECODER:
        sprintf(decoder_name, "mean");
        break;

    default:
        iftError("Invalid decoder type: %d", __func__, decoder);
        return NULL;
    }
    return decoder_name;
}

void ProcessBatch(iftFileSet *fs, int first, int last, 
                char *model_dir, bool use_mask,
                float final_spx, bool fixed_spx,
                float initial_seeds,
                iftGFLIMGraphNodeFeature node_feat,
                iftGFLIMArch *arch, int nLayers, 
                iftGFLIMGraphAttention attention_opt, 
                bool save_intermediate, 
                bool *layers_saveActivations, bool *layers_decode,
                iftGFLIMDecoderType *decoders, int nDecoders,
                char *kernel_neighbors,
                char *pool_neighbors,
                char *stride_neighbors,
                iftGFLIMAggregation aggregation_opt)
{
    int ngraphs_batch = last - first + 1;

    //int nodes[] = {199, 195, 198, 127, 91, 164, 182};
    //int nnodes = 7;

    iftImageArray *orig_imgs = iftCreateImageArray(ngraphs_batch);
    iftImageArray *mask_imgs = iftCreateImageArray(ngraphs_batch);
    iftImageArray *label_imgs = iftCreateImageArray(ngraphs_batch);
    iftImageArray *label_imgs0 = iftCreateImageArray(ngraphs_batch);
    iftGFLIMGraphArray *input = iftCreateGFLIMGraphArray(ngraphs_batch);
    iftMatrix **neighbors = (iftMatrix **)iftAlloc(ngraphs_batch, sizeof(iftMatrix *));
    char **basenames = (char **)iftAlloc(ngraphs_batch, sizeof(char *));
    int initial_seeds_arr[ngraphs_batch];

    //printf("load original images and masks \n");
    // load original images and masks
//#pragma omp parallel for
    for (int k = 0; k <= last - first; k++)
    {
        int i = k + first;
        char ext[10], filename[1024];
        sprintf(ext, "%s", iftFileExt(fs->files[i]->path));
        basenames[k] = iftFilename(fs->files[i]->path, ext);
        orig_imgs->val[k] = iftReadImageByExt(fs->files[i]->path);

        //printf("%d %s \n", k, basenames[k]);

        int initial_seeds_img = (int)initial_seeds;
        int final_spx_img = (int)final_spx;
        int npixels = orig_imgs->val[k]->n;
        if(use_mask)
        {
            strcpy(filename, fs->files[i]->path);
            GetMaskDirInplace(filename);
            mask_imgs->val[k] = iftReadImageByExt(filename);
            npixels = 0;
            for(int p = 0; p < mask_imgs->val[k]->n; p++)
            {
                if(mask_imgs->val[k]->val[p] > 0)
                    npixels++; 
            }
        }else{
            mask_imgs->val[k] = NULL;
        }

        if(final_spx < 1)
            final_spx_img = (int)(npixels * final_spx);
        if(initial_seeds < 1)
            initial_seeds_img = (int)(npixels * initial_seeds);

        initial_seeds_arr[k] = initial_seeds_img;

        input->val[k] = iftImageToGFLIMGraph(orig_imgs->val[k], 
                                        final_spx_img, initial_seeds_img, 
                                        &label_imgs->val[k], 
                                        node_feat, 
                                        IFT_FLIMGRAPH_TYPE_DISF,
                                        mask_imgs->val[k]);

        iftVoxel v;
        v.z = 0;
        /*int max_xpos = 0, yval = 0;
        for(v.x = 1; v.x < mask_imgs->val[k]->xsize; v.x++)
        {
            for(v.y = 0; v.y < mask_imgs->val[k]->ysize; v.y++)
            {
                int p = iftGetVoxelIndex(mask_imgs->val[k], v);
                if(mask_imgs->val[k]->val[p] > 0)
                {
                    max_xpos = v.x;
                    yval = v.y;
                    break;
                }
            }
        }
        v.x = max_xpos; v.y = yval;
        printf("Max xpos in mask: %d, y: %d, index: %d \n", max_xpos, yval, iftGetVoxelIndex(mask_imgs->val[k], v));
        */

        /*int node, pixel;
        iftSet *adj;
        for(v.x = 130; v.x < 133; v.x++)
        {
            for(v.y = 105; v.y < 108; v.y++)
            {
                if(v.x >= label_imgs->val[k]->xsize || v.y >= label_imgs->val[k]->ysize)
                {
                    printf("Graph %d does not have voxel (%d,%d,%d)\n", k, v.x, v.y, v.z);
                }else{
                    pixel = iftGetVoxelIndex(label_imgs->val[k], v);
                    node = label_imgs->val[k]->val[pixel] - 1;
                    if(node < 0)
                    {
                        printf("Graph %d, pixel %d (%d,%d,%d) is background, mask val: %d\n", k, pixel, v.x, v.y, v.z, mask_imgs->val[k]->val[pixel]);
                    }else{
                        printf("Graph %d, pixel %d (%d,%d,%d), mask val: %d, Node %d information: \n", 
                            k, pixel, v.x, v.y, v.z, mask_imgs->val[k]->val[pixel], node);
                        printf("Num neighbors: %d \n", input->val[k]->nodes[node].numNeighbors);
                        printf("Neighbors: ");
                        adj = input->val[k]->nodes[node].adj;
                        while(adj != NULL)
                        {
                            printf("%d ", adj->elem);
                            adj = adj->next;
                        }
                        printf("\n");
                    }
                }
            }
        }*/

        label_imgs0->val[k] = iftCopyImage(label_imgs->val[k]);

        sprintf(filename, "%s/layer0/%s.json", model_dir, basenames[k]);
        iftWriteGFLIMGraph(input->val[k], filename);
        sprintf(filename, "%s/labels0/%s.pgm", model_dir, basenames[k]);
        iftWriteImageByExt(label_imgs->val[k], filename);
    }

    float scale[3];
    scale[0] = 1.0; scale[1] = 1.0; scale[2] = 1.0;

    //printf("Encode layers \n");
    for(int layer = 0; layer < nLayers; layer++)
    {
        char filename[1024];
        iftMatrix *Kmerged = NULL;
        float *bias_merged = NULL;
        int *true_labels = NULL;
        int nkernels = 0;
        int nlabels = 1;
        bool use_layer_skip = false;
        //printf("\nEncoding layer %d of %d \n", layer + 1, nLayers);

        //printf("load kernel weights and biases for layer %d \n", layer+1);
        // load kernel weights and biases
        sprintf(filename, "%s/model/conv%d-bias.txt", model_dir, layer + 1);
        if (iftFileExists(filename)) // is there a bias?
        {
            /* Load kernel and bias */
            sprintf(filename, "%s/model/conv%d", model_dir, layer + 1);
            iftGFLIMLoadMergedModel(filename, &Kmerged, &bias_merged);
        }
        else
        {
            /* Load kernel */
            sprintf(filename, "%s/model/conv%d-kernels.npy", model_dir, layer + 1);
            if (!iftFileExists(filename))
                iftError("Model file not found: %s", __func__, filename);
            Kmerged = iftReadMatrix(filename);
            int ninput_channels = input->val[0]->num_feats;

            /* Load mean and standard deviation */
            float *mean = iftAllocFloatArray(ninput_channels);
            float *stdev = iftAllocFloatArray(ninput_channels);
            sprintf(filename, "%s/model/conv%d", model_dir, layer + 1);
            iftReadMeanStdev(filename, mean, stdev, ninput_channels);

            /* Normalization (because there is no bias)*/
//#pragma omp parallel for
            for (int i = 0; i < input->n; i++){
                iftGFLIMGraphZNormalization(input->val[i], mean, stdev);
            }
            iftFree(mean);
            iftFree(stdev);
        }

        //printf("Load labels indicating class (label) per channel \n");
        /* Load labels indicating class (label) per channel */
        if ((attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE && bias_merged != NULL) ||
            layers_decode[layer] || layers_saveActivations[layer])
        {
            sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer + 1);
            if (!iftFileExists(filename))
                iftError("File %s does not exist", __func__, filename);
            
            true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);  
        }

        // extract features
        if((strcmp(kernel_neighbors, "color") == 0 || strcmp(pool_neighbors, "color") == 0)
           && (layer == 0 || arch->layer[layer-1].pool_stride > 1))
        {
            //printf("Conpute neighbors \n");
            for (int k = 0; k <= last - first; k++)
            {
                neighbors[k] = iftGFLIMConstructGraphRelation(orig_imgs->val[k], 
                                                        label_imgs->val[k], 
                                                        node_feat, 
                                                        iftMax(arch->layer[layer].kernel_size, 
                                                        arch->layer[layer].pool_size));
            }
        }

        /*printf("Encode features: graph size: %d %d, kernels size: %d %d\n", 
                    input->val[0]->num_nodes, input->val[0]->num_feats,
                    Kmerged->nrows, Kmerged->ncols);*/
        iftGFLIMEncodeLayerBatchV2(input, Kmerged, 
                                arch->layer[layer], 
                                bias_merged, 
                                attention_opt, 
                                true_labels, 
                                neighbors,
                                kernel_neighbors,
                                pool_neighbors,
                                aggregation_opt);

        /*printf("kernel_neighbors: %s, pool_neighbors: %s, stride_neighbors: %s \n", 
                kernel_neighbors, pool_neighbors, stride_neighbors);*/
        if((strcmp(kernel_neighbors, "color") == 0 || strcmp(pool_neighbors, "color") == 0)
           && (layer == arch->nlayers - 1 || arch->layer[layer].pool_stride > 1))
        {
            //printf("Destroy neighbors \n");
            for (int k = 0; k <= last - first; k++)
                iftDestroyMatrix(&neighbors[k]);
        }

        //printf("Destroy kernels and biases \n");
        iftDestroyMatrix(&Kmerged);
        iftFree(bias_merged);

        /* stride */
        if(strcmp(stride_neighbors, "color") == 0)
        {
            if(arch->layer[layer].pool_stride > 1)
            {
                //printf("Compute stride \n");

                scale[0] *= (1.0f / (float)arch->layer[layer].pool_stride);
                scale[1] *= (1.0f / (float)arch->layer[layer].pool_stride);
                //printf("Scale: %f, %f, %f \n", scale[0], scale[1], scale[2]);
                for (int k = 0; k <= last - first; k++)
                {   
                    int initial_seeds_img = (int)initial_seeds_arr[k];
                    int final_spx_img = input->val[k]->num_nodes;

                    int npixels;
                    int npixels_prev = orig_imgs->val[k]->n;

                    //printf("resize original images and mask images %d \n", k);
                    // resize original images and mask images
                    iftImage *orig_img_resized = iftInterp2D(orig_imgs->val[k], scale[0], scale[1]);
                    iftDestroyImage(&orig_imgs->val[k]);
                    orig_imgs->val[k] = orig_img_resized;
                    orig_img_resized = NULL;

                    npixels = orig_imgs->val[k]->n;
                    
                    if(use_mask)
                    {
                        npixels_prev = 0;
                        for(int p = 0; p < mask_imgs->val[k]->n; p++)
                        {
                            if(mask_imgs->val[k]->val[p] > 0)
                                npixels_prev++; 
                        }

                        //printf("npixels_prev: %d \n", npixels_prev);
                        iftImage *mask_img_resized = iftInterp2D(mask_imgs->val[k], scale[0], scale[1]);
                        iftDestroyImage(&mask_imgs->val[k]);
                        mask_imgs->val[k] = mask_img_resized;
                        mask_img_resized = NULL;

                        npixels = 0;
                        for(int p = 0; p < mask_imgs->val[k]->n; p++)
                        {
                            if(mask_imgs->val[k]->val[p] > 0)
                                npixels++; 
                        }
                        //printf("npixels: %d \n", npixels);
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
                        initial_seeds_arr[k] = initial_seeds_img;
                    }

                    //printf("initial_seeds_img: %d \n", initial_seeds_img);
                    //printf("final_spx_img: %d \n", final_spx_img);

                    //printf("recompute the graph and superpixel image \n");
                    // recompute the graph and superpixel image
                    iftImage *label_img_resized = NULL;
                    iftImage *label_img = label_imgs->val[k];
                    iftGFLIMGraph *graph = input->val[k];
                    iftGFLIMGraph *graph_resized = iftImageToGFLIMGraph(orig_imgs->val[k],
                                                            final_spx_img, initial_seeds_img,
                                                            &label_img_resized, node_feat,
                                                            IFT_FLIMGRAPH_TYPE_DISF,
                                                            mask_imgs->val[k]);
                    
                    //printf("reallocate the node vectors \n");
                    // reallocate the node vectors
                    for(int j = 0; j < graph_resized->num_nodes; j++)
                    {
                        iftFree(graph_resized->nodes[j].feats);
                        graph_resized->nodes[j].feats = (double *)iftAlloc(graph->num_feats, sizeof(double));
                    }

                    graph_resized->num_feats = input->val[k]->num_feats;

                    //print("Compute stride \n");
                    iftGFLIMStridePrecomputed(graph, 
                                        graph_resized, 
                                        label_img, 
                                        label_img_resized,
                                        scale);

                    //print("reassign label imgs \n");
                    label_img = NULL;
                    iftDestroyImage(&label_imgs->val[k]);
                    label_imgs->val[k] = label_img_resized;
                    label_img_resized = NULL;

                    ////print("reassign graphs \n");
                    graph = NULL;
                    iftDestroyGFLIMGraph(&input->val[k]);
                    input->val[k] = graph_resized;
                    graph_resized = NULL;
                }
            }
        } else {
            //print("Compute stride \n");
            iftGFLIMStrideBatch(input, arch->layer[layer], label_imgs, IFT_GFLIM_GRAPHSTRIDE);
        }

        /*print("After stride: \n");
        for (int k = 0; k < input->n; k++)
        {
            print("Graph %s, id %d: num_nodes: %d, num_feats: %d \n", 
                    basenames[k], k, input->val[k]->num_nodes, input->val[k]->num_feats);
            for (int p = 0; p < input->val[k]->num_nodes; p++)
            {
                //if(p == 199 || p == 195 || p == 198 || p == 127 || p == 91 || p == 164 || p == 182)
                if(p == 199)
                {
                    printf("Node %d: ", p);
                    printf("Feats: ");
                    for(int i = 0; i < input->val[k]->num_feats; i++)
                        printf("%.4f ", input->val[k]->nodes[p].feats[i]);
                    printf("\n");
                }
            }
        }*/
        

        //printf("save intermediate graphs and labels \n");
        // save intermediate graphs and labels
        if (save_intermediate || use_layer_skip)
        {
            char dir[1024];
            sprintf(dir, "%s/layer%d", model_dir, layer + 1);
            if(!iftDirExists(dir))
                iftMakeDir(dir);
            sprintf(dir, "%s/labels%d", model_dir, layer + 1);
            if(!iftDirExists(dir))
                iftMakeDir(dir);

            // write graphs and labels
            //#pragma omp parallel for
            for (int k = 0; k < ngraphs_batch; k++)
            {
                char filename[1024];
                sprintf(filename, "%s/layer%d/%s.json", model_dir, layer + 1, basenames[k]);
                iftWriteGFLIMGraph(input->val[k], filename);
                sprintf(filename, "%s/labels%d/%s.pgm", model_dir, layer + 1, basenames[k]);
                iftWriteImageByExt(label_imgs->val[k], filename);
            }
        }

        //printf("save activations \n");
        // save activations
        if(layers_saveActivations[layer])
        {
            if(true_labels == NULL)
            {
                sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer + 1);
                if (!iftFileExists(filename))
                    iftError("File %s does not exist", __func__, filename);

                true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
            }
            nlabels = iftGFLIMNumberOfTruelabels(true_labels, nkernels);

            for (int k = 0; k < ngraphs_batch; k++)
            {
                iftImage **activations = iftGFLIMGraphToActivations(input->val[k], label_imgs->val[k]);
            
                /* Compute the scale factor w.r.t. the input of the network */
                float scale_act[3];
                scale_act[0] = (float)label_imgs->val[k]->xsize / (float)label_imgs0->val[k]->xsize;
                scale_act[1] = (float)label_imgs->val[k]->ysize / (float)label_imgs0->val[k]->ysize;
                scale_act[2] = (float)label_imgs->val[k]->zsize / (float)label_imgs0->val[k]->zsize;

                for (int label = 0; label < nlabels; label++)
                {
                    char act_dir[1024];
                    sprintf(act_dir, "%s/activation%d/label%d/%s", model_dir, 
                                        layer + 1, label + 1, basenames[k]);
                    if (!iftDirExists(act_dir)) iftMakeDir(act_dir);
                }

                iftImage **temp = (iftImage **)iftAlloc(input->val[k]->num_feats, sizeof(iftImage *));
                //#pragma omp parallel for
                for (int j = 0; j < input->val[k]->num_feats; j++)
                {
                    char filename[1024];
                    temp[j] = iftInterp2D(activations[j], scale_act[0], scale_act[1]);
                    
                    int label = 1;
                    if(true_labels != NULL) label = true_labels[j];

                    sprintf(filename, "%s/activation%d/label%d/%s/kernel-%03d.pgm", 
                                model_dir, layer + 1, label, basenames[k], j + 1);
                    iftWriteImageByExt(temp[j], filename);
                }

                for (int j = 0; j < input->val[k]->num_feats; j++)
                {
                    iftDestroyImage(&activations[j]);
                    iftDestroyImage(&temp[j]);
                }
                iftFree(temp);
                iftFree(activations);
            }
        }
        
        //printf("decode layer \n");
        // decode layer
        if(layers_decode[layer])
        {
            if(true_labels == NULL)
            {
                sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer + 1);
                if (!iftFileExists(filename))
                    iftError("File %s does not exist", __func__, filename);

                if(strcmp(arch->layer[layer].skip_type, "concat") != 0 || layer == 0 ||
                   strcmp(arch->layer[layer].skip_type, "concat_att") != 0)
                    true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
                else{
                    iftIntArray *layers_to_concat = iftCreateIntArray(arch->nlayers);
                    layers_to_concat->val[0] = layer+1; // add current layer
                    int nconcat_layers = getLayersToConcat(arch, layer+1, layer, layers_to_concat, 1);
                    nkernels = 0;
                    for(int i = 0; i < nconcat_layers; i++)
                    {
                        sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layers_to_concat->val[i]);
                        if(!iftFileExists(filename))
                            iftError("File %s does not exist", "main", filename);
                            
                        /* load merged model */
                        int nkernels_i = 0;
                        int *truelabels_i = iftGFLIMLoadKernelLabels(filename, &nkernels_i);
                        
                        nkernels += nkernels_i;
                        true_labels = realloc(true_labels, nkernels * sizeof(int));
                        for(int j = 0; j < nkernels_i; j++){
                            true_labels[nkernels - nkernels_i + j] = truelabels_i[j];
                        }
                        iftFree(truelabels_i);
                    }
                    iftDestroyIntArray(&layers_to_concat);
                }
            }
            nlabels = iftGFLIMNumberOfTruelabels(true_labels, nkernels);

            for(int d = 0; d < nDecoders; d++)
            {
                char output_dir[1024];
                char *decoder_name = iftGFLIMDecoderToString(decoders[d]);

                sprintf(output_dir, "%s/%s_%d", model_dir, decoder_name, layer + 1);
                if(!iftDirExists(output_dir))
                    iftMakeDir(output_dir);
                
                for (int k = 0; k < ngraphs_batch; k++)
                {
                    iftImageArray *saliency = NULL;
                    iftGFLIMGraph *graph = iftCopyGFLIMGraph(input->val[k]->num_feats,input->val[k], true);
                    saliency = iftGFLIMDecodeLayer(graph, label_imgs->val[k], 
                                                label_imgs->val[k], true_labels, 
                                                nlabels, decoders[d]);
                    iftDestroyGFLIMGraph(&graph);
                    
                    /* post-process and save the salience maps */
                    //#pragma omp parallel for
                    for (int label = 0; label < saliency->n; label++)
                    {
                        char sal_dir[1024], filename[1024];
                        sprintf(sal_dir, "%s/label%d", output_dir, label + 1);
                        if (!iftDirExists(sal_dir)) iftMakeDir(sal_dir);

                        if (iftIs3DImage(label_imgs->val[k]))
                            sprintf(filename, "%s/%s.nii.gz", sal_dir, basenames[k]);
                        else
                            sprintf(filename, "%s/%s.png", sal_dir, basenames[k]);
                        iftWriteImageByExt(saliency->val[label], filename);
                    }
                    iftDestroyImageArray(&saliency);
                }
                iftFree(decoder_name);
            }
        }
        iftFree(true_labels); 
    }
    iftFree(neighbors);

    //printf("delete folders \n");
    // delete folders
    if(!save_intermediate)
    {
        for (int layer = 0; layer <= nLayers; layer++)
        {
            char dir[1024];
            sprintf(dir, "%s/layer%d", model_dir, layer);
            iftRemoveDir(dir);
            sprintf(dir, "%s/labels%d", model_dir, layer);
            iftRemoveDir(dir);
        }
    }

    for (int k = 0; k <= last - first; k++)
        iftFree(basenames[k]);
    iftFree(basenames);
    iftDestroyImageArray(&label_imgs);
    iftDestroyImageArray(&label_imgs0);
    iftDestroyImageArray(&orig_imgs);
    iftDestroyImageArray(&mask_imgs);
    iftDestroyGFLIMGraphArray(&input);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 20)
    {
        iftError("Usage: iftGFLIMExtract <P1> <P2> <P3> <P4> <P5> "
                                        "<P6> <P7> <P8> <P9> <P10> "
                                        "<P11> <P12> <P13> <P14> <P15> <P16> <P17> <P18> <P19> \n"
                // parameters related to input data
                "[ 1] Images directory \n"
                "[ 2] Model directory \n"
                "[ 3] Use mask (boolean) \n"
                // parameters related to the graph
                "[ 4] Final superpixels \n"
                "[ 5] Initial seeds \n"
                "[ 6] Fixed number of superpixels (boolean) \n"
                "[ 7] graph node feature (MEAN, SUM) \n"
                // parameters related to the network
                "[ 8] Architecture file \n"
                "[ 9] Number of layers \n"
                "[10] Batch size (> 0) \n"
                "[11] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE) \n"
                "[12] aggregation strategy \n"
                // parameters related to output data
                "[13] Save graphs and labels after each layer (1/0) \n"
                "[14] Layers to save activations (1,2,..) \n"
                "[15] Layers to decode (1,2,..) \n"
                "[16] Decoders (tristate, hybrid, graph-adaptive, gaussian) \n"
                // parameters related to kernels, pooling and stride neighbors
                "[17] Kernel neighbors (act, color) \n"
                "[18] Pool neighbors (act, color) \n"
                "[19] Stride neighbors (act, color) \n"
                "argc: %d", "main", argc);
    }
    tstart = iftTic();
    srand(42);

    // parameters related to input data 
    char *images_dir = argv[1];
    char *model_dir = argv[2];
    bool use_mask = atoi(argv[3]);

    // parameters related to the graph
    float final_spx = atof(argv[4]);
    float initial_seeds = atof(argv[5]);
    bool fixed_num_spx = atoi(argv[6]);
    iftGFLIMGraphNodeFeature node_feat = iftGFLIMGraphGetNodeFeature(argv[7]);

    // parameters related to the network
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[8]);
    int nLayers = atoi(argv[9]);
    int batchsize = atoi(argv[10]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[11]);
    iftGFLIMAggregation aggregation_opt = iftGFLIMGraphGetAggregation(argv[12]);

    // parameters related to output data
    bool save_intermediate = atoi(argv[13]); // save graphs and labels after each layer
    int nLayers_saveAct = 0;
    int *list_int_act = SplitIntList(argv[14], &nLayers_saveAct); // layers to save activations
    int nLayers_decode = 0;
    int *list_int_decode = SplitIntList(argv[15], &nLayers_decode); // layers to decode
    int nDecoders = 0;
    iftGFLIMDecoderType *decoders = SplitDecoderList(argv[16], &nDecoders); // decoders
    // parameters related to kernels, pooling and stride neighbors
    char *kernel_neighbors = argv[17];
    char *pool_neighbors = argv[18];
    char *stride_neighbors = argv[19];
    bool *layers_saveAct = (bool *)iftAlloc(nLayers, sizeof(bool));
    for(int i = 0; i < nLayers_saveAct; i++){
        if(list_int_act[i] > 0)
            layers_saveAct[list_int_act[i]-1] = true;
    }
    iftFree(list_int_act);

    bool *layers_decode = (bool *)iftAlloc(nLayers, sizeof(bool));
    for(int i = 0; i < nLayers_decode; i++){
        if(list_int_decode[i] > 0)
            layers_decode[list_int_decode[i]-1] = true;
    }
    iftFree(list_int_decode);

    validateArgs(images_dir, model_dir, arch, nLayers, batchsize, argc, argv);

    iftFileSet *fs_imgs = iftLoadFileSetFromDirOrCSV(images_dir, 1, false);
    int ngraphs = fs_imgs->n;
    int nbatches = 0;
    int nremaining_graphs = fs_imgs->n;

    if (fs_imgs->n < 1)
        iftError("Image list is empty", __func__);

    char dir[1024];
    sprintf(dir, "%s/layer0", model_dir);
    if(iftDirExists(dir) == false)
        iftMakeDir(dir);
    sprintf(dir, "%s/labels0", model_dir);
    if(iftDirExists(dir) == false)
        iftMakeDir(dir);

    batchsize = iftMin(batchsize, ngraphs);
    nbatches = ngraphs / batchsize;
    nremaining_graphs = nremaining_graphs - nbatches * batchsize;

    printf("\n");
    for (int batch = 0; batch < nbatches; batch++)
    {
        int first = batch * batchsize, last = first + batchsize - 1;
        printf("Processing batch %d of %d batches\r", batch + 1, nbatches);
        fflush(stdout);

        ProcessBatch(fs_imgs, first, last, 
                    model_dir, use_mask,
                    final_spx, fixed_num_spx, 
                    initial_seeds, node_feat,
                    arch, nLayers, 
                    attention_opt,
                    save_intermediate, layers_saveAct, layers_decode,
                    decoders, nDecoders, 
                    kernel_neighbors,
                    pool_neighbors,
                    stride_neighbors,
                    aggregation_opt);
    }

    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last)
    {
        printf("Processing remaining %d images\r", last - first + 1);
        fflush(stdout);

        ProcessBatch(fs_imgs, first, last, 
                    model_dir, use_mask,
                    final_spx, fixed_num_spx, 
                    initial_seeds, node_feat,
                    arch, nLayers,
                    attention_opt, 
                    save_intermediate, layers_saveAct, layers_decode,
                    decoders, nDecoders, 
                    kernel_neighbors,
                    pool_neighbors,
                    stride_neighbors,
                    aggregation_opt);
    }

    iftDestroyFileSet(&fs_imgs);
    iftDestroyGFLIMArch(&arch);
    iftFree(layers_saveAct);
    iftFree(layers_decode);
    iftFree(decoders);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("\nDone ... %s\n", time);
    iftFree(time);
    return 0;
}
