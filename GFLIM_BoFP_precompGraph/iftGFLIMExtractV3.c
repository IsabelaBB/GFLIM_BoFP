#include "ift.h"

/**
 * \file iftGFLIMExtractV3.c
 * \brief Use a model trained with GFLIM or GFLIM-BoFP to extract features from a set of images and decode them.
 * \note Used when kernel_neighbors, pool_neighbors, and stride_neighbors are either "act" or "color" (i.e., computed using activation or color information).
 * \note This version does NOT support residual connections and stride IFT_GFLIM_IMAGESTRIDE.
 * \author Isabela Borlido
 */

void ProcessBatch(iftFileSet *fs, int first, int last, 
    char *model_dir, iftGFLIMArch *arch, int nLayers, 
    float initial_seeds, float nsuperpixels, 
    bool fixed_spx, iftGFLIMGraphNodeFeature graph_feat,
    iftGFLIMGraphSpxMethod spx_method,
    iftGFLIMGraphAttention attention_opt, 
    iftGFLIMAggregation aggregation_opt, 
    bool *layers_saveGraphAct, bool *layers_saveImgAct, 
    bool *layers_decode,
    iftGFLIMDecoderType *decoders, int nDecoders,
    char *kernel_neighbors,
    char *pool_neighbors,
    char *stride_neighbors,
    bool use_mask)
{
    int ngraphs_batch = last - first + 1;
    iftImageArray *orig_imgs = iftCreateImageArray(ngraphs_batch);
    iftImageArray *mask_imgs = iftCreateImageArray(ngraphs_batch);
    iftImageArray *label_imgs = iftCreateImageArray(ngraphs_batch);
    iftImageArray *label_imgs0 = iftCreateImageArray(ngraphs_batch);
    iftGFLIMGraphArray *input = iftCreateGFLIMGraphArray(ngraphs_batch);
    iftMatrix **neighbors = (iftMatrix **)iftAlloc(ngraphs_batch, sizeof(iftMatrix *));
    char **basenames = (char **)iftAlloc(ngraphs_batch, sizeof(char *));
    int initial_seeds_arr[ngraphs_batch];

    
    // load original images and masks
    for (int k = 0; k < ngraphs_batch; k++){
        int i = k + first;
        char ext[10], filename[1024];
        sprintf(ext, "%s", iftFileExt(fs->files[i]->path));
        basenames[k] = iftFilename(fs->files[i]->path, ext);
        orig_imgs->val[k] = iftReadImageByExt(fs->files[i]->path);
        int initial_seeds_img = (int)initial_seeds;
        int nsuperpixels_img = (int)nsuperpixels;
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

        if(nsuperpixels < 1)
            nsuperpixels_img = (int)(npixels * nsuperpixels);
        if(initial_seeds < 1)
            initial_seeds_img = (int)(npixels * initial_seeds);

        initial_seeds_arr[k] = initial_seeds_img;

        input->val[k] = iftImageToGFLIMGraph(orig_imgs->val[k], 
                                        nsuperpixels_img, initial_seeds_img, 
                                        &label_imgs->val[k], 
                                        graph_feat, 
                                        spx_method,
                                        mask_imgs->val[k]);

        label_imgs0->val[k] = iftCopyImage(label_imgs->val[k]);

        sprintf(filename, "%s/layer0/%s.json", model_dir, basenames[k]);
        iftWriteGFLIMGraph(input->val[k], filename);
        sprintf(filename, "%s/labels0/%s.pgm", model_dir, basenames[k]);
        iftWriteImageByExt(label_imgs->val[k], filename);
    }

    float scale[3];
    scale[0] = 1.0; scale[1] = 1.0; scale[2] = 1.0;

    /* extract features for each layer */
    for(int l = 0; l < nLayers; l++)
    {
        iftMatrix *Kmerged = NULL;
        float *bias_merged = NULL;
        int *true_labels = NULL;
        char filename[1024];
        int nkernels = 0;
        int nlabels = 1;

        // load kernel weights and biases
        sprintf(filename, "%s/model/conv%d-bias.txt", model_dir, l + 1);
        if (iftFileExists(filename)) // is there a bias?
        {
            /* Load kernel and bias */
            sprintf(filename, "%s/model/conv%d", model_dir, l + 1);
            iftGFLIMLoadMergedModel(filename, &Kmerged, &bias_merged);
        }
        else
        {
            /* Load kernel */
            sprintf(filename, "%s/model/conv%d-kernels.npy", model_dir, l + 1);
            if (!iftFileExists(filename))
                iftError("Model file not found: %s", __func__, filename);
            Kmerged = iftReadMatrix(filename);
            int ninput_channels = input->val[0]->num_feats;

            /* Load mean and standard deviation */
            float *mean = iftAllocFloatArray(ninput_channels);
            float *stdev = iftAllocFloatArray(ninput_channels);
            sprintf(filename, "%s/model/conv%d", model_dir, l + 1);
            iftReadMeanStdev(filename, mean, stdev, ninput_channels);

            /* Normalization (because there is no bias)*/
//#pragma omp parallel for
            for (int i = 0; i < input->n; i++){
                iftGFLIMGraphZNormalization(input->val[i], mean, stdev);
            }
            iftFree(mean);
            iftFree(stdev);
        }

        /* Load labels indicating class (label) per channel */
        if ((attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE && bias_merged != NULL) ||
            layers_decode[l] || layers_saveImgAct[l])
        {
            sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, l + 1);
            if (!iftFileExists(filename))
                iftError("File %s does not exist", __func__, filename);
            
            true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);  
        }

        // Compute neighbors
        if((strcmp(kernel_neighbors, "color") == 0 || strcmp(pool_neighbors, "color") == 0)
           && (l == 0 || arch->layer[l-1].pool_stride > 1))
        {
            for (int k = 0; k < ngraphs_batch; k++)
            {
                neighbors[k] = iftGFLIMConstructGraphRelation(orig_imgs->val[k], 
                                                        label_imgs->val[k], 
                                                        graph_feat, 
                                                        iftMax(arch->layer[l].kernel_size, 
                                                        arch->layer[l].pool_size));
            }
        }

        // extract features
        iftGFLIMEncodeLayerBatchV2(input, Kmerged, 
            arch->layer[l], bias_merged, 
            attention_opt, true_labels, 
            neighbors,kernel_neighbors,
            pool_neighbors, aggregation_opt);

        if((strcmp(kernel_neighbors, "color") == 0 || strcmp(pool_neighbors, "color") == 0)
           && (l == arch->nlayers - 1 || arch->layer[l].pool_stride > 1))
        {
            for (int k = 0; k < ngraphs_batch; k++)
                iftDestroyMatrix(&neighbors[k]);
        }

        iftDestroyMatrix(&Kmerged);
        iftFree(bias_merged);

        /* stride */
        if(strcmp(stride_neighbors, "color") == 0)
        {
            if(arch->layer[l].pool_stride > 1)
            {
                scale[0] *= (1.0f / (float)arch->layer[l].pool_stride);
                scale[1] *= (1.0f / (float)arch->layer[l].pool_stride);
                for (int k = 0; k < ngraphs_batch; k++)
                {   
                    int initial_seeds_img = (int)initial_seeds_arr[k];
                    int nsuperpixels_img = input->val[k]->num_nodes;

                    int npixels;
                    int npixels_prev = orig_imgs->val[k]->n;

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
                    }

                    if(!fixed_spx)
                    {
                        if(nsuperpixels < 1)
                            nsuperpixels_img = nsuperpixels * npixels;
                        else
                            nsuperpixels_img = (nsuperpixels_img * npixels) / npixels_prev;
                        nsuperpixels_img = iftMin(nsuperpixels_img, npixels);

                        if(initial_seeds < 1)
                            initial_seeds_img = initial_seeds * npixels;
                        else
                            initial_seeds_img = (initial_seeds_img * npixels) / npixels_prev;
                        initial_seeds_img = iftMin(initial_seeds_img, npixels);
                        initial_seeds_arr[k] = initial_seeds_img;
                    }

                    // recompute the graph and superpixel image
                    iftImage *label_img_resized = NULL;
                    iftImage *label_img = label_imgs->val[k];
                    iftGFLIMGraph *graph = input->val[k];
                    iftGFLIMGraph *graph_resized = iftImageToGFLIMGraph(orig_imgs->val[k],
                                                            nsuperpixels_img, initial_seeds_img,
                                                            &label_img_resized, graph_feat,
                                                            spx_method,
                                                            mask_imgs->val[k]);
                    
                    // reallocate the node vectors
                    for(int j = 0; j < graph_resized->num_nodes; j++)
                    {
                        iftFree(graph_resized->nodes[j].feats);
                        graph_resized->nodes[j].feats = (double *)iftAlloc(graph->num_feats, sizeof(double));
                    }

                    graph_resized->num_feats = input->val[k]->num_feats;

                    iftGFLIMStridePrecomputed(graph, 
                                        graph_resized, 
                                        label_img, 
                                        label_img_resized,
                                        scale);

                    label_img = NULL;
                    iftDestroyImage(&label_imgs->val[k]);
                    label_imgs->val[k] = label_img_resized;
                    label_img_resized = NULL;

                    graph = NULL;
                    iftDestroyGFLIMGraph(&input->val[k]);
                    input->val[k] = graph_resized;
                    graph_resized = NULL;
                }
            }
        } else {
            iftGFLIMStrideBatch(input, arch->layer[l], label_imgs, IFT_GFLIM_GRAPHSTRIDE);
        }

        // save intermediate graphs and labels
        if (layers_saveGraphAct[l])
        {
            char dir[1024];
            sprintf(dir, "%s/layer%d", model_dir, l + 1);
            if(!iftDirExists(dir))
                iftMakeDir(dir);
            sprintf(dir, "%s/labels%d", model_dir, l + 1);
            if(!iftDirExists(dir))
                iftMakeDir(dir);

            // write graphs and labels
            //#pragma omp parallel for
            for (int k = 0; k < ngraphs_batch; k++)
            {
                char filename[1024];
                sprintf(filename, "%s/layer%d/%s.json", model_dir, l + 1, basenames[k]);
                iftWriteGFLIMGraph(input->val[k], filename);
                sprintf(filename, "%s/labels%d/%s.pgm", model_dir, l + 1, basenames[k]);
                iftWriteImageByExt(label_imgs->val[k], filename);
            }
        }

        // save activations
        if(layers_saveImgAct[l])
        {
            if(true_labels == NULL)
            {
                sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, l + 1);
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
                                        l + 1, label + 1, basenames[k]);
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
                                model_dir, l + 1, label, basenames[k], j + 1);
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

        // decode layer
        if(layers_decode[l])
        {
            if(true_labels == NULL)
            {
                sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, l + 1);
                if (!iftFileExists(filename))
                    iftError("File %s does not exist", __func__, filename);
                true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
            }
            nlabels = iftGFLIMNumberOfTruelabels(true_labels, nkernels);

            for(int d = 0; d < nDecoders; d++)
            {
                char output_dir[1024];
                char *decoder_name = iftGFLIMDecoderToString(decoders[d]);

                sprintf(output_dir, "%s/%s_%d", model_dir, decoder_name, l + 1);
                if(!iftDirExists(output_dir))
                    iftMakeDir(output_dir);
                
                for (int k = 0; k < ngraphs_batch; k++)
                {
                    iftImageArray *saliency = NULL;
                    iftGFLIMGraph *graph = iftCopyGFLIMGraph(input->val[k]->num_feats,input->val[k], true);
                    // if a error occurs here, try changing label_imgs0 to label_imgs
                    saliency = iftGFLIMDecodeLayer(graph, label_imgs->val[k], 
                                                label_imgs0->val[k], true_labels, 
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

    // delete folders
    for (int l = 1; l <= nLayers; l++)
    {
        if(layers_saveGraphAct[l-1]) continue;
        char dir[1024];
        sprintf(dir, "%s/layer%d", model_dir, l);
        iftRemoveDir(dir);
        sprintf(dir, "%s/labels%d", model_dir, l);
        iftRemoveDir(dir);
    }

    for (int k = 0; k < ngraphs_batch; k++)
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

    if (argc != 19 && argc != 20 && argc != 21)
    {
        iftError("Usage: iftGFLIMExtractV3 P1 P2 P3 P4 P5 "
                                        "P6 P7 P8 P9 P10 "
                                        "P11 P12 P13 P14 P15 "
                                        "P16 P17 P18 P19 [P20] \n"
                // parameters related to input data
                "P1 : Images directory \n"
                "P2 : Model directory \n"
                // parameters related to the network
                "P3 : Architecture file \n"
                "P4 : Number of layers \n"
                "P5 : Batch size (> 0) \n"
                "P6 : attention. See iftGFLIMGraphAttention at iftGFLIMGraph.h for more details \n"
                "P7 : aggregation. See iftGFLIMAggregation at iftGFLIMGraph.h for more details \n"
                // parameters related to the graph
                "P8 : Graph feature. See iftGFLIMGraph.h for more information \n"
                "P9 : superpixel method. See iftGFLIMGraph.h for more information \n"
                "P10: Number of superpixels \n"
                "P11: Fixed number of superpixels (boolean) \n"
                // parameters related to output data
                "P12: Layers to save outputs graphs and superpixel maps (1,2,..) \n"
                "P13  Layers to image save activations (1,2,..) \n"
                "P14: Layers to decode (1,2,..) \n"
                "P15: Decoders. See iftGFLIMDecoderType at iftGFLIM_BoFP.h for more details \n"
                // parameters related to kernels, pooling and stride neighbors
                "P16: Kernel neighbors (act, color) \n"
                "P17: Pool neighbors (act, color) \n"
                "P18: Stride neighbors (act, color) \n"
                // optional parameters
                "P19: Initial seeds (only used when superpixel method is DISF) \n" 
                "P20: Use mask (optional, bool). Compute the graph using only pixels inside the mask \n",
                "argc: %d", "main", argc);
    }
    tstart = iftTic();
    srand(42);

    // parameters related to input data
    char *images_dir = argv[1];
    char *model_dir = argv[2];

    // parameters related to the network
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[3]);
    int nLayers = atoi(argv[4]);
    int batchsize = atoi(argv[5]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[6]);
    iftGFLIMAggregation aggregation_opt = iftGFLIMGraphGetAggregation(argv[7]);
    
    // parameters related to the graph
    iftGFLIMGraphNodeFeature graph_feat = iftGFLIMGraphGetNodeFeature(argv[8]);
    iftGFLIMGraphSpxMethod spx_method = iftGFLIMGraphGetSpxMethod(argv[9]);
    float nsuperpixels = atof(argv[10]);
    bool fixed_num_spx = atoi(argv[11]);
    
    // parameters related to output data
    int nLayers_saveAct = 0;
    int *list_graph_int_act = SplitIntList(argv[12], &nLayers_saveAct); // layers to save graph activations
    int nLayers_saveImgAct = 0;
    int *list_img_int_act = SplitIntList(argv[13], &nLayers_saveImgAct); // layers to save image activations
    int nLayers_decode = 0;
    int *list_int_decode = SplitIntList(argv[14], &nLayers_decode); // layers to decode
    int nDecoders = 0;
    iftGFLIMDecoderType *decoders = SplitDecoderList(argv[15], &nDecoders); // decoders
    
    // parameters related to kernels, pooling and stride neighbors
    char *kernel_neighbors = argv[16];
    char *pool_neighbors = argv[17];
    char *stride_neighbors = argv[18];
    
    // optional parameters
    float initial_seeds = (argc > 19) ? atof(argv[19]) : 0.0f; // only used when superpixel method is DISF
    bool use_mask = (argc == 21) ? atoi(argv[20]) : false;

    // read which layers to save activations and decode
    bool *layers_saveGraphAct = (bool *)iftAlloc(nLayers, sizeof(bool));
    for(int i = 0; i < nLayers_saveAct; i++){
        if(list_graph_int_act[i] > 0)
            layers_saveGraphAct[list_graph_int_act[i]-1] = true;
    }
    iftFree(list_graph_int_act);

    bool *layers_saveImgAct = (bool *)iftAlloc(nLayers, sizeof(bool));
    for(int i = 0; i < nLayers_saveImgAct; i++){
        if(list_img_int_act[i] > 0)
            layers_saveImgAct[list_img_int_act[i]-1] = true;
    }
    iftFree(list_img_int_act);

    bool *layers_decode = (bool *)iftAlloc(nLayers, sizeof(bool));
    for(int i = 0; i < nLayers_decode; i++){
        if(list_int_decode[i] > 0)
            layers_decode[list_int_decode[i]-1] = true;
    }
    iftFree(list_int_decode);

    // validate arguments
    if (nLayers < 1)
        iftError("Invalid number of layers: %d", __func__, nLayers);

    if (nLayers > arch->nlayers)
        iftError("Invalid number of layers: %d", __func__, nLayers);

    if (!iftDirExists(images_dir) && !iftFileExists(images_dir))
        iftError("Images directory does not exist", __func__);

    if (!iftDirExists(model_dir))
        iftError("Model directory does not exist", __func__);
    
    if (nsuperpixels < 1)
        iftError("Invalid number of superpixels: %d", __func__, nsuperpixels);
    
    if (batchsize < 1)
        iftError("Invalid batch size: %d", __func__, batchsize);

    if (spx_method == IFT_FLIMGRAPH_TYPE_DISF)
    {
        if (argc < 20)
            iftError("Initial seeds is required when superpixel method is DISF", __func__);
        if (initial_seeds < 1)
            iftError("Invalid number of initial seeds: %d", __func__, initial_seeds);
        if (nsuperpixels > initial_seeds)
            iftError("Number of superpixels (%d) must be smaller"
                    " than the number of initial seeds (%d)", 
                    __func__, nsuperpixels, initial_seeds);
    }else{
        if(use_mask){
            iftError("Mask can only be used when superpixel method is DISF", __func__);
        }
    }

    iftFileSet *fs = iftLoadFileSetFromDirOrCSV(images_dir, 1, false);
    int ngraphs = fs->n;
    int nbatches = 0;
    int nremaining_graphs = fs->n;

    if (fs->n < 1)
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

        ProcessBatch(fs, first, last, 
            model_dir, arch, nLayers, 
            initial_seeds, nsuperpixels, 
            fixed_num_spx, graph_feat, spx_method,
            attention_opt, aggregation_opt, 
            layers_saveGraphAct, layers_saveImgAct, layers_decode,
            decoders, nDecoders, 
            kernel_neighbors,
            pool_neighbors,
            stride_neighbors,
            use_mask);
    }

    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last)
    {
        printf("Processing remaining %d images\r", last - first + 1);
        fflush(stdout);

        ProcessBatch(fs, first, last, 
            model_dir, arch, nLayers, 
            initial_seeds, nsuperpixels,
            fixed_num_spx, graph_feat, spx_method,
            attention_opt, aggregation_opt, 
            layers_saveGraphAct, layers_saveImgAct, layers_decode,
            decoders, nDecoders, 
            kernel_neighbors,
            pool_neighbors,
            stride_neighbors,
            use_mask);
    }

    sprintf(dir, "%s/layer0", model_dir);
    iftRemoveDir(dir);
    sprintf(dir, "%s/labels0", model_dir);
    iftRemoveDir(dir);
    
    iftDestroyFileSet(&fs);
    iftDestroyGFLIMArch(&arch);
    iftFree(layers_saveGraphAct);
    iftFree(layers_saveImgAct);
    iftFree(layers_decode);
    iftFree(decoders);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("\nDone ... %s\n", time);
    iftFree(time);
    return 0;
}
