#include "ift.h"

/**
 * \file iftGFLIMExtract.c
 * \brief Use a model trained with GFLIM or GFLIM-BoFP to extract features from a set of images and decode them.
 * \note This version does supports residual connections and stride IFT_GFLIM_IMAGESTRIDE.
 * \note This version does NOT supports neighbors based on color (i.e., GFLIM with precomputed graphs).
 * \author Isabela Borlido
 */


void ProcessBatch(iftFileSet *fs, int first, int last, 
    char *model_dir, 
    iftGFLIMArch *arch, int nLayers, 
    iftGFLIMStrideType stride_type,
    int initial_seeds, int nsuperpixels,
    iftGFLIMGraphNodeFeature graph_feat, 
    iftGFLIMGraphSpxMethod spx_method,
    iftGFLIMGraphAttention attention_opt, 
    iftGFLIMAggregation aggregation_opt, 
    bool kernel_removal, bool *layers_saveGraphAct, 
    bool *layers_saveImgAct, bool *layers_decode,
    iftGFLIMDecoderType *decoders, int nDecoders,
    int *finer_layers, bool use_mask)
{
    iftGFLIMGraphArray *input = NULL;
    int ngraphs_batch = last - first + 1;

    bool require_label_imgs0 = false;
    for (int layer = 0; layer < nLayers; layer++)
    {
        if(layers_saveImgAct[layer] || layers_decode[layer]){
            require_label_imgs0 = true;
            break;
        }
    }

    iftImageArray *label_imgs = iftCreateImageArray(ngraphs_batch);
    iftImageArray *label_imgs0 = NULL;
    char **basenames = (char **)iftAlloc(ngraphs_batch, sizeof(char *));
    bool arch_has_skip = false;

    if(require_label_imgs0)
        label_imgs0 = iftCreateImageArray(ngraphs_batch);

    /* Create graphs */
    input = iftGFLIMImageToFLIMGraphBatch(fs, first, last, initial_seeds, 
                                            nsuperpixels, graph_feat, spx_method, 
                                            label_imgs, use_mask);

    for (int k = 0; k < ngraphs_batch; k++){
        int i = k + first;
        char ext[10], filename[1024];
        sprintf(ext, "%s", iftFileExt(fs->files[i]->path));
        basenames[k] = iftFilename(fs->files[i]->path, ext);
        if(require_label_imgs0)
        {
            label_imgs0->val[k] = iftCopyImage(label_imgs->val[k]);
        }
    }

    for (int l = 0; l < nLayers; l++)
    {
        if(strcmp(arch->layer[l].skip_type, " ") != 0)
        {
            arch_has_skip = true;
            break;
        }
    }

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
            
            if(strcmp(arch->layer[l].skip_type, "concat") != 0 || l == 0 ||
               strcmp(arch->layer[l].skip_type, "concat_att") != 0)
            {
                true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
            }else{
                iftIntArray *layers_to_concat = iftCreateIntArray(arch->nlayers);
                layers_to_concat->val[0] = l+1; // add current layer
                int nconcat_layers = getLayersToConcat(arch, l+1, l, layers_to_concat, 1);
                nkernels = 0;
                for(int i = 0; i < nconcat_layers; i++)
                {
                    sprintf(filename, "%s/model/conv%d-labels.txt", 
                        model_dir, layers_to_concat->val[i]);
                    if(!iftFileExists(filename))
                        iftError("File %s does not exist", "main", filename);
                    
                    /* load merged model */
                    int nkernels_i = 0;
                    int *truelabels_i = iftGFLIMLoadKernelLabels(filename, &nkernels_i);

                    nkernels += nkernels_i;
                    true_labels = realloc(true_labels, nkernels * sizeof(int));
                    for(int j = 0; j < nkernels_i; j++)
                        true_labels[nkernels - nkernels_i + j] = truelabels_i[j];
                    iftFree(truelabels_i);
                }
                iftDestroyIntArray(&layers_to_concat);
            }
        }

        // extract features
        iftGFLIMEncodeLayerBatch(input, Kmerged, 
            arch->layer[l], 
            label_imgs, bias_merged, 
            stride_type, attention_opt, 
            true_labels, aggregation_opt, 
            kernel_removal);

        iftDestroyMatrix(&Kmerged);
        iftFree(bias_merged);
        iftFree(true_labels);
        true_labels = NULL;

        // process residual connections
        if(l > 0 && arch_has_skip)
        {    
            sprintf(filename, "%s/model/kernels_map.txt", model_dir);
            if (!iftFileExists(filename))
                iftError("Model file not found: %s", __func__, filename);
            iftGFLIMSkipConnectionBatch(input, label_imgs, fs, first, last, 
                                        l + 1, arch, filename, 
                                        model_dir);
        }

        // check if any further layer uses skip connection from this layer
        bool use_layer_skip = false;
        for(int i = l + 1; i < nLayers; i++)
        {
            if(arch->layer[i].skip_connection[l] > 0)
            {
                use_layer_skip = true;
                break;
            }
        }

        // save intermediate graphs and labels
        if (layers_saveGraphAct[l] || use_layer_skip)
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
                sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, finer_layers[l]);
                if (!iftFileExists(filename))
                    iftError("File %s does not exist", __func__, filename);

                if(strcmp(arch->layer[l].skip_type, "concat") != 0 || l == 0 ||
                   strcmp(arch->layer[l].skip_type, "concat_att") != 0)
                    true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
                else{
                    iftIntArray *layers_to_concat = iftCreateIntArray(arch->nlayers);
                    layers_to_concat->val[0] = l+1; // add current layer
                    int nconcat_layers = getLayersToConcat(arch, l+1, l, layers_to_concat, 1);
                    nkernels = 0;
                    for(int i = 0; i < nconcat_layers; i++)
                    {
                        sprintf(filename, "%s/model/conv%d-labels.txt", 
                                                model_dir, 
                                                layers_to_concat->val[i]);
                        if(!iftFileExists(filename))
                            iftError("File %s does not exist", "main", filename);
                            
                        /* load merged model */
                        int nkernels_i = 0;
                        int *truelabels_i = iftGFLIMLoadKernelLabels(filename, &nkernels_i);

                        nkernels += nkernels_i;
                        true_labels = realloc(true_labels, nkernels * sizeof(int));
                        for(int j = 0; j < nkernels_i; j++)
                            true_labels[nkernels - nkernels_i + j] = truelabels_i[j];
                        iftFree(truelabels_i);
                    }
                    iftDestroyIntArray(&layers_to_concat);
                }
            }
            nlabels = iftGFLIMNumberOfTruelabels(true_labels, nkernels);

            for (int k = 0; k < ngraphs_batch; k++)
            {
                iftImage **activations = iftGFLIMGraphToActivations(input->val[k], label_imgs->val[k]);
            
                /* Compute the scale factor w.r.t. the input of the network */
                float scale[3];
                scale[0] = (float)label_imgs->val[k]->xsize / (float)label_imgs0->val[k]->xsize;
                scale[1] = (float)label_imgs->val[k]->ysize / (float)label_imgs0->val[k]->ysize;
                scale[2] = (float)label_imgs->val[k]->zsize / (float)label_imgs0->val[k]->zsize;

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
                    temp[j] = iftInterp2D(activations[j], scale[0], scale[1]);
                    
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
                sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, finer_layers[l]);
                if (!iftFileExists(filename))
                    iftError("File %s does not exist", __func__, filename);

                if(strcmp(arch->layer[l].skip_type, "concat") != 0 || l == 0 ||
                   strcmp(arch->layer[l].skip_type, "concat_att") != 0)
                    true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
                else{
                    iftIntArray *layers_to_concat = iftCreateIntArray(arch->nlayers);
                    layers_to_concat->val[0] = l+1; // add current layer
                    int nconcat_layers = getLayersToConcat(arch, l+1, l, layers_to_concat, 1);
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

                sprintf(output_dir, "%s/%s_%d", model_dir, decoder_name, l + 1);
                if(!iftDirExists(output_dir))
                    iftMakeDir(output_dir);
                
                for (int k = 0; k < ngraphs_batch; k++)
                {
                    iftImageArray *saliency = NULL;
                    iftGFLIMGraph *graph = iftCopyGFLIMGraph(input->val[k]->num_feats,input->val[k], true);
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
    iftDestroyGFLIMGraphArray(&input);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 17 && argc != 18 && argc != 19)
    {
        iftError("Usage: iftGFLIMExtract P1 P2 P3 P4 P5 "
                                        "P6 P7 P8 P9 P10 "
                                        "P11 P12 P13 P14 P15 "
                                        "P16 [P17 P18] \n"
                 // parameters related to input data
                 "P1 : Images directory \n"
                 "P2 : Model directory \n"
                 // parameters related to the network
                 "P3 : Architecture file \n"
                 "P4 : Number of layers \n"
                 "P5 : Batch size (> 0) \n"
                 "P6 : Stride type. See iftGFLIMStrideType in iftGFLIM_BoFP.h for details\n"
                 "P7 : attention. See iftGFLIMGraphAttention in iftGFLIMGraph.h for details\n"
                 "P8 : aggregation. See iftGFLIMAggregation in iftGFLIMGraph.h for details\n"
                 "P9 : reduce kernels (1/0) \n"
                 // parameters related to the graph
                 "P10 : Graph feature. See iftGFLIMGraphNodeFeature in iftGFLIMGraph.h for details\n"
                 "P11 : superpixel method. See iftGFLIMGraphSpxMethod in iftGFLIMGraph.h for details\n"
                 "P12 : Number of superpixels (> 1) \n"
                 // parameters related to output data
                 "P13 : Layers to save the outputs graphs and superpixel maps (1,2,..) \n"
                 "P14 : Layers to save image activations (1,2,..) \n"
                 "P15 : Layers to decode (1,2,..) \n"
                 "P16 : List of decoders to use. See iftGFLIMDecoderType in iftGFLIM_BoFP.h for details\n"
                 // optional parameters
                 "P17 : Initial seeds (only used when superpixel method is DISF). It must be > number of superpixels \n" 
                 "P18 : Use mask (optional, bool). Compute the graph using only pixels inside the mask --- can only be used when superpixel method is DISF \n",
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
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[6]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[7]);
    iftGFLIMAggregation aggregation_opt = iftGFLIMGraphGetAggregation(argv[8]);
    bool kernel_removal = atoi(argv[9]);
    
    // parameters related to the graph
    iftGFLIMGraphNodeFeature graph_feat = iftGFLIMGraphGetNodeFeature(argv[10]);
    iftGFLIMGraphSpxMethod spx_method = iftGFLIMGraphGetSpxMethod(argv[11]);
    int nsuperpixels = atoi(argv[12]);
    
    // parameters related to output data
    int nLayers_saveGraphAct = 0;
    int *list_int_graph_act = SplitIntList(argv[13], &nLayers_saveGraphAct); // layers to save graph activations
    int nLayers_saveImgAct = 0;
    int *list_int_img_act = SplitIntList(argv[14], &nLayers_saveImgAct); // layers to save image activations
    int nLayers_decode = 0;
    int *list_int_decode = SplitIntList(argv[15], &nLayers_decode); // layers to decode
    int nDecoders = 0;
    iftGFLIMDecoderType *decoders = SplitDecoderList(argv[16], &nDecoders); // decoders

    // optional parameters
    int initial_seeds = (argc >= 18) ? atoi(argv[17]) : 0;
    bool use_mask = (argc == 19) ? atoi(argv[18]) : false;

    // read which layers to save activations and decode
    bool *layers_saveGraphAct = (bool *)iftAlloc(nLayers, sizeof(bool));
    for(int i = 0; i < nLayers_saveGraphAct; i++){
        if(list_int_graph_act[i] > 0)
            layers_saveGraphAct[list_int_graph_act[i]-1] = true;
    }
    iftFree(list_int_graph_act);

    bool *layers_saveImgAct = (bool *)iftAlloc(nLayers, sizeof(bool));
    for(int i = 0; i < nLayers_saveImgAct; i++){
        if(list_int_img_act[i] > 0)
            layers_saveImgAct[list_int_img_act[i]-1] = true;
    }
    iftFree(list_int_img_act);

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
        if (argc < 15)
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

    // get the finer residual layer (with more nodes) to be used as reference in the upsampling
    int *finer_layers = iftGFLIMGetFinerResidualLayer(arch);

    printf("\n");
    for (int batch = 0; batch < nbatches; batch++)
    {
        int first = batch * batchsize, last = first + batchsize - 1;
        printf("Processing batch %d of %d batches\r", batch + 1, nbatches);
        fflush(stdout);

        ProcessBatch(fs, first, last, 
            model_dir, arch, nLayers, stride_type,
            initial_seeds, nsuperpixels,
            graph_feat, spx_method,
            attention_opt, aggregation_opt, kernel_removal,
            layers_saveGraphAct, layers_saveImgAct, layers_decode,
            decoders, nDecoders, finer_layers, use_mask);
    }

    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last)
    {
        printf("Processing remaining %d images\r", last - first + 1);
        fflush(stdout);

        ProcessBatch(fs, first, last, 
            model_dir, arch, nLayers, stride_type,
            initial_seeds, nsuperpixels,
            graph_feat, spx_method,
            attention_opt, aggregation_opt, kernel_removal,
            layers_saveGraphAct, layers_saveImgAct, layers_decode,
            decoders, nDecoders, finer_layers, use_mask);
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
    iftFree(finer_layers);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("\nDone ... %s\n", time);
    iftFree(time);
    return 0;
}
