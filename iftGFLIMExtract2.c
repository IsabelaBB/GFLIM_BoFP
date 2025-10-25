#include "ift.h"

/**
 * \file iftGFLIMExtract.c
 * \brief Extract features from a set of images and decode from a GFLIM or a GFLIM-BoFP trained model.
 */

void validateArgs(char *images_dir, char *model_dir, iftGFLIMArch *arch, int nLayers, 
                  int batchsize, 
                  int argc, char *argv[])
{
    if (nLayers < 1)
        iftError("Invalid number of layers: %d", __func__, nLayers);

    if (nLayers > arch->nlayers)
        iftError("Invalid number of layers: %d", __func__, nLayers);

    if (!iftDirExists(images_dir))
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
                char *graphs_dir, char *labels_dir, char *model_dir, 
                iftGFLIMArch *arch, int nLayers, 
                iftGFLIMStrideType stride_type,
                iftGFLIMGraphAttention attention_opt, 
                iftGFLIMAggregation aggregation_opt, 
                bool kernel_removal,
                bool save_intermediate, 
                bool *layers_saveActivations, bool *layers_decode,
                iftGFLIMDecoderType *decoders, int nDecoders,
                int *finer_layers)
{
    int ngraphs_batch = last - first + 1;

    bool require_label_imgs0 = false;
    for (int layer = 0; layer < nLayers; layer++)
    {
        if(layers_saveActivations[layer] || layers_decode[layer]){
            require_label_imgs0 = true;
            break;
        }
    }

    iftGFLIMGraphArray *input = iftCreateGFLIMGraphArray(ngraphs_batch);
    iftImageArray *label_imgs = iftCreateImageArray(ngraphs_batch);
    iftImageArray *label_imgs0 = NULL;
    char **basenames = (char **)iftAlloc(ngraphs_batch, sizeof(char *));

    if(require_label_imgs0)
        label_imgs0 = iftCreateImageArray(ngraphs_batch);
    
    /* load graphs and labels */
    #pragma omp parallel for
    for (int k = 0; k <= last - first; k++)
    {
        int i = k + first;
        char ext[10], filename[1024];
        sprintf(ext, "%s", iftFileExt(fs->files[i]->path));
        basenames[k] = iftFilename(fs->files[i]->path, ext);
        sprintf(filename, "%s/%s.json", graphs_dir, basenames[k]);
        input->val[k] = iftReadGFLIMGraph(filename);
        sprintf(filename, "%s/%s.pgm", labels_dir, basenames[k]);
        label_imgs->val[k] = iftReadImageByExt(filename);
        if(require_label_imgs0)
        {
            sprintf(filename, "%s/labels0/%s.pgm", model_dir, basenames[k]);
            label_imgs0->val[k] = iftReadImageByExt(filename);
        }
    }

    /* extract features for each layer */
    for (int layer = 0; layer < nLayers; layer++)
    {
        iftMatrix *Kmerged = NULL;
        float *bias_merged = NULL;
        int *true_labels = NULL;
        char filename[1024];
        int nkernels = 0;
        int nlabels = 1;

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
#pragma omp parallel for
            for (int i = 0; i < input->n; i++){
                iftGFLIMGraphZNormalization(input->val[i], mean, stdev);
            }

            iftFree(mean);
            iftFree(stdev);
        }

        /* Load labels indicating class (label) per channel */
        if ((attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE && bias_merged != NULL) ||
            layers_decode[layer] || layers_saveActivations[layer])
        {
            sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer + 1);
            if (!iftFileExists(filename))
                iftError("File %s does not exist", __func__, filename);
            true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
        }

        iftGFLIMEncodeLayerBatch_v2(input, Kmerged, arch->layer[layer], label_imgs, bias_merged, stride_type, attention_opt, true_labels, aggregation_opt, kernel_removal);
        iftFree(true_labels); true_labels = NULL;
        iftDestroyMatrix(&Kmerged);
        iftFree(bias_merged);

        // process residual connections
        if(layer > 0)
        {    
            sprintf(filename, "%s/model/kernels_map.txt", model_dir);
            if (!iftFileExists(filename))
                iftError("Model file not found: %s", __func__, filename);
            iftGFLIMSkipConnectionBatch(input, label_imgs, fs, first, last, layer + 1, arch, filename, model_dir);
        }

        bool use_layer_skip = false;
        for(int i = layer + 1; i < nLayers; i++)
        {
            if(arch->layer[i].skip_connection[layer] > 0)
            {
                use_layer_skip = true;
                break;
            }
        }

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
            #pragma omp parallel for
            for (int k = 0; k < ngraphs_batch; k++)
            {
                char filename[1024];
                sprintf(filename, "%s/layer%d/%s.json", model_dir, layer + 1, basenames[k]);
                iftWriteGFLIMGraph(input->val[k], filename);
                sprintf(filename, "%s/labels%d/%s.pgm", model_dir, layer + 1, basenames[k]);
                iftWriteImageByExt(label_imgs->val[k], filename);
            }
        }

        // save activations
        if(layers_saveActivations[layer])
        {
            if(true_labels == NULL)
            {
                sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, finer_layers[layer]);
                if (!iftFileExists(filename))
                    iftError("File %s does not exist", __func__, filename);
                true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
            }
            nlabels = iftGFLIMNumberOfTruelabels(true_labels, input->val[0]->num_feats);

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
                    sprintf(act_dir, "%s/activation%d/label%d/%s", model_dir, layer + 1, label + 1, basenames[k]);
                    if (!iftDirExists(act_dir)) iftMakeDir(act_dir);
                }

                iftImage **temp = (iftImage **)iftAlloc(input->val[k]->num_feats, sizeof(iftImage *));
                #pragma omp parallel for
                for (int j = 0; j < input->val[k]->num_feats; j++)
                {
                    char filename[1024];
                    temp[j] = iftInterp2D(activations[j], scale[0], scale[1]);
                    
                    int label = 1;
                    if(true_labels != NULL) label = true_labels[j];

                    sprintf(filename, "%s/activation%d/label%d/%s/kernel-%03d.pgm", model_dir, layer + 1, label, basenames[k], j + 1);
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
        if(layers_decode[layer])
        {
            if(true_labels == NULL)
            {
                sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, finer_layers[layer]);
                if (!iftFileExists(filename))
                    iftError("File %s does not exist", __func__, filename);
                true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
            }
            nlabels = iftGFLIMNumberOfTruelabels(true_labels, input->val[0]->num_feats);

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
                    saliency = iftGFLIMDecodeLayer(graph, label_imgs->val[k], label_imgs0->val[k], true_labels, nlabels, decoders[d]);
                    iftDestroyGFLIMGraph(&graph);
                    
                    /* post-process and save the salience maps */
                    #pragma omp parallel for
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
    if(!save_intermediate)
    {
        for (int layer = 1; layer <= nLayers; layer++)
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
    iftDestroyGFLIMGraphArray(&input);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 16)
    {
        iftError("Usage: iftGFLIMExtract <P1> <P2> <P3> <P4> <P5> <P6> <P7> <P8> <P9> <P10> <P11> <P12> <P13> <P14> <P15> \n"
                // parameters related to input data
                "[ 1] Images directory \n"
                "[ 2] Graphs directory \n"
                "[ 3] Labels directory \n"
                "[ 4] Model directory \n"
                // parameters related to the network
                "[ 5] Architecture file \n"
                "[ 6] Number of layers \n"
                "[ 7] Batch size (> 0) \n"
                "[ 8] Stride type (none, image-based, graph-based) \n"
                "[ 9] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE) \n"
                "[10] aggregation (CONCAT, MEAN, WMEAN) \n"
                "[11] reduce kernels (1/0) \n"
                // parameters related to output data
                "[12] Save graphs and labels after each layer (1/0) \n"
                "[13] Layers to save activations (1,2,..) \n"
                "[14] Layers to decode (1,2,..) \n"
                "[15] Decoders (tristate, hybrid, graph-adaptive, gaussian) \n"
                "argc: %d", "main", argc);
    }
    tstart = iftTic();

    // parameters related to input data 
    char *images_dir = argv[1];
    char *graphs_dir = argv[2];
    char *labels_dir = argv[3];
    char *model_dir = argv[4];

    // parameters related to the network
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[5]);
    int nLayers = atoi(argv[6]);
    int batchsize = atoi(argv[7]);
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[8]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[9]);
    iftGFLIMAggregation aggregation_opt = iftGLIMGraphGetAggregation(argv[10]);
    bool kernel_removal = atoi(argv[11]);

    // parameters related to output data
    bool save_intermediate = atoi(argv[12]); // save graphs and labels after each layer
    int nLayers_saveAct = 0;
    int *list_int_act = SplitIntList(argv[13], &nLayers_saveAct); // layers to save activations
    int nLayers_decode = 0;
    int *list_int_decode = SplitIntList(argv[14], &nLayers_decode); // layers to decode
    int nDecoders = 0;
    iftGFLIMDecoderType *decoders = SplitDecoderList(argv[15], &nDecoders); // decoders

    /*printf("Layers to save activations: ");
    for(int i = 0; i < nLayers_saveAct; i++)
        printf("%d ", list_int_act[i]);
    printf("\n");

    printf("Layers to decode: ");
    for(int i = 0; i < nLayers_decode; i++)
        printf("%d ", list_int_decode[i]);
    printf("\n");*/

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

    iftFileSet *fs_imgs = iftLoadFileSetFromDir(images_dir, 1);
    int ngraphs = fs_imgs->n;
    int nbatches = 0;
    int nremaining_graphs = fs_imgs->n;

    if (fs_imgs->n < 1)
        iftError("Image list is empty", __func__);

    batchsize = iftMin(batchsize, ngraphs);
    nbatches = ngraphs / batchsize;
    nremaining_graphs = nremaining_graphs - nbatches * batchsize;

    // get the finer residual layer (with more nodes) to be used as reference in the upsampling
    int *finer_layers = iftGFLIMGetFinerResidualLayer(arch);

    for (int batch = 0; batch < nbatches; batch++)
    {
        int first = batch * batchsize, last = first + batchsize - 1;
        printf("                                                        \r");
        printf("Processing batch %d of %d batches\r", batch + 1, nbatches);
        fflush(stdout);

        ProcessBatch(fs_imgs, first, last, graphs_dir, labels_dir, model_dir, 
                     arch, nLayers, stride_type,
                     attention_opt, aggregation_opt, kernel_removal,
                     save_intermediate, layers_saveAct, layers_decode,
                     decoders, nDecoders, finer_layers);
    }

    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last)
    {
        printf("                                                        \r");
        printf("Processing remaining %d images\r", last - first + 1);
        fflush(stdout);

        ProcessBatch(fs_imgs, first, last, graphs_dir, labels_dir, model_dir, 
                     arch, nLayers, stride_type,
                     attention_opt, aggregation_opt, kernel_removal,
                     save_intermediate, layers_saveAct, layers_decode,
                     decoders, nDecoders, finer_layers);
    }

    char dir[1024];
    sprintf(dir, "%s/layer0", model_dir);
    iftRemoveDir(dir);
    sprintf(dir, "%s/labels0", model_dir);
    iftRemoveDir(dir);

    iftDestroyFileSet(&fs_imgs);
    iftDestroyGFLIMArch(&arch);
    iftFree(layers_saveAct);
    iftFree(layers_decode);
    iftFree(decoders);
    iftFree(finer_layers);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return 0;
}
