#include "ift.h"

/**
 * \file iftGFLIMExtract.c
 * \brief Extract features from a set of images and decode from a GFLIM or a GFLIM-BoFP trained model.
 */

char* read_file_scales(const char* filename) { 
    FILE* file = fopen(filename, "r"); 
    if (!file) { 
        perror("Failed to open file"); 
        return NULL; 
    } 
 
    // Allocate initial memory for the file content 
    size_t capacity = 1024; 
    char* content = malloc(capacity); 
    if (!content) { 
        perror("Failed to allocate memory"); 
        fclose(file); 
        return NULL; 
    } 
 
    size_t length = 0; 
    size_t bytesRead; 
 
    // Read the file in chunks 
    while ((bytesRead = fread(content + length, sizeof(char), 1024, file)) > 0) { 
        length += bytesRead; 
 
        // If we reach capacity, resize the buffer 
        if (length + 1024 > capacity) { 
            capacity *= 2; // Double the capacity 
            content = realloc(content, capacity); 
            if (!content) { 
                perror("Failed to reallocate memory"); 
                fclose(file); 
                return NULL; 
            } 
        } 
    } 
 
    // Null-terminate the string 
    if (length < capacity) { 
        content[length] = '\0'; 
    } else { 
        content = realloc(content, length + 1); 
        content[length] = '\0'; 
    } 
 
    fclose(file); 
    return content; 
} 


iftIntArray **iftGFLIMCompute_kernel_maps_scales(char *filename)
{   
    char *buf = read_file_scales(filename);

    int nlines = 100, line_count = 0;
    const char line_delimiter[] = "\n";
    const char number_delimiter[] = " ";
    char *line, *endptr; 
    int nkernels = 100, nkernels0 = 0, kernel_count = 0;

    char* line_saveptr = NULL;
    char* number_saveptr = NULL;

    int **kernel_maps_scale0 = iftAlloc(nlines, sizeof(int *));

    // Get the first line 
    line = strtok_r(buf, line_delimiter, &line_saveptr); 

    while(line != NULL)
    {
        // break into numbers
        char* inner_token = strtok_r(line, number_delimiter, &number_saveptr);

        // allocate memory for more lines if needed
        if(line_count >= nlines)
        {
            nlines += 100;
            kernel_maps_scale0 = realloc(kernel_maps_scale0, nlines * sizeof(int *));
            if(kernel_maps_scale0 == NULL)
                iftError("out of memory, nlines=%d", __func__, nlines);
        }

        kernel_count = 0, nkernels = 100;
        kernel_maps_scale0[line_count] = iftAllocIntArray(nkernels);
        
        // get each number
        while (inner_token != NULL) {

            // allocate memory for more kernels if needed
            if(kernel_count >= nkernels)
            {
                nkernels += 100;
                kernel_maps_scale0[line_count] = realloc(kernel_maps_scale0[line_count], nkernels * sizeof(int));
                if(kernel_maps_scale0[line_count] == NULL)
                    iftError("out of memory, nkernels=%d", __func__, nkernels);
            }
            kernel_maps_scale0[line_count][kernel_count] = (int) strtol(inner_token,&endptr,10);
            kernel_count++;
            inner_token = strtok_r(NULL, number_delimiter, &number_saveptr);
        }
        if(nkernels != kernel_count)
            kernel_maps_scale0[line_count] = realloc(kernel_maps_scale0[line_count], kernel_count * sizeof(int));
        if(line_count == 0)
            nkernels0 = kernel_count;
        line_count++;
        line = strtok_r(NULL, line_delimiter, &line_saveptr);
    }
    nlines = line_count;
    free(buf);

    printf("nlines: %d, nkernels0: %d\n", nlines, nkernels0);

    if(line_count == 0)
        iftError("No lines found in the file", __func__);
    if(nkernels0 == 0)
        iftError("No kernels found in the first line of the file", __func__);

    iftIntArray **kernel_maps = iftAlloc(nlines+1, sizeof(iftIntArray *));
    
    // compute the kernel maps for all scales
    // from scale 0 to scale i
    
    kernel_maps[0] = iftCreateIntArray(nkernels0); 
    for(int i=0; i < nkernels0; i++)
        kernel_maps[0]->val[i] = i;
    for(int l = 0; l < nlines; l++)
    {
        kernel_maps[l+1] = iftCreateIntArray(nkernels0); 
        int nkernelsi = 0;
        for(int i=0; i < nkernels0; i++)
        {
            int k0 = kernel_maps_scale0[l][i] - 1;
            if(k0 == i)
            {
                kernel_maps[l+1]->val[i] = nkernelsi;
                nkernelsi++;
            }else
                kernel_maps[l+1]->val[i] = kernel_maps[l+1]->val[k0];
        }
        //printf("line %d: nkernels: %d\n", l, nkernelsi);
        iftFree(kernel_maps_scale0[l]);
    }   
    iftFree(kernel_maps_scale0);

    return kernel_maps;
}


iftGFLIMGraphArray *iftGFLIMMergeScales(char **graphs_dir_list, char **labels_dir_list, 
                        int nscales, char **basenames,
                        iftFileSet *fs, int first, int last, 
                        char *model_dir, int *superpixels_list,
                        char *merge_option, char *kernels_file, 
                        iftGFLIMArch *arch,
                        iftGFLIMStrideType stride_type, iftGFLIMGraphAttention attention_opt,
                        iftGFLIMAggregation aggregation_opt, 
                        bool kernel_removal,
                        iftImageArray *spx_imgs_output)
{
    int ngraphs_batch = last - first + 1;

    // compute the kernel maps for the layers that will be used in the skip connection
    iftIntArray **kernels_maps = iftGFLIMCompute_kernel_maps_scales(kernels_file); // compute the kernel maps for all layers

    iftGFLIMGraphArray **input = iftAlloc(nscales, sizeof(iftGFLIMGraphArray *));
    iftImageArray **spx_imgs = iftAlloc(nscales, sizeof(iftImageArray *));

    for(int j = 0; j < nscales; j++)
    {
        input[j] = iftCreateGFLIMGraphArray(ngraphs_batch);
        spx_imgs[j] = iftCreateImageArray(ngraphs_batch);

        // read graphs and labels of the current scale
        #pragma omp parallel for
        for (int k = 0; k < ngraphs_batch; k++)
        {
            char filename[1024];
            
            // read graph
            sprintf(filename, "%s/%s.json", graphs_dir_list[j], basenames[k]);
            input[j]->val[k] = iftReadGFLIMGraph(filename);

            // read label
            sprintf(filename, "%s/%s.pgm", labels_dir_list[j], basenames[k]);
            spx_imgs[j]->val[k] = iftReadImageByExt(filename);
        }

        // read kernel weights, bias, and labels
        iftMatrix *Kmerged = NULL;
        float *bias_merged = NULL;
        int *true_labels = NULL;
        char filename[1024];
        int nkernels = 0;

        sprintf(filename, "%s/model/spx%d-conv%d-bias.txt", model_dir, superpixels_list[j], 1);
        if (iftFileExists(filename)) // is there a bias?
        {
            /* Load kernel and bias */
            sprintf(filename, "%s/model/spx%d-conv%d", model_dir, superpixels_list[j], 1);
            iftGFLIMLoadMergedModel(filename, &Kmerged, &bias_merged);
        }
        else
        {
            /* Load kernel */
            sprintf(filename, "%s/model/spx%d-conv%d-kernels.npy", model_dir, superpixels_list[j], 1);
            if (!iftFileExists(filename))
                iftError("Model file not found: %s", __func__, filename);
            Kmerged = iftReadMatrix(filename);
            int ninput_channels = input[j]->val[0]->num_feats;

            /* Load mean and standard deviation */
            float *mean = iftAllocFloatArray(ninput_channels);
            float *stdev = iftAllocFloatArray(ninput_channels);
            sprintf(filename, "%s/model/spx%d-conv%d", model_dir, superpixels_list[j], 1);
            iftReadMeanStdev(filename, mean, stdev, ninput_channels);

            /* Normalization (because there is no bias)*/
    #pragma omp parallel for
            for (int k = 0; k < ngraphs_batch; k++){
                iftGFLIMGraphZNormalization(input[j]->val[k], mean, stdev);
            }

            iftFree(mean);
            iftFree(stdev);
        }
    
        /* Load labels indicating class (label) per channel */
        if (attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE && bias_merged != NULL)
        {
            sprintf(filename, "%s/model/spx%d-conv%d-labels.txt", model_dir, superpixels_list[j], 1);
            if (!iftFileExists(filename))
                iftError("File %s does not exist", __func__, filename);
            true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
        }

        iftGFLIMEncodeLayerBatch_v2(input[j], Kmerged, arch->layer[1], spx_imgs[j], bias_merged, stride_type, attention_opt, true_labels, aggregation_opt, kernel_removal);
        iftFree(true_labels); 
        iftDestroyMatrix(&Kmerged);
        iftFree(bias_merged);
    }

    iftGFLIMGraphArray *output = iftCreateGFLIMGraphArray(ngraphs_batch);

    // merge scales
    for (int k = 0; k <= last - first; k++)
    {
        int **node_maps = iftAlloc(nscales, sizeof(int *));
        iftMatrix **matrices = iftAlloc(nscales, sizeof(iftMatrix *));
        
        iftGFLIMMaxNormGraph(input[0]->val[k]);

        node_maps[0] = iftAlloc(input[0]->val[k]->num_nodes, sizeof(int));
        for(int i = 0; i < input[0]->val[k]->num_nodes; i++)
            node_maps[0][i] = i;
        matrices[0] = iftGFLIMGraphToNodeFeaturesMatrix(input[0]->val[k]);

        for(int j = 1; j < nscales; j++)
        {
            iftGFLIMMaxNormGraph(input[j]->val[k]);
            matrices[j] = iftGFLIMGraphToNodeFeaturesMatrix(input[j]->val[k]);
            iftDestroyGFLIMGraph(&input[j]->val[k]);

            node_maps[j] = iftGFLIMComputeNodeMap(spx_imgs[j]->val[k], spx_imgs[0]->val[k]);
            iftDestroyImage(&spx_imgs[j]->val[k]);

            iftMatrix *tmp = iftGFLIMUpsampleFeatureMatrix(matrices[j], node_maps[j], kernels_maps[j], matrices[0]->nrows, matrices[0]->ncols);
            iftDestroyMatrix(&matrices[j]);
            matrices[j] = tmp;
            tmp = NULL;
        }

        // merge the feature matrices
        iftMatrix *merged = NULL;
        if(strcmp(merge_option, "dotprod") == 0)
            merged = iftGFLIMMergeMatrices_dotProd(matrices, nscales, 0, false, false);
        else if(strcmp(merge_option, "dotprod_sumscale0") == 0)
            merged = iftGFLIMMergeMatrices_dotProd_sumScale0(matrices, nscales, 0, false, false);
        else if(strcmp(merge_option, "dotprod_sumscalei") == 0)
            merged = iftGFLIMMergeMatrices_dotProd_sumScalei(matrices, nscales, 0, false, false);
        else if(strcmp(merge_option, "sum") == 0)
            merged = iftGFLIMMergeMatrices_sum(matrices, nscales, 0, false, false, -1);
        else if(strcmp(merge_option, "sum_sumscale0") == 0)
            merged = iftGFLIMMergeMatrices_sum(matrices, nscales, 0, false, false, 0);
        else if(strcmp(merge_option, "sum_sumscalei") == 0)
            merged = iftGFLIMMergeMatrices_sum(matrices, nscales, 0, false, false, nscales-1);
        else
            iftError("Unknown merge function: %s",  __func__, merge_option);

        for(int j = 0; j < nscales; j++)
        {
            iftDestroyMatrix(&matrices[j]);
            iftFree(node_maps[j]);
        }
        iftFree(matrices);
        iftFree(node_maps);

        // copy the merged features to the graph of the current layer
        if(input[0]->val[k]->num_nodes != merged->nrows || input[0]->val[k]->num_feats != merged->ncols)
            iftError("Graph and merged matrix dimensions do not match: (%ld, %ld) != (%d, %d)", __func__, 
                    input[0]->val[k]->num_nodes, input[0]->val[k]->num_feats, merged->nrows, merged->ncols); 
        for(int i = 0; i < input[0]->val[k]->num_nodes; i++)
        {
            for(int j = 0; j < input[0]->val[k]->num_feats; j++)
                input[0]->val[k]->nodes[i].feats[j] = iftMatrixElem(merged, j, i);
        }
        // min-max normalization per channel
        /*for(int j = 0; j < graph_finer->num_feats; j++)
        {
            float min = iftMatrixElem(merged, j, 0);
            float max = iftMatrixElem(merged, j, 0);
            for(int i = 1; i < graph_finer->num_nodes; i++)
            {
                if(iftMatrixElem(merged, j, i) < min)
                    min = iftMatrixElem(merged, j, i);
                if(iftMatrixElem(merged, j, i) > max)
                    max = iftMatrixElem(merged, j, i);
            }
            for(int i = 0; i < graph_finer->num_nodes; i++)
                graph_finer->nodes[i].feats[j] = (iftMatrixElem(merged, j, i) - min) / (max - min);
        }*/
        output->val[k] = input[0]->val[k];
        spx_imgs_output->val[k] = spx_imgs[0]->val[k];
        input[0]->val[k] = NULL;
        spx_imgs[0]->val[k] = NULL;
        iftDestroyMatrix(&merged);
   }

    iftFree(spx_imgs);
    iftFree(input);
    for(int j = 0; j < nscales; j++){
        iftDestroyIntArray(&kernels_maps[j]);
    }
    iftFree(kernels_maps);
    return output;
}


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

char **SplitStringList(char *str, int *nvalues)
{
    *nvalues = 100;
    char **values = (char **)iftAlloc(*nvalues, sizeof(char *));
    char *token = strtok(str, ",");
    int i = 0;
    while (token != NULL && i < *nvalues)
    {
        if(i == *nvalues)
        {
            *nvalues += 100;
            values = (char **)iftRealloc(values, *nvalues * sizeof(char *));
        }
        values[i++] = iftAllocCharArray(strlen(token)+1);
        strcpy(values[i-1], token);
        token = strtok(NULL, ",");
    }
    if(i != *nvalues)
        values = (char **)iftRealloc(values, i * sizeof(char *));
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

void SaveIntermediate(char *model_dir, int layer, 
                iftGFLIMGraphArray *input, iftImageArray *label_imgs, 
                int ngraphs_batch, char **basenames)
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

void SaveActivates(char *model_dir, int layer, 
            iftGFLIMGraphArray *input, iftImageArray *label_imgs, iftImageArray *label_imgs0,
            int ngraphs_batch, char **basenames,
            int *finer_layers, int *true_labels, int *nkernels)
{
    char filename[1024];
    if(true_labels == NULL)
    {
        sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, finer_layers[layer]);
        if (!iftFileExists(filename))
            iftError("File %s does not exist", __func__, filename);
        true_labels = iftGFLIMLoadKernelLabels(filename, nkernels);
    }
    int nlabels = iftGFLIMNumberOfTruelabels(true_labels, input->val[0]->num_feats);

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

void DecodeLayer(char *model_dir, int layer, 
            iftGFLIMGraphArray *input, iftImageArray *label_imgs, iftImageArray *label_imgs0, 
            int ngraphs_batch, char **basenames, int *finer_layers, int *true_labels, int *nkernels,
            int nDecoders, iftGFLIMDecoderType *decoders)
{
    char filename[1024];
    if(true_labels == NULL)
    {
        sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, finer_layers[layer]);
        if (!iftFileExists(filename))
            iftError("File %s does not exist", __func__, filename);
        true_labels = iftGFLIMLoadKernelLabels(filename, nkernels);
    }
    int nlabels = iftGFLIMNumberOfTruelabels(true_labels, input->val[0]->num_feats);

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

void ProcessBatch(iftFileSet *fs, int first, int last, 
                char **graphs_dir_list, char **labels_dir_list, int nscales_graphs,
                int *superpixels_list,
                char *model_dir, iftGFLIMArch *arch, int nLayers, 
                iftGFLIMStrideType stride_type, iftGFLIMGraphAttention attention_opt, 
                iftGFLIMAggregation aggregation_opt, 
                bool kernel_removal, bool save_intermediate, 
                bool *layers_saveActivations, bool *layers_decode,
                iftGFLIMDecoderType *decoders, int nDecoders,
                int *finer_layers, char *merge_option, char *kernels_file)
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

    iftGFLIMGraphArray *input = NULL;
    iftImageArray *label_imgs = iftCreateImageArray(ngraphs_batch);
    iftImageArray *label_imgs0 = NULL;
    char **basenames = (char **)iftAlloc(ngraphs_batch, sizeof(char *));

    if(require_label_imgs0)
        label_imgs0 = iftCreateImageArray(ngraphs_batch);
    
    /* load graphs and labels of the first scale*/
    #pragma omp parallel for
    for (int k = 0; k <= last - first; k++)
    {
        int i = k + first;
        char ext[10], filename[1024];
        sprintf(ext, "%s", iftFileExt(fs->files[i]->path));
        basenames[k] = iftFilename(fs->files[i]->path, ext);
        if(require_label_imgs0)
        {
            sprintf(filename, "%s/%s.pgm", labels_dir_list[0], basenames[k]);
            label_imgs0->val[k] = iftReadImageByExt(filename);
        }
    }

    input = iftGFLIMMergeScales(graphs_dir_list, labels_dir_list, nscales_graphs, basenames,
                                fs, first, last, model_dir, superpixels_list,
                                merge_option, kernels_file, arch,
                                stride_type, attention_opt, aggregation_opt,
                                kernel_removal, label_imgs);

    bool use_layer_skip = false;
    for(int i = 1; i < nLayers; i++)
    {
        if(arch->layer[i].skip_connection[0] > 0)
        {
            use_layer_skip = true;
            break;
        }
    }

    int *true_labels = NULL;
    char filename[1024];
    int nkernels = 0;

    nkernels = input->val[0]->num_feats;

    /* Load labels indicating class (label) per channel */
    sprintf(filename, "%s/model/conv%d-bias.txt", model_dir, 1); // bias file
    if ((attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE && iftFileExists(filename)) ||
        layers_decode[0] || layers_saveActivations[0])
    {
        sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, 1);
        if (!iftFileExists(filename))
            iftError("File %s does not exist", __func__, filename);
        true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
    }
    
    // save intermediate graphs and labels
    if (save_intermediate || use_layer_skip)
        SaveIntermediate(model_dir, 0, input, label_imgs, ngraphs_batch, basenames);

    // save activations
    if(layers_saveActivations[0])
        SaveActivates(model_dir, 0, input, label_imgs, label_imgs0, ngraphs_batch, basenames, finer_layers, true_labels, &nkernels);
        
    // decode layer
    if(layers_decode[0])
        DecodeLayer(model_dir, 0, input, label_imgs, label_imgs0, ngraphs_batch, basenames, finer_layers, true_labels, &nkernels, nDecoders, decoders);
        
    iftFree(true_labels);


    /* extract features for each layer */
    for (int layer = 1; layer < nLayers; layer++)
    {
        iftMatrix *Kmerged = NULL;
        float *bias_merged = NULL;
        int *true_labels = NULL;
        char filename[1024];
        int nkernels = 0;

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

        use_layer_skip = false;
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
            SaveIntermediate(model_dir, layer, input, label_imgs, ngraphs_batch, basenames);

        // save activations
        if(layers_saveActivations[layer])
            SaveActivates(model_dir, layer, input, label_imgs, label_imgs0, ngraphs_batch, basenames, finer_layers, true_labels, &nkernels);
        
        // decode layer
        if(layers_decode[layer])
            DecodeLayer(model_dir, layer, input, label_imgs, label_imgs0, ngraphs_batch, basenames, finer_layers, true_labels, &nkernels, nDecoders, decoders);
        
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

    if (argc != 19)
    {
        iftError("Usage: iftGFLIMExtractMultiscale <P1> <P2> <P3> <P4> <P5> <P6> <P7> <P8> <P9> <P10> <P11> <P12> <P13> <P14> <P15> <P16> <P17> <P18> \n"
                // parameters related to input data
                "[ 1] Images directory \n"
                "[ 2] Graphs directories list \n"
                "[ 3] Labels directories list \n"
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
                "[16] Merge option (dotprod, dotprod_sumscale0, dotprod_sumscalei, sum) \n"
                "[17] Superpixels list \n"
                "[18] Kernel maps file \n"
                "argc: %d", "main", argc);
    }
    tstart = iftTic();

    // parameters related to input data 
    char *images_dir = argv[1];
    char *graphs_dir = argv[2];
    char *labels_dir = argv[3];
    char *model_dir = argv[4];

    int nscales_graphs = 0, nscales_labels = 0;
    char **graphs_dir_list = SplitStringList(graphs_dir, &nscales_graphs);
    char **labels_dir_list = SplitStringList(labels_dir, &nscales_labels);
    if(nscales_graphs != nscales_labels)
        iftError("Number of graph scales (%d) is different from number of label scales (%d)", __func__, nscales_graphs, nscales_labels);
    
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
    char *merge_option = argv[16];
    int nSuperpixels = 0;
    int *superpixels_list = SplitIntList(argv[17], &nSuperpixels);
    char *kernels_file = argv[18];

    if(nSuperpixels != nscales_graphs)
        iftError("Number of superpixel scales (%d) is different from number of graphs directories provided (%d)", __func__, nSuperpixels, nscales_graphs);


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

    if(strcmp(merge_option, "dotprod") != 0 && strcmp(merge_option, "dotprod_sumscale0") != 0 &&
       strcmp(merge_option, "dotprod_sumscalei") != 0 && strcmp(merge_option, "sum") != 0)
        iftError("Invalid merge option: %s", __func__, merge_option);

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

        ProcessBatch(fs_imgs, first, last, 
            graphs_dir_list, labels_dir_list, nscales_graphs, superpixels_list,
            model_dir, arch, nLayers, stride_type,
            attention_opt, aggregation_opt, kernel_removal,
            save_intermediate, layers_saveAct, layers_decode,
            decoders, nDecoders, finer_layers, merge_option, kernels_file);
    }

    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last)
    {
        printf("                                                        \r");
        printf("Processing remaining %d images\r", last - first + 1);
        fflush(stdout);

        ProcessBatch(fs_imgs, first, last, 
            graphs_dir_list, labels_dir_list, nscales_graphs, superpixels_list,
            model_dir, arch, nLayers, stride_type,
            attention_opt, aggregation_opt, kernel_removal,
            save_intermediate, layers_saveAct, layers_decode,
            decoders, nDecoders, finer_layers, merge_option, kernels_file);
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
