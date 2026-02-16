#include "ift.h"

/**
 * \file iftGFLIMDecodeLayer.c
 * \brief Decode a layer of a GFLIM or GFLIM-BoFP model.
 * \author Isabela Borlido
 */

void validateArgs(int layer, char *model_dir, iftGFLIMDecoderType decoder, iftGFLIMStrideType stride_type, 
                char *input_dir, char *labels_dir, char *input_net_labels, int batchsize, char *output_dir)
{
    if (layer < 1)
        iftError("Layer number must be greater than 0", "main");

    if (!iftDirExists(model_dir))
        iftError("Model directory does not exist: %s", "main", model_dir);

    sprintf(input_dir, "%s/layer%d", model_dir, layer);
    if (!iftDirExists(input_dir))
        iftError("Graph directory does not exist: %s", "main", input_dir);

    if(stride_type != IFT_GFLIM_NOSTRIDE)
    {
        sprintf(labels_dir, "%s/labels%d", model_dir, layer);
        if (!iftDirExists(labels_dir))
            iftError("Labels directory does not exist: %s", "main", labels_dir);
    }else{
        sprintf(labels_dir, "%s/labels0", model_dir);
        if (!iftDirExists(labels_dir))
            iftError("Labels directory does not exist: %s", "main", labels_dir);
    }
    
    sprintf(input_net_labels, "%s/labels0", model_dir);
    if (!iftDirExists(input_net_labels))
        iftError("Labels directory input network does not exist: %s", "main", input_net_labels);

    if (batchsize < 1)
        iftError("Batch size must be greater than 0", "main");

    char *decoder_name = iftGFLIMDecoderToString(decoder); // validate decoder type
    sprintf(output_dir, "%s/%s_%d", model_dir, decoder_name, layer);
    iftFree(decoder_name);
}

void ProcessBatch(iftFileSet *fs, int first, int last, 
                char *model_dir, int layer, int layer_skip,
                iftGFLIMArch *arch, 
                iftGFLIMDecoderType decoder, 
                iftGFLIMStrideType stride_type,
                int nkernels, int *truelabels, int nlabels,
                char *output_dir)
{
    int ngraphs_batch = last - first + 1;
    iftImageArray *label_imgs = iftCreateImageArray(ngraphs_batch);
    iftImageArray *label_imgs0 = iftCreateImageArray(ngraphs_batch);
    iftGFLIMGraphArray *input = iftCreateGFLIMGraphArray(ngraphs_batch);

    char **basenames = (char **)iftAlloc(ngraphs_batch, sizeof(char *));
    for (int k = 0; k <= last - first; k++)
    {
        int i = k + first;
        char ext[10], filename[1024];
        sprintf(ext, "%s", iftFileExt(fs->files[i]->path));
        basenames[k] = iftFilename(fs->files[i]->path, ext);
        sprintf(filename, "%s/layer%d/%s.json", model_dir, layer, basenames[k]);
        input->val[k] = iftReadGFLIMGraph(filename);
        sprintf(filename, "%s/labels%d/%s.pgm", model_dir, layer, basenames[k]);
        label_imgs->val[k] = iftReadImageByExt(filename);
        sprintf(filename, "%s/labels0/%s.pgm", model_dir, basenames[k]);
        label_imgs0->val[k] = iftReadImageByExt(filename);
    }

    for (int k = 0; k <= last - first; k++)
    {
        iftImageArray *saliency = NULL;

        if (decoder != IFT_GFLIM_TRISTATE_DECODER && nkernels == 0)
        {
            char filename[1024];
            int nkernels_i = 0, *truelabels_i = NULL, nlabels_i = 1;
            sprintf(filename, "%s/model/%s-conv%d-labels.txt", model_dir, basenames[k], layer_skip);
            if (iftFileExists(filename))
            {
                /* load image model */
                nkernels_i = input->val[k]->num_feats; 
                truelabels_i = iftGFLIMLoadKernelLabels(filename, &nkernels_i);
                nlabels_i = iftGFLIMNumberOfTruelabels(truelabels_i, nkernels_i);
            }
            else iftError("File %s does not exist", "main", filename);

            // decode the graph
            saliency = iftGFLIMDecodeLayer(input->val[k], 
                                        label_imgs->val[k], 
                                        label_imgs0->val[k], 
                                        truelabels_i, 
                                        nlabels_i, 
                                        decoder);
            iftFree(truelabels_i);
        }else{
            // decode the graph
            saliency = iftGFLIMDecodeLayer(input->val[k], 
                                        label_imgs->val[k], 
                                        label_imgs0->val[k], 
                                        truelabels, 
                                        nlabels, 
                                        decoder);
        }

#pragma omp parallel for
        /* post-process and save the salience maps */
        for (int l = 0; l < saliency->n; l++)
        {
            char sal_dir[1024], filename[1024];
            sprintf(sal_dir, "%s/label%d", output_dir, l + 1);
            if (!iftDirExists(sal_dir)) iftMakeDir(sal_dir);

            if (iftIs3DImage(label_imgs->val[k]))
                sprintf(filename, "%s/%s.nii.gz", sal_dir, basenames[k]);
            else
                sprintf(filename, "%s/%s.png", sal_dir, basenames[k]);
            iftWriteImageByExt(saliency->val[l], filename);
        }
        iftDestroyImageArray(&saliency);
        iftFree(basenames[k]);
    }
    
    iftDestroyImageArray(&label_imgs);
    iftDestroyImageArray(&label_imgs0);
    iftDestroyGFLIMGraphArray(&input);
    iftFree(basenames);
}

int main(int argc, char *argv[])
{
    if (argc != 7)
    {
        iftError("Usage: iftGFLIMGSDecodeLayer P1 P2 P3 P4 P5 P6 \n"
                 "P1 : model directory \n"
                 "P2 : layer number (> 0) \n"
                 "P3 : architecture of the network\n"
                 "P4 : decoder type. See iftGFLIMDecoderType in iftGFLIM_BoFP.h for details\n"
                 "P5 : stride type. See iftGFLIMStrideType in iftGFLIM_BoFP.h for details\n"
                 "P6 : batch size (> 0) \n",
                 "main");
    }

    timer *tstart = iftTic();

    char *model_dir = argv[1];
    int layer = atoi(argv[2]);
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[3]);
    iftGFLIMDecoderType decoder = iftGFLIMGetDecoderType(argv[4]);
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[5]);
    int batchsize = atoi(argv[6]);

    char input_dir[1024], labels_dir[1024], input_net_labels[1024], output_dir[1024];

    validateArgs(layer, model_dir, decoder, stride_type, input_dir, labels_dir, input_net_labels, batchsize, output_dir);

    iftFileSet *fs = NULL;

    if(iftDirExists(input_dir))
        fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", true);
    else{
        if(!iftFileExists(input_dir))
            iftError("Input directory does not exist: %s", "main", input_dir);
        fs = iftLoadFileSetFromCSV(input_dir, false);
    }
    if (fs == NULL || fs->n == 0)
        iftError("No images found in the input directory: %s", "main", input_dir);

    if(!iftDirExists(output_dir))
        iftMakeDir(output_dir);

    // if there are skip connections, find the fist skip connection layer
    int layer_skip = layer;
    for(int i = 0; i < layer; i++){
        if(arch->layer[layer-1].skip_connection[i] > 0){
            layer_skip = i+1;
            break;
        }
    }

    int ngraphs = fs->n;
    int nbatches = 0;
    int nremaining_graphs = fs->n;

    batchsize = iftMin(batchsize, ngraphs);
    nbatches = ngraphs / batchsize;
    nremaining_graphs = nremaining_graphs - nbatches * batchsize;

    int *truelabels = NULL, nlabels = 1, nkernels = 0;
    char filename[1024];

    sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer_skip);
    if (decoder != IFT_GFLIM_TRISTATE_DECODER && iftFileExists(filename))
    {
        if(strcmp(arch->layer[layer-1].skip_type, "concat") != 0 && 
           strcmp(arch->layer[layer-1].skip_type, "concat_att") != 0)
        {
            truelabels = iftGFLIMLoadKernelLabels(filename, &nkernels);
            nlabels = iftGFLIMNumberOfTruelabels(truelabels, nkernels);
        }else{
            iftIntArray *layers_to_concat = iftCreateIntArray(arch->nlayers);
            layers_to_concat->val[0] = layer; // add current layer
            int nconcat_layers = getLayersToConcat(arch, layer, layer-1, layers_to_concat, 1);
                
            for(int i = 0; i < nconcat_layers; i++)
            {
                sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layers_to_concat->val[i]);
                if(!iftFileExists(filename))
                    iftError("File %s does not exist", __func__, filename);
                    
                /* load merged model */
                int nkernels_i = 0;
                int *truelabels_i = iftGFLIMLoadKernelLabels(filename, &nkernels_i);
                
                nkernels += nkernels_i;
                truelabels = realloc(truelabels, nkernels * sizeof(int));
                for(int j = 0; j < nkernels_i; j++)
                    truelabels[nkernels - nkernels_i + j] = truelabels_i[j];
                iftFree(truelabels_i);
            }
            iftDestroyIntArray(&layers_to_concat);
            nlabels = iftGFLIMNumberOfTruelabels(truelabels, nkernels);
        }
    }

    for (int batch = 0; batch < nbatches; batch++)
    {
        int first = batch * batchsize, last = first + batchsize - 1;
        printf("Processing batch %d of %d batches\r", batch + 1, nbatches);
        fflush(stdout);

        ProcessBatch(fs, first, last, 
            model_dir, layer, layer_skip, arch, 
            decoder, stride_type,
            nkernels, truelabels, nlabels,
            output_dir);
    }

    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last)
    {
        printf("Processing remaining %d images\r", last - first + 1);
        fflush(stdout);

        ProcessBatch(fs, first, last, 
            model_dir, layer, layer_skip, arch, 
            decoder, stride_type,
            nkernels, truelabels, nlabels,
            output_dir);
    }
    
    if(nkernels == 0) iftFree(truelabels);
    iftDestroyFileSet(&fs);
    iftDestroyGFLIMArch(&arch);
    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
