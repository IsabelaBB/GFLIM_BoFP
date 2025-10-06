#include "ift.h"

/**
 * @brief Decode a layer of a GFLIM or GFLIM-BoFP model.
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

    switch (decoder)
    {
    case IFT_GFLIM_TRISTATE_DECODER:
        sprintf(output_dir, "%s/tristate_%d", model_dir, layer);
        break;
    
    case IFT_GFLIM_HYBRID_DECODER:
        sprintf(output_dir, "%s/hybrid_%d", model_dir, layer);
        break;

    case IFT_GFLIM_GRAPHADAPTIVE_DECODER:
        sprintf(output_dir, "%s/graph-adaptive_%d", model_dir, layer);
        break;

    case IFT_GFLIM_GAUSSIAN_DECODER:
        sprintf(output_dir, "%s/gaussian_%d", model_dir, layer);
        break;
    
    case IFT_GFLIM_TEST_DECODER:
        sprintf(output_dir, "%s/test_%d", model_dir, layer);
        break;
    
    case IFT_GFLIM_MEAN_DECODER:
        sprintf(output_dir, "%s/mean_%d", model_dir, layer);
        break;

    default:
        iftError("Invalid decoder type: %d", "main", decoder);
        break;
    }
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
        char ext[10], filename[512];
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
            char filename[512];
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
            saliency = iftGFLIMDecodeLayer(input->val[k], label_imgs->val[k], label_imgs0->val[k], truelabels_i, nlabels_i, decoder);
            iftFree(truelabels_i);
        }else{
            // decode the graph
            saliency = iftGFLIMDecodeLayer(input->val[k], label_imgs->val[k], label_imgs0->val[k], truelabels, nlabels, decoder);
        }

#pragma omp parallel for
        /* post-process and save the salience maps */
        for (int l = 0; l < saliency->n; l++)
        {
            char sal_dir[512], filename[512];
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
        iftError("Usage: iftGFLIMGSDecodeLayer <P1> <P2> <P3> <P4> <P5> <P6> \n"
                 "[1] model directory \n"
                 "[2] layer number (1, 2,..) \n"
                 "[3] architecture of the network\n"
                 "[4] decoder type (tristate, hybrid, graph-adaptive, gaussian)\n"
                 "[5] stride type (none, image-based, graph-based)\n"
                 "[6] batch size \n",
                 "main");
    }

    timer *tstart = iftTic();

    char *model_dir = argv[1];
    int layer = atoi(argv[2]);
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[3]);
    iftGFLIMDecoderType decoder = iftGLFLIMGetDecoderType(argv[4]);
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[5]);
    int batchsize = atoi(argv[6]);

    char input_dir[200], labels_dir[200], input_net_labels[200], output_dir[200];

    validateArgs(layer, model_dir, decoder, stride_type, input_dir, labels_dir, input_net_labels, batchsize, output_dir);

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", true);
    if (fs->n == 0)
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
    char filename[512];
    sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer_skip);
    if (decoder != IFT_GFLIM_TRISTATE_DECODER && iftFileExists(filename))
    {
        // read one graph to get the number of kernels
        iftGFLIMGraph *g = iftReadGFLIMGraph(fs->files[0]->path);
        nkernels = g->num_feats; // number of kernels is the same for all graphs in the batch
        iftDestroyGFLIMGraph(&g);

        /* load merged model */
        truelabels = iftGFLIMLoadKernelLabels(filename, &nkernels);
        nlabels = iftGFLIMNumberOfTruelabels(truelabels, nkernels);
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
