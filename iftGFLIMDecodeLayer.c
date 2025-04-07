#include "ift.h"

/**
 * @brief Decode a layer of a GFLIM model.
 */

void validateArgs(int layer, char *model_dir, iftGFLIMDecoderType decoder, iftGFLIMStrideType stride_type, 
                char *input_dir, char *labels_dir, char *input_net_labels, char *output_dir)
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

    switch (decoder)
    {
    case IFT_GFLIM_TRISTATE_DECODER:
        sprintf(output_dir, "%s/tristate_%d", model_dir, layer);
        break;
    
    case IFT_GFLIM_HYBRID_DECODER:
        sprintf(output_dir, "%s/hybrid_%d", model_dir, layer);
        break;

    case IFT_GFLIM_GRAPHADAPTIVE_DECODER:
        sprintf(output_dir, "%s/graphadaptive_%d", model_dir, layer);
        break;

    case IFT_GFLIM_GAUSSIAN_DECODER:
        sprintf(output_dir, "%s/gaussian_%d", model_dir, layer);
        break;

    default:
        iftError("Invalid decoder type: %d", "main", decoder);
        break;
    }
}


int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        iftError("Usage: iftGFLIMGSDecodeLayer <P1> <P2> <P3> <P4> <P5> \n"
                 "[1] model directory \n"
                 "[2] layer number (1, 2,..) \n"
                 "[3] architecture of the network\n"
                 "[4] decoder type (tristate, hybrid, graph-adaptive, gaussian)"
                 "[5] stride type (none, image-based, graph-based) \n",
                 "main");
    }

    timer *tstart = iftTic();

    char *model_dir = argv[1];
    int layer = atoi(argv[2]);
    iftFLIMArch *arch = iftReadFLIMArch(argv[3]);
    iftGFLIMDecoderType decoder = iftGLFLIMGetDecoderType(argv[4]);
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[5]);

    char input_dir[200], labels_dir[200], input_net_labels[200], output_dir[200];

    validateArgs(layer, model_dir, decoder, stride_type, input_dir, labels_dir, input_net_labels, output_dir);

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", true);
    if (fs->n == 0)
        iftError("No images found in the input directory: %s", "main", input_dir);

    if(!iftDirExists(output_dir))
        iftMakeDir(output_dir);

    char *filename = iftAllocCharArray(512);
    char sal_dir[512];

    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, ".json");

        // load the graph and superpixel image
        iftFLIMGraph *graph = iftReadFLIMGraph(fs->files[i]->path);
        sprintf(filename, "%s/%s.pgm", labels_dir, basename);
        iftImage *label_img = iftReadImageByExt(filename);

        // load the superpixel image of the input network
        sprintf(filename, "%s/%s.pgm", input_net_labels, basename);
        iftImage *input = iftReadImageByExt(filename);

        int nkernels = graph->num_feats;
        int *truelabels = NULL, nlabels = 0;
        sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer);
        if (iftFileExists(filename))
        { 
            /* load merged model */
            truelabels = iftGFLIMLoadKernelLabels(filename, &nkernels);
            nlabels = iftGFLIMNumberOfTruelabels(truelabels, nkernels);
        }
        else
        {
            sprintf(filename, "%s/model/%s-conv%d-labels.txt",
                    model_dir, basename, layer);
            if (iftFileExists(filename))
            { 
                /* load image model */
                truelabels = iftGFLIMLoadKernelLabels(filename, &nkernels);
                nlabels = iftGFLIMNumberOfTruelabels(truelabels, nkernels);
            }else{
                iftError("File %s does not exist", "main", filename);
            }
        }

        // decode the graph
        iftImageArray *saliency = iftGFLIMDecodeLayer(graph, label_img, input, truelabels, nlabels, decoder);

        iftDestroyImage(&input);
        iftDestroyFLIMGraph(&graph);
        iftFree(truelabels);

        /* post-process and save the salience maps */
        if(decoder == IFT_GFLIM_GRAPHADAPTIVE_DECODER) nlabels--;
        for (int l = 0; l < nlabels; l++)
        {
            sprintf(sal_dir, "%s/label%d", output_dir, l + 1);
            if (!iftDirExists(sal_dir))
                iftMakeDir(sal_dir);

            if (iftIs3DImage(label_img))
            {
                sprintf(filename, "%s/%s.nii.gz", sal_dir, basename);
            }
            else
            {
                sprintf(filename, "%s/%s.png", sal_dir, basename);
            }
            iftWriteImageByExt(saliency->val[l], filename);
        }
        iftDestroyImage(&label_img);
        iftFree(basename);
        iftDestroyImageArray(&saliency);
    }
    iftFree(filename);
    iftDestroyFileSet(&fs);
    iftDestroyFLIMArch(&arch);
    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("\nDone ... %s\n", time);
    iftFree(time);
    return (0);
}
