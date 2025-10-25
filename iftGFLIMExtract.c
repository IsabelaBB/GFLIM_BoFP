#include "ift.h"

/**
 * \file iftGFLIMExtract.c
 * \brief Extract features from a set of images and decode from a GFLIM or a GFLIM-BoFP trained model.
 * \warning This does not support skip connections.
 */

void validateArgs(char *images_dir, char *model_dir, iftGFLIMArch *arch, int nLayers, 
                  iftGFLIMGraphSpxMethod graph_type, int nsuperpixels, int batchsize, 
                  int argc, char *argv[], int *initial_seeds)
{
    if (nLayers < 1)
        iftError("Invalid number of layers: %d", "main", nLayers);

    if (nLayers > arch->nlayers)
        iftError("Invalid number of layers: %d", "main", nLayers);

    if (!iftDirExists(images_dir))
        iftError("Images directory does not exist", "main");

    if (!iftDirExists(model_dir))
        iftError("Model directory does not exist", "main");

    if (nsuperpixels < 1)
        iftError("Invalid number of superpixels: %d", "main", nsuperpixels);
    
    if (batchsize < 1)
        iftError("Invalid batch size: %d", "main", batchsize);

    if (graph_type == IFT_FLIMGRAPH_TYPE_DISF)
    {
        if (argc < 15)
            iftError("Initial seeds is required when superpixel method is DISF", "main");
        (*initial_seeds) = atoi(argv[14]);
        if ((*initial_seeds) < 1)
            iftError("Invalid number of initial seeds: %d", "main", (*initial_seeds));
        if (nsuperpixels > (*initial_seeds))
            iftError("Number of superpixels (%d) must be smaller than the number of initial seeds (%d)", "main", nsuperpixels, (*initial_seeds));
    }
}

void ProcessBatch(iftFileSet *fs_imgs, int first, int last, 
                char *images_dir, char *model_dir, char *saliency_dir,
                int initial_seeds, int nsuperpixels, int nLayers,
                iftGFLIMGraphNodeFeature graph_feat, iftGFLIMGraphSpxMethod graph_type,
                iftGFLIMGraphAttention attention_opt, iftGFLIMStrideType stride_type,
                iftGFLIMAggregation aggregation_opt, bool kernel_removal,
                iftGFLIMDecoderType decoder, iftGFLIMArch *arch)
{
    iftGFLIMGraphArray *input = NULL;
    int ngraphs_batch = last - first + 1;
    iftImageArray *label_imgs = iftCreateImageArray(ngraphs_batch);
    iftImageArray *label_imgs_0 = iftCreateImageArray(ngraphs_batch);
    char filename[1024];

    /* Create graphs */
    input = iftGFLIMImageToFLIMGraphBatch_v2(fs_imgs, first, last, initial_seeds, nsuperpixels, graph_feat, graph_type, label_imgs);

    for (int i = 0; i < ngraphs_batch; i++)
        label_imgs_0->val[i] = iftCopyImage(label_imgs->val[i]);

    /* For each layer do */
    for (int l = 0; l < nLayers; l++)
    {
        iftMatrix *Kmerged = NULL;
        float *bias_merged = NULL;
        int *true_labels = NULL;

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
                iftError("Model file not found: %s", "encodeLayerBatch", filename);
            Kmerged = iftReadMatrix(filename);
            int ninput_channels = input->val[0]->num_feats;

            /* Load mean and standard deviation */
            float *mean = iftAllocFloatArray(ninput_channels);
            float *stdev = iftAllocFloatArray(ninput_channels);
            sprintf(filename, "%s/model/conv%d", model_dir, l + 1);
            iftReadMeanStdev(filename, mean, stdev, ninput_channels);

            /* Normalization (because there is no bias)*/
#pragma omp parallel for
            for (int i = 0; i < input->n; i++)
                iftGFLIMGraphZNormalization(input->val[i], mean, stdev);
            iftFree(mean);
            iftFree(stdev);
        }

        /* Load labels indicating class (label) per channel */
        if (attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE && bias_merged != NULL)
        {
            sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, l + 1);
            if (!iftFileExists(filename))
                iftError("File %s does not exist", "main", filename);
            int nkernels = 0;
            true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);
        }

        iftGFLIMEncodeLayerBatch_v2(input, Kmerged, arch->layer[l], label_imgs, bias_merged, stride_type, attention_opt, true_labels, aggregation_opt, kernel_removal);

        iftDestroyMatrix(&Kmerged);
        iftFree(bias_merged);
        iftFree(true_labels);
    }

    /* Load labels indicating class (object) per channel */
    int *truelabels = NULL, nlabels = 0, nkernels;
    sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, nLayers);
    if (iftFileExists(filename))
    {
        truelabels = iftGFLIMLoadKernelLabels(filename, &nkernels);
        nlabels = iftGFLIMNumberOfTruelabels(truelabels, nkernels);
    }
    else
        nlabels = 1;

    // decode
    for (int k = 0; k < ngraphs_batch; k++)
    {
        char ext[10];
        sprintf(ext, "%s", iftFileExt(fs_imgs->files[k + first]->path));
        char *basename = iftFilename(fs_imgs->files[k + first]->path, ext);

        iftImageArray *saliency = iftGFLIMDecodeLayer(input->val[k], label_imgs->val[k], label_imgs_0->val[k], truelabels, nlabels, decoder);

        char tmp[1024], out_file[1024];
        /* post-process and save the salience maps */
        if (nlabels == 1)
        {
            sprintf(tmp, "%s", saliency_dir);
            if (!iftDirExists(tmp))
                iftMakeDir(tmp);

            if (iftIs3DImage(label_imgs->val[k]))
                sprintf(out_file, "%s/%s.nii.gz", tmp, basename);
            else
                sprintf(out_file, "%s/%s.png", tmp, basename);
            iftWriteImageByExt(saliency->val[0], out_file);
        }
        else
        {
            for (int l = 0; l < nlabels; l++)
            {
                sprintf(tmp, "%s/label%d", saliency_dir, l + 1);
                if (!iftDirExists(tmp))
                    iftMakeDir(tmp);

                if (iftIs3DImage(label_imgs->val[k]))
                    sprintf(out_file, "%s/%s.nii.gz", tmp, basename);
                else
                    sprintf(out_file, "%s/%s.png", tmp, basename);
                iftWriteImageByExt(saliency->val[l], out_file);
            }
        }
        iftDestroyImageArray(&saliency);
        iftFree(basename);
    }

    iftDestroyImageArray(&label_imgs_0);
    iftDestroyImageArray(&label_imgs);
    iftDestroyGFLIMGraphArray(&input);
    iftFree(truelabels);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 15 && argc != 14)
    {
        iftError("Usage: iftGFLIMExtract <P1> <P2> <P3> <P4> <P5> <P6> <P7> <P8> <P9> <P10> <P11> <P12> <P13> [<P14>] \n"
                 "[ 1] Images directory \n"
                 "[ 2] Model directory \n"
                 "[ 3] Architecture file \n"
                 "[ 4] Number of layers \n"
                 "[ 5] Graph feature {MEAN, SUM, SUM_MULT_RELSIZE, SUM_CONCAT_RELSIZE} (see iftGFLIMGraph.h for more information) \n"
                 "[ 6] superpixel method {1:DISF, 2:ERGC} \n"
                 "[ 7] Number of superpixels \n"
                 "[ 8] decoder type (tristate, hybrid, graph-adaptive, gaussian) \n"
                 "[ 9] Batch size (> 0) \n"
                 "[10] Stride type (none, image-based, graph-based) \n"
                 "[11] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE) \n"
                 "[12] aggregation (CONCAT, MEAN, WMEAN) \n"
                 "[13] reduce kernels (1/0) \n"
                 "[14] Initial seeds (only used when superpixel method is DISF) \n",
                 "argc: %d",
                 "main", argc);
    }
    tstart = iftTic();

    char *images_dir = argv[1];
    char *model_dir = argv[2];
    iftGFLIMArch *arch = iftReadGFLIMArch(argv[3]);
    int nLayers = atoi(argv[4]);
    iftGFLIMGraphNodeFeature graph_feat = iftGFLIMGraphGetNodeFeature(argv[5]);
    iftGFLIMGraphSpxMethod graph_type = iftGFLIMGraphGetSpxMethod(argv[6]);
    int nsuperpixels = atoi(argv[7]);
    iftGFLIMDecoderType decoder = iftGLFLIMGetDecoderType(argv[8]);
    int batchsize = atoi(argv[9]);
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[10]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttention(argv[11]);
    iftGFLIMAggregation aggregation_opt = iftGLIMGraphGetAggregation(argv[12]);
    bool kernel_removal = atoi(argv[13]);

    int initial_seeds = 0;
    validateArgs(images_dir, model_dir, arch, nLayers, graph_type, nsuperpixels, batchsize, argc, argv, &initial_seeds);

    iftFileSet *fs_imgs = iftLoadFileSetFromDir(images_dir, 1);
    int ngraphs = fs_imgs->n;
    int nbatches = 0;
    int nremaining_graphs = fs_imgs->n;
    char saliency_dir[1024];

    if (fs_imgs->n < 1)
        iftError("Image list is empty", __func__);

    batchsize = iftMin(batchsize, ngraphs);
    nbatches = ngraphs / batchsize;
    nremaining_graphs = nremaining_graphs - nbatches * batchsize;

    switch (decoder)
    {
    case IFT_GFLIM_TRISTATE_DECODER:
        sprintf(saliency_dir, "%s/tristate_%d", model_dir, nLayers);
        break;

    case IFT_GFLIM_HYBRID_DECODER:
        sprintf(saliency_dir, "%s/hybrid_%d", model_dir, nLayers);
        break;

    case IFT_GFLIM_GRAPHADAPTIVE_DECODER:
        sprintf(saliency_dir, "%s/graph-adaptive_%d", model_dir, nLayers);
        break;

    case IFT_GFLIM_GAUSSIAN_DECODER:
        sprintf(saliency_dir, "%s/gaussian_%d", model_dir, nLayers);
        break;
    
    case IFT_GFLIM_TEST_DECODER:
        sprintf(saliency_dir, "%s/test_%d", model_dir, nLayers);
        break;
        
    case IFT_GFLIM_MEAN_DECODER:
        sprintf(saliency_dir, "%s/mean_%d", model_dir, nLayers);
        break;

    default:
        iftError("Invalid decoder type: %d", "main", decoder);
        break;
    }
    iftMakeDir(saliency_dir);

    for (int batch = 0; batch < nbatches; batch++)
    {
        int first = batch * batchsize, last = first + batchsize - 1;
        printf("Processing batch %d of %d batches\r", batch + 1, nbatches);
        fflush(stdout);

        ProcessBatch(fs_imgs, first, last, images_dir, model_dir, saliency_dir,
                    initial_seeds, nsuperpixels, nLayers,
                    graph_feat, graph_type,
                    attention_opt, stride_type,
                    aggregation_opt, kernel_removal,
                    decoder, arch);
    }

    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last)
    {
        printf("Processing remaining %d images\r", last - first + 1);
        fflush(stdout);

        ProcessBatch(fs_imgs, first, last, images_dir, model_dir, saliency_dir,
            initial_seeds, nsuperpixels, nLayers,
            graph_feat, graph_type,
            attention_opt, stride_type,
            aggregation_opt, kernel_removal,
            decoder, arch);
    }

    iftDestroyFileSet(&fs_imgs);
    iftDestroyGFLIMArch(&arch);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return 0;
}
