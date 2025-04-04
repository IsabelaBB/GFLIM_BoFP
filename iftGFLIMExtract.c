#include "ift.h"

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 15 && argc != 14){
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
    iftFLIMArch *arch = iftReadFLIMArch(argv[3]);
    int nLayers = atoi(argv[4]);
    iftFLIMGraphFeature graph_feat = iftGFLIMGetGraphFeature(argv[5]);
    iftFLIMGraphType graph_type = iftGFLIMGetGraphType(argv[6]);
    int nsuperpixels = atoi(argv[7]);
    iftGFLIMDecoderType decoder = iftGLFLIMGetDecoderType(argv[8]);
    int batchsize = atoi(argv[9]);
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[10]);
    iftFLIMGraphAttention attention_opt = iftGLIMGraphGetAttentionType(argv[11]);
    iftGFLIMAggregationType aggregation_opt = iftGLIMGraphGetAggregationType(argv[12]);
    bool kernel_removal = atoi(argv[13]);

    int initial_seeds = 0;

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
        
    if (graph_type == IFT_FLIMGRAPH_TYPE_DISF)
    {
        if (argc < 15)
            iftError("Initial seeds is required when superpixel method is DISF", "main");
        initial_seeds = atoi(argv[14]);
        if (initial_seeds < 1)
            iftError("Invalid number of initial seeds: %d", "main", initial_seeds);
        if (nsuperpixels > initial_seeds)
            iftError("Number of superpixels (%d) must be smaller than the number of initial seeds (%d)", "main", nsuperpixels, initial_seeds);
    }

    iftFileSet *fs_imgs = iftLoadFileSetFromDir(images_dir, 1);
    int ngraphs = fs_imgs->n;
    int nbatches = 0;
    int nremaining_graphs = fs_imgs->n;
    int ninput_channels = 3;
    char saliency_dir[200];

    if (fs_imgs->n < 1) iftError("Image list is empty", __func__);

    if (batchsize < 1) iftError("Invalid batch size: %d", "main", batchsize);

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
        sprintf(saliency_dir, "%s/graphadaptive_%d", model_dir, nLayers);
        break;

    case IFT_GFLIM_GAUSSIAN_DECODER:
        sprintf(saliency_dir, "%s/gaussian_%d", model_dir, nLayers);
        break;

    default:
        iftError("Invalid decoder type: %d", "main", decoder);
        break;
    }
    iftMakeDir(saliency_dir);

    for (int batch = 0; batch < nbatches; batch++)
    {
        int first = batch * batchsize, last = first + batchsize - 1;
        printf("Processing batch %d of %d batches\n", batch + 1, nbatches);
        fflush(stdout);

        iftFLIMGraphArray *input = NULL;
        int ngraphs_batch = last - first + 1;
        iftImageArray *label_imgs = iftCreateImageArray(ngraphs_batch);
        iftImageArray *label_imgs_0 = iftCreateImageArray(ngraphs_batch);
        char filename[200];

        /* Create graphs */
        input = iftGFLIMImageToFLIMGraphBatch_v2(fs_imgs, first, last, initial_seeds, nsuperpixels, graph_feat, graph_type, label_imgs);
        
        for(int i = 0; i < ngraphs_batch; i++)
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
                sprintf(filename, "%s/model/conv%d.txt", model_dir, l + 1);
                iftGFLIMLoadMergedModel(filename, &Kmerged, &bias_merged);
            }
            else
            {
                /* Load kernel */
                sprintf(filename, "%s/model/conv%d-kernels.npy", model_dir, l + 1);
                if (!iftFileExists(filename))
                    iftError("Model file not found: %s", "encodeLayerBatch", filename);
                Kmerged = iftReadMatrix(filename);

                /* Load mean and standard deviation */
                float *mean = iftAllocFloatArray(ninput_channels);
                float *stdev = iftAllocFloatArray(ninput_channels);
                sprintf(filename, "%s/model/conv%d", model_dir, l + 1);
                iftReadMeanStdev(filename, mean, stdev, ninput_channels);

                /* Normalization (because there is no bias)*/
                for (int i = 0; i < input->n; i++)
                    iftNormalizeFLIMGraphByZScore(input->val[i], mean, stdev);
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
            if (bias_merged != NULL)
                iftFree(bias_merged);
            if (true_labels != NULL)
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

        // decode
#pragma omp parallel for
        for (int k = 0; k < ngraphs_batch; k++)
        {
            char ext[10];
            sprintf(ext, "%s", iftFileExt(fs_imgs->files[k + first]->path));
            char *basename = iftFilename(fs_imgs->files[k + first]->path, ext);
            
            iftImageArray *saliency = iftGFLIMDecodeLayer(input->val[k], label_imgs->val[k], label_imgs_0->val[k], truelabels, nlabels, decoder);

            char tmp[400], out_file[512];
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
            iftDestroyFLIMGraph(&input->val[k]);
            iftFree(basename);
        }
        iftDestroyImageArray(&label_imgs_0);
        iftDestroyImageArray(&label_imgs);
        iftFree(input);
        iftFree(truelabels);
    }

    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last)
    {
        printf("Processing remaining %d images\n", last - first + 1);
        fflush(stdout);

        iftFLIMGraphArray *input = NULL;
        int ngraphs_batch = last - first + 1;
        iftImageArray *label_imgs = iftCreateImageArray(ngraphs_batch);
        iftImageArray *label_imgs_0 = iftCreateImageArray(ngraphs_batch);
        char filename[200];

        /* Create graphs */
        input = iftGFLIMImageToFLIMGraphBatch_v2(fs_imgs, first, last, initial_seeds, nsuperpixels, graph_feat, graph_type, label_imgs);

        for(int i = 0; i < ngraphs_batch; i++)
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
                sprintf(filename, "%s/model/conv%d.txt", model_dir, l + 1);
                iftGFLIMLoadMergedModel(filename, &Kmerged, &bias_merged);
            }
            else
            {
                /* Load kernel */
                sprintf(filename, "%s/model/conv%d-kernels.npy", model_dir, l + 1);
                if (!iftFileExists(filename))
                    iftError("Model file not found: %s", "encodeLayerBatch", filename);
                Kmerged = iftReadMatrix(filename);

                /* Load mean and standard deviation */
                float *mean = iftAllocFloatArray(ninput_channels);
                float *stdev = iftAllocFloatArray(ninput_channels);
                sprintf(filename, "%s/model/conv%d", model_dir, l + 1);
                iftReadMeanStdev(filename, mean, stdev, ninput_channels);

                /* Normalization (because there is no bias)*/
                for (int i = 0; i < input->n; i++)
                    iftNormalizeFLIMGraphByZScore(input->val[i], mean, stdev);
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
            if (bias_merged != NULL)
                iftFree(bias_merged);
            if (true_labels != NULL)
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

        // decode
#pragma omp parallel for
        for (int k = 0; k < ngraphs_batch; k++)
        {
            char ext[10];
            sprintf(ext, "%s", iftFileExt(fs_imgs->files[k + first]->path));
            char *basename = iftFilename(fs_imgs->files[k + first]->path, ext);
            
            iftImageArray *saliency = iftGFLIMDecodeLayer(input->val[k], label_imgs->val[k], label_imgs_0->val[k], truelabels, nlabels, decoder);
            
            char tmp[400], out_file[512];
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
            iftDestroyFLIMGraph(&input->val[k]);
            iftFree(basename);
        }
        iftDestroyImageArray(&label_imgs_0);
        iftDestroyImageArray(&label_imgs);
        iftFree(input);
        iftFree(truelabels);
    }

    iftDestroyFileSet(&fs_imgs);
    iftDestroyFLIMArch(&arch);

    printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
    return 0;
}
