#include "ift.h"

/**
 * \file iftGFLIMExtract.c
 * \brief Extract features from a set of images and decode from a GFLIM or a GFLIM-BoFP trained model.
 */

void validateArgs(char *images_dir, char *model_dir, iftFLIMArch *arch, int nLayers, 
                  int batchsize, 
                  int argc, char *argv[])
{
    if (nLayers < 1)
        iftError("Invalid number of layers: %d", "main", nLayers);

    if (nLayers > arch->nlayers)
        iftError("Invalid number of layers: %d", "main", nLayers);

    if (!iftDirExists(images_dir))
        iftError("Images directory does not exist", "main");

    if (!iftDirExists(model_dir))
        iftError("Model directory does not exist", "main");
    
    if (batchsize < 1)
        iftError("Invalid batch size: %d", "main", batchsize);
}

void ProcessBatch(iftFileSet *fs_imgs, int first, int last, 
                char *graphs_dir, char *labels_dir, 
                char *model_dir, 
                int nLayers,
                iftFLIMGraphAttention attention_opt, iftGFLIMStrideType stride_type,
                iftGFLIMAggregationType aggregation_opt, bool kernel_removal,
                iftFLIMArch *arch)
{
    int ngraphs_batch = last - first + 1;
    iftImageArray *label_imgs = iftCreateImageArray(ngraphs_batch);
    
    /* Create graphs */
    iftFLIMGraphArray *input = iftCreateEmptyFLIMGraphArray(ngraphs_batch);

    for (int k = 0; k <= last - first; k++)
    {
        int i = k + first;
        char ext[10], filename[512];
        sprintf(ext, "%s", iftFileExt(fs_imgs->files[i]->path));
        char *basename = iftFilename(fs_imgs->files[i]->path, ext);
        sprintf(filename, "%s/%s.json", graphs_dir, basename);
        input->val[k] = iftReadFLIMGraph(filename);
        sprintf(filename, "%s/%s.pgm", labels_dir, basename);
        label_imgs->val[k] = iftReadImageByExt(filename); 
        iftFree(basename);
    }   

    /* For each layer do */
    for (int l = 0; l < nLayers; l++)
    {
        iftMatrix *Kmerged = NULL;
        float *bias_merged = NULL;
        int *true_labels = NULL;
        char filename[512];

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
        iftFree(bias_merged);
        iftFree(true_labels);

        char dir[512];
        sprintf(dir, "%s/layer%d", model_dir, l+1);
        if(!iftDirExists(dir))
            iftMakeDir(dir);
        sprintf(dir, "%s/labels%d", model_dir, l+1);
        if(!iftDirExists(dir))
            iftMakeDir(dir);

        // write graphs and labels
        for (int k = 0; k < ngraphs_batch; k++)
        {
            char ext[10];
            char filename[512];
            sprintf(ext, "%s", iftFileExt(fs_imgs->files[k + first]->path));
            char *basename = iftFilename(fs_imgs->files[k + first]->path, ext);

            sprintf(filename, "%s/layer%d/%s.json", model_dir, l+1, basename);
            iftWriteFLIMGraph(input->val[k], filename);

            sprintf(filename, "%s/labels%d/%s.pgm", model_dir, l+1, basename);
            iftWriteImageByExt(label_imgs->val[k], filename);
            iftFree(basename);
        }
    }
    /*
    char dir[512];
    sprintf(dir, "%s/layer%d", model_dir, nLayers);
    if(!iftDirExists(dir))
        iftMakeDir(dir);
    sprintf(dir, "%s/labels%d", model_dir, nLayers);
    if(!iftDirExists(dir))
        iftMakeDir(dir);
    
    // write graphs and labels
    for (int k = 0; k < ngraphs_batch; k++)
    {
        char ext[10];
        char filename[512];
        sprintf(ext, "%s", iftFileExt(fs_imgs->files[k + first]->path));
        char *basename = iftFilename(fs_imgs->files[k + first]->path, ext);

        sprintf(filename, "%s/layer%d/%s.json", model_dir, nLayers, basename);
        iftWriteFLIMGraph(input->val[k], filename);

        sprintf(filename, "%s/labels%d/%s.pgm", model_dir, nLayers, basename);
        iftWriteImageByExt(label_imgs->val[k], filename);
        iftFree(basename);
    }*/

    iftDestroyImageArray(&label_imgs);
    iftDestroyFLIMGraphArray(&input);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 12)
    {
        iftError("Usage: iftGFLIMExtract <P1> <P2> <P3> <P4> <P5> <P6> <P7> <P8> <P9> <P10> <P11> \n"
                 "[ 1] Images directory \n"
                 "[ 2] Graphs directory \n"
                 "[ 3] Labels directory \n"
                 "[ 4] Model directory \n"
                 "[ 5] Architecture file \n"
                 "[ 6] Number of layers \n"
                 "[ 7] Batch size (> 0) \n"
                 "[ 8] Stride type (none, image-based, graph-based) \n"
                 "[ 9] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE) \n"
                 "[10] aggregation (CONCAT, MEAN, WMEAN) \n"
                 "[11] reduce kernels (1/0) \n"
                 "argc: %d",
                 "main", argc);
    }
    tstart = iftTic();

    char *images_dir = argv[1];
    char *graphs_dir = argv[2];
    char *labels_dir = argv[3];
    char *model_dir = argv[4];
    iftFLIMArch *arch = iftReadFLIMArch(argv[5]);
    int nLayers = atoi(argv[6]);
    int batchsize = atoi(argv[7]);
    iftGFLIMStrideType stride_type = iftGFLIMGetStrideType(argv[8]);
    iftFLIMGraphAttention attention_opt = iftGLIMGraphGetAttentionType(argv[9]);
    iftGFLIMAggregationType aggregation_opt = iftGLIMGraphGetAggregationType(argv[10]);
    bool kernel_removal = atoi(argv[11]);

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

    for (int batch = 0; batch < nbatches; batch++)
    {
        int first = batch * batchsize, last = first + batchsize - 1;
        printf("Processing batch %d of %d batches\r", batch + 1, nbatches);
        fflush(stdout);

        ProcessBatch(fs_imgs, first, last, graphs_dir, labels_dir, model_dir, 
                    nLayers,
                    attention_opt, stride_type,
                    aggregation_opt, kernel_removal,
                    arch);
    }

    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last)
    {
        printf("Processing remaining %d images\r", last - first + 1);
        fflush(stdout);

        ProcessBatch(fs_imgs, first, last, 
            graphs_dir, labels_dir, model_dir, 
            nLayers,
            attention_opt, stride_type,
            aggregation_opt, kernel_removal,
            arch);
    }

    iftDestroyFileSet(&fs_imgs);
    iftDestroyFLIMArch(&arch);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return 0;
}
