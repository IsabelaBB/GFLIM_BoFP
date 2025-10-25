#include "ift.h"

/**
 * \file iftGFLIMMergeLayerModels.c
 * \brief Merge the kernels, bias, and labels of a layer created by iftGFLIMCreateLayerModel.
 */

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 3)
        iftError("Usage: iftGFLIMMergeLayerModels <P1> <P2> \n"
                 "[1] model directory \n"
                 "[2] layer number (1, 2, 3) \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    int layer = atoi(argv[2]);
    
    char *filename = iftAllocCharArray(1024);
    char input_dir[1024];

    sprintf(input_dir, "%s/layer%d", model_dir, layer-1);
    if(!iftDirExists(input_dir))
        iftError("Graph directory does not exist", "main");

    if(layer < 1)
        iftError("Invalid layer number: %d", "main", layer);

    if(!iftDirExists(model_dir))
        iftError("Model directory does not exist: %s", "main", model_dir);

    sprintf(filename, "%s/model", model_dir);   
    if(!iftDirExists(filename))
        iftError("Model directory does not exist: %s", "main", filename);
    
    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", true);

    iftMatrix *Kmerged = NULL;
    float *bias_merged = NULL;
    int *truelabels_merged = NULL;

    /* Compute the total number of kernels and the feature vector size */
    int ncols = 0, nrows = 0;
    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, ".json");
        sprintf(filename, "%s/model/%s-conv%d-kernels.npy", model_dir, basename, layer);
        if (iftFileExists(filename))
        {   /* encode layer using its kernels */
            iftMatrix *K = iftReadMatrix(filename);
            ncols += K->ncols;
            if (nrows == 0) nrows = K->nrows;
            else
            {
                if (nrows != K->nrows)
                    iftWarning("Cannot merge kernels with different labels vector sizes. Adopting one kernel per marked image.", "main");
            }
        }
        iftFree(basename);
    }

    /* Merge data */

    Kmerged = iftCreateMatrix(ncols, nrows);
    bias_merged = iftAllocFloatArray(ncols);
    truelabels_merged = iftAllocIntArray(ncols);

    int col = 0;
    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, ".json");
        sprintf(filename, "%s/model/%s-conv%d-kernels.npy", model_dir, basename, layer);
        if (iftFileExists(filename))
        {   
            /* read kernels, bias and kernel labels for the current graph */
            iftMatrix *K = iftReadMatrix(filename);
            sprintf(filename, "%s/model/%s-conv%d", model_dir, basename, layer);
            float *bias = iftGFLIMLoadBias(filename);
            if(bias == NULL) iftError("Bias is NULL", "main");
            sprintf(filename, "%s/model/%s-conv%d-labels.txt",
                    model_dir, basename, layer);
            int nkernels = 0;
            int *truelabels = iftGFLIMLoadKernelLabels(filename, &nkernels);

            // merge them
            for (int c = 0; c < K->ncols; c++, col++)
            {   bias_merged[col] = bias[c];
                truelabels_merged[col] = truelabels[c];
                for (int r = 0; r < K->nrows; r++)
                    iftMatrixElem(Kmerged, col, r) = iftMatrixElem(K, c, r);
            }

            iftDestroyMatrix(&K);
            iftFree(bias);
            iftFree(truelabels);
        }
        iftFree(basename);
    }

    /* Save merged kernels, bias and kernel labels */
    sprintf(filename, "%s/model/conv%d", model_dir, layer);
    iftGFLIMSaveMergedModel(filename, Kmerged, bias_merged, truelabels_merged);

    iftDestroyMatrix(&Kmerged);
    iftFree(bias_merged);
    iftFree(truelabels_merged);
    iftFree(filename);
    iftDestroyFileSet(&fs);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
