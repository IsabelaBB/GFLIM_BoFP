#include "ift.h"

/**
 * \file iftGFLIMWriteActivations.c
 * \brief Write the activations of a GFLIM or a GFLIM-BoFP layer
 */

/*iftDataSet **ReadPCAMatrixAndClasses(char *model_dir, int *nclasses)
{
    int nclasses_dataset = 0;
    char filename[512];

    sprintf(filename, "%s/model/PCAMatrix_%d.zip", model_dir, nclasses_dataset);
    while(iftFileExists(filename))
    {
        nclasses_dataset++;
        sprintf(filename, "%s/model/PCAMatrix_%d.zip", model_dir, nclasses_dataset);
    }

    if(nclasses_dataset == 0)
        iftError("No PCA matrix found in %s", "main", model_dir);

    iftDataSet **pcaMatrix = (iftDataSet **)iftAlloc(nclasses_dataset, sizeof(iftDataSet *));
    for(int c = 0; c < nclasses_dataset; c++)
    {
        sprintf(filename, "%s/model/PCAMatrix_%d.zip", model_dir, c);
        pcaMatrix[c] = iftCreateDataSet(0, 1);
        pcaMatrix[c]->fsp = iftReadPCAMatrix(filename);
        pcaMatrix[c]->nfeats = pcaMatrix[c]->fsp.nfeats;
    }
    (*nclasses) = nclasses_dataset;
    return pcaMatrix;
}
*/

/*float *iftGFLIMLoadBiasPCA(char *basepath, int *nkernels)
{
#ifdef IFT_DEBUG
  printf("%s \n", __func__);
  printf("\t basepath: %s\n", basepath == NULL ? "NULL" : basepath);
#endif

    char filename[512];
    FILE *fp;
    float *bias;

    sprintf(filename, "%s-bias.txt", basepath);
    fp = fopen(filename, "r");
    if(fp == NULL) iftError("Error opening file %s", filename);
    if(fscanf(fp, "%d", nkernels) != 1)
      iftError("Error reading number of kernels from file %s", filename);
    bias = iftAllocFloatArray(*nkernels);
    for (int k = 0; k < *nkernels; k++)
    {
      if(fscanf(fp, "%f ", &bias[k]) != 1)
        iftError("Error reading bias values from file %s", filename);
    }
    fclose(fp);
    return bias;
}
*/

void validateArgs(char *model_dir, int layer, char *input_dir, char *labels_dir, char *input_net_labels)
{
    if (layer < 1)
        iftError("Layer number must be greater than 0", "main");

    if (!iftDirExists(model_dir))
        iftError("Model directory does not exist: %s", "main", model_dir);

    if (!iftDirExists(input_dir))
        iftError("Graph directory does not exist: %s", "main", input_dir);

    if (!iftDirExists(input_net_labels))
        iftError("Labels directory input network does not exist: %s", "main", input_net_labels);

    if (!iftDirExists(labels_dir))
    {
        sprintf(labels_dir, "%s/labels0", model_dir);
        if (!iftDirExists(labels_dir))
            iftError("Labels directory does not exist: %s", "main", labels_dir);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        iftError("Usage: iftGFLIMWriteActivations <P1> <P2> \n"
                 "[1] model directory \n"
                 "[2] layer number (1, 2,..) \n"
                 //"[3] --pca \n"
                 ,
                 "main");
    }

    timer *tstart = iftTic();

    char *model_dir = argv[1];
    int layer = atoi(argv[2]);
    // bool pca = false;
    /*if (argc == 4)
    {
      if (strcmp(argv[3], "--pca") == 0)
        pca = true;
      else
        iftError("Invalid argument: %s", "main", argv[3]);
    }*/

    char input_dir[200], labels_dir[200], input_net_labels[200], output_dir[200];

    sprintf(input_dir, "%s/layer%d", model_dir, layer);
    sprintf(input_net_labels, "%s/labels0", model_dir);
    sprintf(labels_dir, "%s/labels%d", model_dir, layer);

    validateArgs(model_dir, layer, input_dir, labels_dir, input_net_labels);

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", true);
    if (fs->n == 0)
        iftError("No images found in the input directory: %s", "main", input_dir);

    char act_dir[400];
    char filename[512];

    bool merged_model = false;
    int *truelabels = NULL, nlabels = 0;
    int nkernels = 0 /*, nclasses = 0*/;

    // if (!pca){
    sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer);
    if (iftFileExists(filename)) // merged layer in GFLIM-BoFP
    {                            /* load merged model */
        truelabels = iftGFLIMLoadKernelLabels(filename, &nkernels);
        nlabels = iftGFLIMNumberOfTruelabels(truelabels, nkernels);
        sprintf(output_dir, "%s/activation%d", model_dir, layer);
        merged_model = true;
    }
    else
    {
        char *basename = iftFilename(fs->files[0]->path, ".json");
        sprintf(filename, "%s/model/%s-conv%d-bias.txt", model_dir, basename, layer);
        free(basename);
        if (iftFileExists(filename))                                         // is there a bias?
            sprintf(output_dir, "%s/activationNoMerge%d", model_dir, layer); // non merged layer in GFLIM-BoFP
        else                                                                 // GFLIM
        {
            merged_model = true;
            sprintf(output_dir, "%s/activation%d", model_dir, layer);
        }
    }

    printf("Merged: %d \n", merged_model);

    iftMakeDir(output_dir);
    /*} else { // not using it

      iftDataSet **PCAMatrix = ReadPCAMatrixAndClasses(model_dir, &nlabels);
      nclasses = nlabels;

      sprintf(filename, "%s/model/conv%d-1-kernels.npy", model_dir, layer);
      if (iftFileExists(filename))
      {
        merged_model = true;
        for (int c = 0; c < nlabels; c++)
          nkernels += PCAMatrix[c]->fsp.nfeats;

        truelabels = (int *)iftAlloc(nkernels, sizeof(int));
        int k = 0;
        for (int c = 0; c < nlabels; c++)
        {
          for (int j = 0; j < PCAMatrix[c]->fsp.nfeats; j++, k++)
            truelabels[k] = c + 1;
          iftDestroyDataSet(&PCAMatrix[c]);
        }
        iftFree(PCAMatrix);
        sprintf(output_dir, "%s/activation%d", model_dir, layer);
      }
      else
        sprintf(output_dir, "%s/activationNoMerge%d", model_dir, layer);
      iftMakeDir(output_dir);
    }*/

    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, ".json");
        printf("Writing activation for image %s \n", basename);

        // read the graph and the superpixels image
        iftGFLIMGraph *graph = iftReadFLIMGraph(fs->files[i]->path);
        sprintf(filename, "%s/%s.pgm", labels_dir, basename);
        iftImage *label_img = iftReadImageByExt(filename);
        int nkernels = graph->num_feats;

        /* Compute the scale factor w.r.t. the input of the network */
        float scale[3];
        sprintf(filename, "%s/%s.pgm", input_net_labels, basename);
        iftImage *input = iftReadImageByExt(filename);
        scale[0] = (float)input->xsize / (float)label_img->xsize;
        scale[1] = (float)input->ysize / (float)label_img->ysize;
        scale[2] = (float)input->zsize / (float)label_img->zsize;
        iftDestroyImage(&input);

        /* Load labels indicating class (object) per channel */
        if (!merged_model)
        {
            // if(!pca) {
            sprintf(filename, "%s/model/%s-conv%d-labels.txt",
                    model_dir, basename, layer);
            if (iftFileExists(filename))
            { /* load image model */
                truelabels = iftGFLIMLoadKernelLabels(filename, &nkernels);
                nlabels = iftGFLIMNumberOfTruelabels(truelabels, nkernels);
            }
            /*}else{ // not using it
              truelabels = iftAllocIntArray(nkernels);
              int k = 0;
              int nlabels = 1;

              sprintf(filename, "%s/model/%s-conv%d-%d", model_dir, basename, layer, nlabels);
              while (nlabels <= nclasses && iftFileExists(filename))
              {
                int nkernelsClass = 0;
                float *bias = iftGFLIMLoadBiasPCA(filename, &nkernelsClass);
                for (int j = 0; j < nkernelsClass; j++, k++)
                  truelabels[k] = nlabels;
                iftFree(bias);
                nlabels++;
                sprintf(filename, "%s/model/%s-conv%d-%d", model_dir, basename, layer, nlabels);
              }
              nlabels--;
            }*/
        }

        // convert from graph to activations (one activation per graph channel)
        iftImage **activations = iftGFLIMGraphToActivations(graph, label_img);
        iftDestroyImage(&label_img);

        if (nlabels != 0) // GFLIM-BoFP
        {
            for (int l = 0; l < nlabels; l++)
            {
                sprintf(act_dir, "%s/label%d/%s", output_dir, l + 1, basename);
                iftMakeDir(act_dir);
                for (int k = 0; k < graph->num_feats; k++)
                {
                    if ((l + 1) == truelabels[k])
                    {
                        iftImage *temp = iftInterp2D(activations[k], scale[0], scale[1]);
                        sprintf(filename, "%s/kernel-%03d.%s", act_dir, k + 1, "pgm");
                        iftWriteImageByExt(temp, filename);
                        iftDestroyImage(&temp);
                    }
                }
            }
            for (int k = 0; k < graph->num_feats; k++)
                iftDestroyImage(&activations[k]);
        }
        else // GFLIM
        {
            sprintf(act_dir, "%s/label%d/%s", output_dir, 1, basename);
            iftMakeDir(act_dir);
            for (int k = 0; k < graph->num_feats; k++)
            {
                iftImage *temp = iftInterp2D(activations[k], scale[0], scale[1]);
                sprintf(filename, "%s/kernel-%03d.%s", act_dir, k + 1, "pgm");
                iftWriteImageByExt(temp, filename);
                iftDestroyImage(&temp);
                iftDestroyImage(&activations[k]);
            }
        }

        iftDestroyFLIMGraph(&graph);
        if (!merged_model)
            iftFree(truelabels);
        iftFree(activations);
        iftFree(basename);
    }
    iftDestroyFileSet(&fs);
    if (merged_model)
        iftFree(truelabels);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
