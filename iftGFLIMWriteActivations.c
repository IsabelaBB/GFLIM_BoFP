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

void ProcessBatch(iftFileSet *fs, int first, int last, 
                char *model_dir, int layer, int skip,
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
        iftImage **activations = iftGFLIMGraphToActivations(input->val[k], label_imgs->val[k]);
        int nkernels_i = 0, *truelabels_i = NULL;

        /* Compute the scale factor w.r.t. the input of the network */
        float scale[3];
        scale[0] = (float)label_imgs0->val[k]->xsize / (float)label_imgs->val[k]->xsize;
        scale[1] = (float)label_imgs0->val[k]->ysize / (float)label_imgs->val[k]->ysize;
        scale[2] = (float)label_imgs0->val[k]->zsize / (float)label_imgs->val[k]->zsize;

        if (nkernels == 0)
        {
            char filename[1024];
            sprintf(filename, "%s/model/%s-conv%d-labels.txt", model_dir, basenames[k], skip);
            if (iftFileExists(filename))
            {
                /* load image model */
                nkernels_i = input->val[k]->num_feats; 
                truelabels_i = iftGFLIMLoadKernelLabels(filename, &nkernels_i);
                nlabels = iftGFLIMNumberOfTruelabels(truelabels_i, nkernels_i);
            }
            else iftError("File %s does not exist", "main", filename);
        }

        for (int l = 0; l < nlabels; l++)
        {
            char act_dir[1024];
            sprintf(act_dir, "%s/label%d/%s", output_dir, l + 1, basenames[k]);
            if (!iftDirExists(act_dir)) iftMakeDir(act_dir);
        }

        iftImage **temp = (iftImage **)iftAlloc(input->val[k]->num_feats, sizeof(iftImage *));
        #pragma omp parallel for
        for (int j = 0; j < input->val[k]->num_feats; j++)
        {
            char filename[1024];
            temp[j] = iftInterp2D(activations[j], scale[0], scale[1]);
            
            int label = 1;
            if(truelabels != NULL) label = truelabels[j];
            else {
                if(truelabels_i != NULL) label = truelabels_i[j];
            }

            sprintf(filename, "%s/label%d/%s/kernel-%03d.pgm", output_dir, label, basenames[k], j + 1);
            iftWriteImageByExt(temp[j], filename);
        }
        for (int j = 0; j < input->val[k]->num_feats; j++)
        {
            iftDestroyImage(&activations[j]);
            iftDestroyImage(&temp[j]);
        }
        
        iftFree(temp);
        iftFree(activations);
        iftFree(basenames[k]);
        iftFree(truelabels_i);
    }
    
    iftDestroyImageArray(&label_imgs);
    iftDestroyImageArray(&label_imgs0);
    iftDestroyGFLIMGraphArray(&input);
    iftFree(basenames);
}


int getLayersToConcat(iftGFLIMArch *arch, int layer, int skip_layer, iftIntArray *layers_to_concat, int index)
{
    while(skip_layer > 0){ // itera sobre as camadas indicadas no skip connection
        if(arch->layer[layer-1].skip_connection[skip_layer-1])
        {
            if(index == layers_to_concat->n) // realoca se necessario
            {
                layers_to_concat->n *= 2;
                layers_to_concat->val = realloc(layers_to_concat->val, (layers_to_concat->n * 2) * sizeof(int));
            }
            layers_to_concat->val[index++] = skip_layer; // adiciona a camada skip na lista
            // se for concat, busca recursivamente as camadas a serem concatenadas
            if(strcmp(arch->layer[skip_layer-1].skip_type, "concat") == 0)
                index = getLayersToConcat(arch, skip_layer, skip_layer-1, layers_to_concat, index);
        }
        skip_layer--;
    }
    return index;
}

int main(int argc, char *argv[])
{
    if (argc != 4 && argc != 5 && argc != 6)
    {
        iftError("Usage: iftGFLIMWriteActivations <P1> <P2> <P3> [<P4> <P5>]\n"
                 "[1] model directory \n"
                 "[2] layer number (1, 2,..) \n"
                 "[3] batch size \n"
                 "[4] skip connection (0: no skip, > 1: largest scale of skip) \n"
                 "[5] arch to be used when skip connection is concat (optional)"
                 , "main");
    }

    timer *tstart = iftTic();

    char *model_dir = argv[1];
    int layer = atoi(argv[2]);
    int batchsize = atoi(argv[3]);
    int skip = layer;
    iftGFLIMArch *arch = NULL;

    if (argc > 4) skip = atoi(argv[4]);
    if(argc > 5){
        arch = iftReadGFLIMArch(argv[5]);
    }

    if (skip < 1) iftError("Skip connection must be greater than 0", "main");

    char input_dir[1024], labels_dir[1024], input_net_labels[1024], output_dir[1024];

    sprintf(input_dir, "%s/layer%d", model_dir, layer);
    sprintf(input_net_labels, "%s/labels0", model_dir);
    sprintf(labels_dir, "%s/labels%d", model_dir, layer);

    validateArgs(model_dir, layer, input_dir, labels_dir, input_net_labels);

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", true);
    if (fs->n == 0)
        iftError("No images found in the input directory: %s", "main", input_dir);

    int ngraphs = fs->n;
    int nbatches = 0;
    int nremaining_graphs = fs->n;

    batchsize = iftMin(batchsize, ngraphs);
    nbatches = ngraphs / batchsize;
    nremaining_graphs = nremaining_graphs - nbatches * batchsize;

    int *truelabels = NULL, nlabels = 1, nkernels = 0;
    char filename[1024];
    sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, skip);
    if (iftFileExists(filename)) // merged layer in GFLIM-BoFP
    {
        if(arch == NULL || strcmp(arch->layer[layer-1].skip_type, "concat") != 0)
            truelabels = iftGFLIMLoadKernelLabels(filename, &nkernels);
        else{
            
            iftIntArray *layers_to_concat = iftCreateIntArray(arch->nlayers);
            layers_to_concat->val[0] = layer; // add current layer
            int nconcat_layers = getLayersToConcat(arch, layer, layer-1, layers_to_concat, 1);
                
            for(int i = 0; i < nconcat_layers; i++)
            {
                sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layers_to_concat->val[i]);
                if(!iftFileExists(filename))
                    iftError("File %s does not exist", "main", filename);
                    
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
            iftDestroyGFLIMArch(&arch);
        }
        nlabels = iftGFLIMNumberOfTruelabels(truelabels, nkernels);

        sprintf(output_dir, "%s/activation%d", model_dir, layer);
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
            sprintf(output_dir, "%s/activation%d", model_dir, layer);
        }
    }

    iftMakeDir(output_dir);
    
    for (int batch = 0; batch < nbatches; batch++)
    {
        int first = batch * batchsize, last = first + batchsize - 1;
        printf("Processing batch %d of %d batches\r", batch + 1, nbatches);
        fflush(stdout);

        ProcessBatch(fs, first, last, 
            model_dir, layer, skip, 
            nkernels, truelabels, nlabels,
            output_dir);
    }

    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last)
    {
        printf("Processing remaining %d images\r", last - first + 1);
        fflush(stdout);

        ProcessBatch(fs, first, last, 
            model_dir, layer, skip, 
            nkernels, truelabels, nlabels,
            output_dir);
    }
    
    if(nkernels != 0) iftFree(truelabels);
    iftDestroyFileSet(&fs);
    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
