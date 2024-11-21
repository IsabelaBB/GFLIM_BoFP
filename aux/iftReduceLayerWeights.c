#include "ift.h"
#include "iftGFLIM.h"

float *LoadBias(char *basepath)
{
    int number_of_kernels;
    char filename[200];
    FILE *fp;
    float *bias;

    sprintf(filename, "%s-bias.txt", basepath);
    fp = fopen(filename, "r");
    fscanf(fp, "%d", &number_of_kernels);
    bias = iftAllocFloatArray(number_of_kernels);
    for (int k = 0; k < number_of_kernels; k++)
    {
        fscanf(fp, "%f ", &bias[k]);
    }
    fclose(fp);

    return (bias);
}

int *LoadKernelLabels(char *filename, int *n)
{
    int number_of_kernels;
    FILE *fp;
    int *truelabels;

    fp = fopen(filename, "r");
    fscanf(fp, "%d", &number_of_kernels);
    truelabels = iftAllocIntArray(number_of_kernels);
    for (int k = 0; k < number_of_kernels; k++)
    {
        fscanf(fp, "%d ", &truelabels[k]);
    }
    fclose(fp);
    *n = number_of_kernels;
    return (truelabels);
}

void SaveBias(char *basepath, float *bias, int number_of_kernels)
{
    char filename[200];
    FILE *fp;

    sprintf(filename, "%s-bias.txt", basepath);
    fp = fopen(filename, "w");
    fprintf(fp, "%d\n", number_of_kernels);
    for (int k = 0; k < number_of_kernels; k++)
    {
        fprintf(fp, "%f ", bias[k]);
    }
    fclose(fp);
}

void SaveKernelLabels(char *basepath, int *truelabel, int number_of_kernels)
{
    char filename[200];
    FILE *fp;

    sprintf(filename, "%s-labels.txt", basepath);
    fp = fopen(filename, "w");
    fprintf(fp, "%d\n", number_of_kernels);
    for (int k = 0; k < number_of_kernels; k++)
        fprintf(fp, "%d ", truelabel[k]);

    fclose(fp);
}

int NumberOfTruelabels(int *truelabels, int n)
{
  int nlabels = 0;
  for (int i = 0; i < n; i++)
    if (truelabels[i] > nlabels)
      nlabels = truelabels[i];
  return (nlabels);
}

void normalizeImage(iftMImage *mimg)
{
  float *maxval = iftAllocFloatArray(mimg->m);
  float *minval = iftAllocFloatArray(mimg->m);

  /* Normalize activations */
  for (int b = 0; b < mimg->m; b++){
    maxval[b] = iftMMaximumValue(mimg, b);
    minval[b] = iftMMinimumValue(mimg, b);
  }

  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      mimg->val[p][b] = (mimg->val[p][b]-minval[b])/(maxval[b]-minval[b]);
    }
  }
  iftFree(minval);
  iftFree(maxval);
}

double* ComputeMAE(iftMImage *gt_img, iftMImage *pred_img, int *trueLabels)
{
  // assume gt_img->val[p] in {0, 1} and pred_img->val[p][b] in [0, 1]
  double *mae = iftAllocDoubleArray(pred_img->m);
  for (int p = 0; p < gt_img->n; p++)
  {
    for(int b = 0; b < pred_img->m; b++)
    {
      if(trueLabels[b] == 1) // background 
        mae[b] += fabs(1 - gt_img->val[p][0] - pred_img->val[p][b]);
      else // foreground
        mae[b] += fabs(gt_img->val[p][0] - pred_img->val[p][b]);
    }
  }
  for(int b = 0; b < pred_img->m; b++) mae[b] /= (double)gt_img->n;
  return mae;
}

void WriteActivations(iftMImage *mimg, iftImage *label_img, 
                  char *model_dir, int layer, 
                  int *truelabels, int nlabels,
                  char *basename)
{
    
    char filename[200], act_dir[200];

    /* Compute the scale factor w.r.t. the input of the network */
    float scale[3];
    sprintf(filename, "%s/labels0/%s.pgm", model_dir, basename);
    iftImage *input = iftReadImageByExt(filename);
    scale[0] = (float)input->xsize / (float)label_img->xsize;
    scale[1] = (float)input->ysize / (float)label_img->ysize;
    scale[2] = (float)input->zsize / (float)label_img->zsize;
    iftDestroyImage(&input);

    iftMImage *mimg_interp = iftMInterp2D(mimg, scale[0], scale[1]);
    
    if (nlabels != 0)
    {
      for (int l = 0; l < nlabels; l++)
      {
        sprintf(act_dir, "%s/activations%d/label%d/", model_dir, layer, l+1);
        iftMakeDir(act_dir);
        sprintf(act_dir, "%s/activations%d/label%d/%s", model_dir, layer, l+1, basename);
        iftMakeDir(act_dir);
        for (int k = 0; k < mimg->m; k++)
        {
          if ((l + 1) == truelabels[k])
          {
            iftImage *temp = iftMImageToImage(mimg_interp, 255, k);
            sprintf(filename, "%s/kernel-%03d.%s", act_dir, k + 1, "pgm");
            iftWriteImageByExt(temp, filename);
            iftDestroyImage(&temp);
          }
        }
      }
    }
    else
    {
      sprintf(act_dir, "%s/activations%d/label%d/%s", model_dir, layer, 1, basename);
      iftMakeDir(act_dir);
      for (int k = 0; k < mimg->m; k++)
      {
        iftImage *temp = iftMImageToImage(mimg_interp, 255, k);
        sprintf(filename, "%s/kernel-%03d.%s", act_dir, k + 1, "pgm");
        iftWriteImageByExt(temp, filename);
        iftDestroyImage(&temp);
      }
    }
    iftDestroyMImage(&mimg_interp);
}

int main(int argc, char *argv[])
{
  timer *tstart;

  if (argc != 5)
    iftError("Usage: iftReduceMergedLayerWeights <P1> <P2> <P3> <P4> \n"
             "[1] model directory \n"
             "[2] layer number \n"
             "[3] label image directory \n"
             "[4] ground truth directory \n",
             "main");

  tstart = iftTic();

  char *model_dir = argv[1];
  int layer = atoi(argv[2]);
  char *label_dir = argv[3];
  char *gt_dir = argv[4];

  if (layer < 1)
    iftError("Invalid layer number: %d", "main", layer);

  if (!iftDirExists(model_dir))
    iftError("Model directory does not exist", "main");
  
  if (!iftDirExists(label_dir))
    iftError("Label image directory does not exist", "main");

  char input_dir[200];
  char *filename = iftAllocCharArray(512);
  int nkernels = 0;
  
  sprintf(input_dir, "%s/layer%d", model_dir, layer);
  if (!iftDirExists(input_dir))
    iftError("Graph directory does not exist", "main");

  iftFileSet *fs = iftLoadFileSetFromDirBySuffix(gt_dir, ".png", 1);
  if(fs->n < 1)
    iftError("Ground truth directory %s is empty", "main", gt_dir);

  sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer);
  int *truelabels = LoadKernelLabels(filename, &nkernels);
  int nlabels = NumberOfTruelabels(truelabels, nkernels);
  
  /*sprintf(filename, "%s/activations%d", model_dir, layer);
  iftMakeDir(filename);*/
  
  int *kernels_per_label = iftAllocIntArray(nlabels);
  for(int b=0; b < nkernels; b++)
    kernels_per_label[truelabels[b]-1]++;

  bool *maintain_kernel = iftAllocBoolArray(nkernels);

  for (int i = 0; i < fs->n; i++)
  {
    double *MAE_img = NULL;
    char *basename = iftFilename(fs->files[i]->path, ".png");

    sprintf(filename, "%s/%s.json", input_dir, basename);
    iftFLIMGraph *graph = iftReadFLIMGraph(filename);

    if(graph->num_feats != nkernels)
    {
      iftFree(basename);
      iftDestroyFLIMGraph(&graph);
      iftError("Number of kernels does not match the number of activations: %s, %ld != %d", "main", filename, graph->num_feats, nkernels);
      continue;
    }

    iftImage *gt_img = iftReadImageByExt(fs->files[i]->path);
    iftMImage *gt_mimg = iftImageToMImage(gt_img, GRAY_CSPACE);
    iftDestroyImage(&gt_img);
    normalizeImage(gt_mimg);

    if(gt_mimg->m > 1){
      iftError("Ground truth image must have only one band: %d", "main", gt_mimg->m);
    }

    sprintf(filename, "%s/%s.pgm", label_dir, basename);
    iftImage *label_img = iftReadImageByExt(filename);

    float scale[2];
    scale[0] = (float)gt_mimg->xsize / (float)label_img->xsize;
    scale[1] = (float)gt_mimg->ysize / (float)label_img->ysize;
    if(scale[0] > 0 || scale[1] > 0)
    {
      iftImage *label_interp = iftInterp2D(label_img, scale[0], scale[1]);
      iftDestroyImage(&label_img);
      label_img = label_interp;
    }
    
    iftMImage *mimg = iftFLIMGraphToMImage(graph, label_img);
    iftDestroyFLIMGraph(&graph);
    normalizeImage(mimg);

    /*printf("write activations \n");
    WriteActivations(mimg, label_img, 
                  model_dir, layer, 
                  truelabels, nlabels,
                  basename);*/
    iftDestroyImage(&label_img);

    // compute MAE  
    MAE_img = ComputeMAE(gt_mimg, mimg, truelabels);
    iftDestroyMImage(&gt_mimg);
    iftDestroyMImage(&mimg);

    for(int l = 0; l < nlabels; l++)
    {
      iftImage *MAE_means_otsu = iftCreateImage(kernels_per_label[l], 1, 1);

      int j = 0;
      for (int b = 0; b < nkernels; b++)
      {
        if (truelabels[b] == l + 1)
        {
          MAE_means_otsu->val[j] = MAE_img[b] * 255;
          j++;
        }
      }

      int otsu_MAE = iftOtsu(MAE_means_otsu);

      for (int b = 0, j = 0; b < nkernels; b++)
      {
        if (truelabels[b] == l + 1)
        {
          if (MAE_means_otsu->val[j] <= otsu_MAE)
          {
            maintain_kernel[b] = true;
          }
          j++;
        }
      }
      iftDestroyImage(&MAE_means_otsu);
    }
    iftFree(MAE_img);
    iftFree(basename);
  }
  
  iftDestroyFileSet(&fs);
  
  int num_new_kernels = 0;
  for(int b = 0; b < nkernels; b++)
  {
    if(maintain_kernel[b])
      num_new_kernels++;
  }
  
  iftFree(kernels_per_label);
  
  // load kernels and bias
  sprintf(filename, "%s/model/conv%d-kernels.npy", model_dir, layer);
  if(!iftFileExists(filename))
    iftError("Kernel file %s does not exist", "main", filename);
  iftMatrix *K = iftReadMatrix(filename);
  sprintf(filename, "%s/model/conv%d", model_dir, layer);
  float *bias = LoadBias(filename);

  // alloc new kernels and bias
  iftMatrix *newKernels = iftCreateMatrix(num_new_kernels, K->nrows);
  float *newBias = iftAllocFloatArray(num_new_kernels);
  int *newLabels = iftAllocIntArray(num_new_kernels);

  int j = 0;
  for(int b = 0; b < nkernels; b++)
  {
    if(maintain_kernel[b])
    {
      newLabels[j] = truelabels[b];
      newBias[j] = bias[b];
      for(int r = 0; r < K->nrows; r++)
        iftMatrixElem(newKernels, j, r) = iftMatrixElem(K, b, r);
      j++;
    }
  }
  iftFree(maintain_kernel);

  iftDestroyMatrix(&K);
  iftFree(bias);
  iftFree(truelabels);
  
  sprintf(filename, "%s/model/conv%d-kernels.npy", model_dir, layer);
  iftWriteMatrix(newKernels, filename);
  sprintf(filename, "%s/model/conv%d", model_dir, layer);
  SaveBias(filename, newBias, num_new_kernels);
  SaveKernelLabels(filename, newLabels, num_new_kernels);

  iftFree(newBias);
  iftFree(newLabels);
  iftDestroyMatrix(&newKernels);
  
  iftFree(filename);
  printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
  return (0);
}
