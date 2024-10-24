#include "ift.h"

int *LoadKernelLabels(char *filename)
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

  return (truelabels);
}

float *AdaptiveWeightsOtsu(iftMImage *mimg)
{
  float *weight = iftAllocFloatArray(mimg->m);
  float *mean = iftAllocFloatArray(mimg->m);
  float *maxval = iftAllocFloatArray(mimg->m);

  /* Normalize activations */
  for (int b = 0; b < mimg->m; b++)
  {
    maxval[b] = iftMMaximumValue(mimg, b);
  }

  // #pragma omp parallel for shared(mimg, maxval)
  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      if (maxval[b] > 0.0)
        mimg->val[p][b] /= maxval[b];
      else
        mimg->val[p][b] = 0.0;
    }
  }
  iftFree(maxval);

  /* Estimate the mean activation per channel */

  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      mean[b] += mimg->val[p][b];
    }
  }

  // Just so it is easier to compute the otsu
  iftImage *means_image = iftCreateImage(mimg->m, 1, 1);
  float max_mean = 0.0;
  for (int b = 0; b < mimg->m; b++)
  {
    mean[b] /= mimg->n;
    means_image->val[b] = mean[b] * 255;
    if (means_image->val[b] > max_mean)
      max_mean = means_image->val[b];
  }
  if (max_mean > 0.0)
  {

    /* Estimate mean and standard deviation of the above mean activations */

    float mu = 0.0, stdev = 0.0;
    for (int b = 0; b < mimg->m; b++)
    {
      mu += mean[b];
    }
    mu /= mimg->m;

    int means_otsu = iftOtsu(means_image);

    for (int b = 0; b < mimg->m; b++)
    {
      stdev += (mean[b] - mu) * (mean[b] - mu);
    }
    stdev = sqrtf(stdev / mimg->m) * sqrtf(stdev / mimg->m);

    float low_end, high_end;
    if (means_otsu > 0.1)
      low_end = means_otsu - stdev;
    else
      low_end = means_otsu;
    high_end = means_otsu + stdev;

    /* Estimate the weight of each channel */

    for (int b = 0; b < mimg->m; b++)
    {
      iftImage *band = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
      for (int p = 0; p < band->n; p++)
        band->val[p] = mimg->val[p][b] * 255;
      float fg_proportion = 0;
      int max_band = 0.0;
      for (int p = 0; p < band->n; p++)
      {
        if (band->val[p] > max_band)
          max_band = band->val[p];
      }
      if (max_band > 0.0)
      {
        int otsu_band = iftOtsu(band);

        for (int p = 0; p < band->n; p++)
        {
          if (band->val[p] >= otsu_band)
            fg_proportion += 1;
        }
        fg_proportion /= band->n;
        if (mean[b] >= high_end && fg_proportion > 0.2)
        {
          weight[b] = -1.0;
        }
        else if (mean[b] <= low_end && fg_proportion < 0.1)
        {
          weight[b] = +1.0;
        }
        else
        {
          weight[b] = 0.0;
        }
      }
      else
      {
        weight[b] = 0.0;
      }
    }
  }
  else
  {
    for (int b = 0; b < mimg->m; b++)
      weight[b] = 0.0;
  }

  iftFree(mean);
  iftDestroyImage(&means_image);

  return (weight);
}

float *AdaptiveWeights(iftMImage *mimg)
{
  float *weight = iftAllocFloatArray(mimg->m);
  float *mean = iftAllocFloatArray(mimg->m);
  float *maxval = iftAllocFloatArray(mimg->m);

  /* Normalize activations */
  for (int b = 0; b < mimg->m; b++)
  {
    maxval[b] = iftMMaximumValue(mimg, b);
  }

#pragma omp parallel for shared(mimg, maxval)
  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      mimg->val[p][b] /= maxval[b];
    }
  }
  iftFree(maxval);

  /* Estimate the mean activation per channel */

  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      mean[b] += mimg->val[p][b];
    }
  }

  for (int b = 0; b < mimg->m; b++)
  {
    mean[b] /= mimg->n;
  }

  /* Estimate mean and standard deviation of the above mean activations */

  float mu = 0.0, stdev = 0.0;
  for (int b = 0; b < mimg->m; b++)
  {
    mu += mean[b];
  }
  mu /= mimg->m;

  for (int b = 0; b < mimg->m; b++)
  {
    stdev += powf(mean[b] - mu, 2);
  }
  stdev = sqrtf(stdev / mimg->m) * sqrtf(stdev / mimg->m);

  /* Estimate the weight of each channel */

  for (int b = 0; b < mimg->m; b++)
  {
    if (mean[b] <= mu - stdev)
    {
      weight[b] = +1.0;
    }
    else
    {
      if (mean[b] >= mu + stdev)
      {
        weight[b] = -1.0;
      }
      else
      {
        weight[b] = 0.0;
      }
    }
  }

  iftFree(mean);

  return (weight);
}

iftFImage *FixedWeightDecoder(iftMImage *mimg, float *weight)
{
  iftFImage *salie = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);

  /* decode layer */

#pragma omp parallel for shared(salie, mimg, weight)
  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      salie->val[p] += mimg->val[p][b] * weight[b];
    }
    if (salie->val[p] < 0)
      salie->val[p] = 0; /* ReLU (Sigmoid?) */
  }
  return (salie);
}

iftFImage *AdaptiveDecoder1(iftMImage *mimg)
{
  iftFImage *salie = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);
  float *weight = AdaptiveWeightsOtsu(mimg);

  /* decode layer */
  // #pragma omp parallel for shared(salie, mimg, weight)
  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      salie->val[p] += mimg->val[p][b] * weight[b];
    }
    if (salie->val[p] < 0)
      salie->val[p] = 0; /* ReLU (Sigmoid?) */
  }

  iftFree(weight);

  return (salie);
}

iftFImage *AdaptiveDecoder2(iftMImage *mimg, float adj_radius, float *channel_label)
{
  iftFImage *salie = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);
  float *weight = iftAllocFloatArray(mimg->m);
  float *maxval = iftAllocFloatArray(mimg->m);
  int nobj = 0, nbkg = 0;

  /* Normalize activations per channel */

  for (int b = 0; b < mimg->m; b++)
  {
    maxval[b] = iftMMaximumValue(mimg, b);
    if (channel_label[b] > 0)
    { /* object channel */
      nobj++;
    }
    else
    { /* background channel */
      nbkg++;
    }
  }

  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      if (maxval[b] > IFT_EPSILON)
        mimg->val[p][b] = mimg->val[p][b] / maxval[b];
    }
  }
  iftFree(maxval);

  /* Estimate saliency of each pixel */

  iftAdjRel *A = NULL;
  if (iftIs3DMImage(mimg))
    A = iftSpheric(adj_radius);
  else
    A = iftCircular(adj_radius);

  for (int p = 0; p < mimg->n; p++)
  {

    /* estimate mean and stdev of activations across channels */

    float mean[2] = {0.0, 0.0}, var[2] = {0.0, 0.0};
    iftVoxel u = iftMGetVoxelCoord(mimg, p);

    for (int i = 0; i < A->n; i++)
    {
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);

      if (iftMValidVoxel(mimg, v))
      {
        int q = iftMGetVoxelIndex(mimg, v);

        for (int b = 0; b < mimg->m; b++)
        {
          if (channel_label[b] > 0)
          { /* object channel */
            mean[0] += mimg->val[q][b];
          }
          else
          { /* background channel */
            mean[1] += mimg->val[q][b];
          }
        }
      }
    }
    mean[0] /= (nobj * A->n);
    mean[1] /= (nbkg * A->n);

    for (int i = 0; i < A->n; i++)
    {
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);

      if (iftMValidVoxel(mimg, v))
      {
        int q = iftMGetVoxelIndex(mimg, v);

        for (int b = 0; b < mimg->m; b++)
        {
          if (channel_label[b] > 0)
          { /* object channel */
            var[0] += powf(mimg->val[q][b] - mean[0], 2);
          }
          else
          { /* background channel */
            var[1] += powf(mimg->val[q][b] - mean[1], 2);
          }
        }
      }
    }
    var[0] = var[0] / (nobj * A->n);
    var[1] = var[1] / (nbkg * A->n);

    /* It is expected two Gaussian distributions G0 and G1 of
       activations around mean[0] and mean[1]. We may expect that
       object pixels generate mean[0] >> mean[1] and the other way
       around for background pixels. We may use the the probability
       values of these distributions to estimate weights. */

    float P0, P1;
    for (int b = 0; b < mimg->m; b++)
    {
      if (var[0] > IFT_EPSILON)
        P0 = exp(-powf(mimg->val[p][b] - mean[0], 2) / (2.0 * var[0]));
      else
      {
        var[0] = 1.0;
        P0 = exp(-powf(mimg->val[p][b] - mean[0], 2) / (2.0 * var[0]));
      }
      if (var[1] > IFT_EPSILON)
        P1 = exp(-powf(mimg->val[p][b] - mean[1], 2) / (2.0 * var[1]));
      else
      {
        var[1] = 1.0;
        P1 = exp(-powf(mimg->val[p][b] - mean[1], 2) / (2.0 * var[1]));
      }

      if ((channel_label[b] > 0) && (P0 > P1))
      {
        weight[b] = 1; // P0;
      }
      else
      {
        if ((channel_label[b] < 0) && (P0 < P1))
        {
          weight[b] = -1; // P1
        }
        else
        {
          weight[b] = 0;
        }
      }
    }

    /* Estimate the salience map using one neuron per pixel -- it is
       no longer a point-wise convolution */

    for (int b = 0; b < mimg->m; b++)
    {
      salie->val[p] += mimg->val[p][b] * weight[b];
    }
    if (salie->val[p] < 0) /* RELU (Sigmoid?) */
      salie->val[p] = 0;
  }
  iftDestroyAdjRel(&A);
  iftFree(weight);

  return (salie);
}

int NumberOfTruelabels(int *truelabels, int n)
{
  int nlabels = 0;
  for (int i = 0; i < n; i++)
    if (truelabels[i] > nlabels)
      nlabels = truelabels[i];
  return (nlabels);
}

int main(int argc, char *argv[])
{
  /* Example: iftDecodeLayer 1 arch.json flim_models salie [masks] */

  if ((argc != 4) && (argc != 5))
  {
    iftError("Usage: iftDecodeLayer <P1> <P2> <P3> <P4> \n"
             "[1] model directory \n"
             "[2] layer number (1, 2,..) \n"
             "[3] architecture of the network\n"
             "[4] optional folder with regions of interest (.nii.gz or .png)\n",
             "main");
  }

  timer *tstart = iftTic();

  char *model_dir = argv[1];
  int layer = atoi(argv[2]);
  iftFLIMArch *arch = iftReadFLIMArch(argv[3]);
  char *roi_dir = NULL;

  if (argc == 5)
    roi_dir = argv[4];

  if(layer < 1) 
    iftError("Layer number must be greater than 0", "main");
  
  if(!iftDirExists(model_dir))
    iftError("Model directory does not exist: %s", "main", model_dir);

  char input_dir[200], labels_dir[200], output_dir[200];

  sprintf(input_dir, "%s/layer%d", model_dir, layer);
  if (!iftDirExists(input_dir))
    iftError("Graph directory does not exist: %s", "main", input_dir);

  sprintf(labels_dir, "%s/labels0", model_dir);
  if(!iftDirExists(labels_dir))
    iftError("Labels directory does not exist: %s", "main", labels_dir);

  sprintf(output_dir, "%s/saliency%d", model_dir, layer);
  iftMakeDir(output_dir);
  
  iftFileSet *fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", true);
  if(fs->n == 0)
    iftError("No images found in the input directory: %s", "main", input_dir);
  
  iftMakeDir(output_dir);
  
  char *filename = iftAllocCharArray(512);
  char sal_dir[512];
  int Imax;

  for (int i = 0; i < fs->n; i++)
  {
    printf("Processing image %d of %ld\r", i + 1, fs->n);
    char *basename = iftFilename(fs->files[i]->path, ".json");
    
    iftFLIMGraph *graph = iftReadFLIMGraph(fs->files[i]->path);
    sprintf(filename, "%s/%s.pgm", labels_dir, basename);
    iftImage *label_img = iftReadImageByExt(filename);
    iftMImage *mimg = iftFLIMGraphToMImage(graph, label_img);
    iftDestroyImage(&label_img);
    int nkernels = graph->num_feats;
    iftDestroyFLIMGraph(&graph);
    
    iftImage *mask = NULL;
    
    /* Compute the scale factor w.r.t. the input of the network */
    float scale[] = {1.0, 1.0, 1.0};

    /* Load labels indicating class (object) per channel */

    int *truelabels = NULL, nlabels = 0;
    sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer);
    if (iftFileExists(filename))
    { /* load merged model */
      truelabels = LoadKernelLabels(filename);
      nlabels = NumberOfTruelabels(truelabels, nkernels);
    }
    else
    {
      sprintf(filename, "%s/model/%s-conv%d-labels.txt",
              model_dir, basename, layer);
      if (iftFileExists(filename))
      { /* load image model */
        truelabels = LoadKernelLabels(filename);
        nlabels = NumberOfTruelabels(truelabels, nkernels);
      }
    }

    /* Estimate one salience map per object (class) */

    iftFImage **salie = NULL;
    if (nlabels != 0)
    {
      salie = (iftFImage **)calloc(nlabels, sizeof(iftFImage *));
      float *weight = iftAllocFloatArray(nkernels);
      for (int l = 0; l < nlabels; l++)
      {
        /* set weights */
        for (int k = 0; k < nkernels; k++)
        {
          if ((l + 1) == truelabels[k])
            weight[k] = +1;
          else
            weight[k] = -1;
        }
        // salie[l]  = FixedWeightDecoder(mimg,weight);
        salie[l] = AdaptiveDecoder2(mimg, 1.5, weight);
      }
      iftFree(weight);
    }
    else
    {
      salie = (iftFImage **)calloc(1, sizeof(iftFImage *));
      salie[0] = AdaptiveDecoder1(mimg);
      nlabels = 1;
    }

    for (int l = 0; l < nlabels; l++)
    {
      /* Interpolate and normalize the salience maps */

      iftFImage *smap_temp = NULL;
      if (iftIs3DMImage(mimg))
      {
        smap_temp = iftFInterp(salie[l], scale[0], scale[1], scale[2]);
        Imax = 4095;
      }
      else
      {
        smap_temp = iftFInterp2D(salie[l], scale[0], scale[1]);
        Imax = 255;
      }

      iftImage *smap = iftFImageToImage(smap_temp, Imax);
      iftDestroyFImage(&smap_temp);

      /* post-process and save the salience map */

      if (argc == 6)
      {
        if (iftIs3DMImage(mimg))
        {
          sprintf(filename, "%s/%s.nii.gz", roi_dir, basename);
        }
        else
        {
          sprintf(filename, "%s/%s.png", roi_dir, basename);
        }
        mask = iftReadImageByExt(filename);
      }

      sprintf(sal_dir, "%s/label%d", output_dir, l + 1);
      if (!iftDirExists(sal_dir))
        iftMakeDir(sal_dir);

      if (iftIs3DMImage(mimg))
      {
        sprintf(filename, "%s/%s_layer%d.nii.gz", sal_dir, basename, layer);
      }
      else
      {
        sprintf(filename, "%s/%s_layer%d.png", sal_dir, basename, layer);
      }

      if (mask != NULL)
      {
        iftImage *aux = iftMask(smap, mask);
        iftDestroyImage(&smap);
        smap = aux;
        iftWriteImageByExt(smap, filename);
        iftDestroyImage(&mask);
      }
      else
      {
        iftWriteImageByExt(smap, filename);
      }
      iftDestroyImage(&smap);
      iftDestroyFImage(&salie[l]);
    }

    iftDestroyMImage(&mimg);
    iftFree(basename);
  }

  iftFree(filename);
  iftDestroyFileSet(&fs);
  iftDestroyFLIMArch(&arch);

  printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

  return (0);
}
