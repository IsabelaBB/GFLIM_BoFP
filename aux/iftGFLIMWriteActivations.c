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
  if (argc != 3)
  {
    iftError("Usage: iftGFLIMWriteActivations <P1> <P2> <P3> \n"
             "[1] model directory \n"
             "[2] layer number (1, 2,..) \n",
             "main");
  }

  timer *tstart = iftTic();

  char *model_dir = argv[1];
  int layer = atoi(argv[2]);

  if(layer < 1) 
    iftError("Layer number must be greater than 0", "main");
  
  if(!iftDirExists(model_dir))
    iftError("Model directory does not exist: %s", "main", model_dir);

  char input_dir[200], labels_dir[200], input_net_labels[200], output_dir[200];

  sprintf(input_dir, "%s/layer%d", model_dir, layer);
  if (!iftDirExists(input_dir))
    iftError("Graph directory does not exist: %s", "main", input_dir);

  sprintf(input_net_labels, "%s/labels0", model_dir);
  if(!iftDirExists(input_net_labels))
    iftError("Labels directory input network does not exist: %s", "main", input_net_labels);

  sprintf(labels_dir, "%s/labels%d", model_dir, layer);
  if(!iftDirExists(labels_dir))
    sprintf(labels_dir, "%s/labels0", model_dir);

  sprintf(output_dir, "%s/activation%d", model_dir, layer);
  iftMakeDir(output_dir);
  
  iftFileSet *fs = iftLoadFileSetFromDirBySuffix(input_dir, ".json", true);
  if(fs->n == 0)
    iftError("No images found in the input directory: %s", "main", input_dir);
  
  char act_dir[512];
  char filename[512];

  for (int i = 0; i < fs->n; i++)
  {
    printf("Processing image %d of %ld\r", i + 1, fs->n);
    char *basename = iftFilename(fs->files[i]->path, ".json");
    
    iftFLIMGraph *graph = iftReadFLIMGraph(fs->files[i]->path);
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

    iftImage **activations = iftFlimGraphToActivations(graph, label_img);
    iftDestroyFLIMGraph(&graph);
    iftDestroyImage(&label_img);
    
    if (nlabels != 0)
    {
      for (int l = 0; l < nlabels; l++)
      {
        sprintf(act_dir, "%s/label%d/%s", output_dir, l+1, basename);
        iftMakeDir(act_dir);
        for (int k = 0; k < nkernels; k++)
        {
          if ((l + 1) == truelabels[k]){
            iftImage *temp = iftInterp2D(activations[k], scale[0], scale[1]);
            sprintf(filename, "%s/kernel-%03d.%s", act_dir, k + 1, "pgm");
            iftWriteImageByExt(temp, filename);
            iftDestroyImage(&temp);
          }
        }
      }
      for (int k = 0; k < nkernels; k++)
        iftDestroyImage(&activations[k]);
    }
    else
    {
      sprintf(act_dir, "%s/label%d/%s", output_dir, 1, basename);
      iftMakeDir(act_dir);
      for (int k = 0; k < nkernels; k++)
      {
        iftImage *temp = iftInterp2D(activations[k], scale[0], scale[1]);
        sprintf(filename, "%s/kernel-%03d.%s", act_dir, k + 1, "pgm");
        iftWriteImageByExt(temp, filename);
        iftDestroyImage(&temp);
        iftDestroyImage(&activations[k]);
      }
    }
    iftFree(activations);
    iftFree(truelabels);
    iftFree(basename);
  }
  iftDestroyFileSet(&fs);

  printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

  return (0);
}
