/**
 * \file iftGFLIMActivDataSet.c
 * \brief Create GFLIM graph dataset from activation graphs.
 * \author Isabela Borlido
 */

#include "ift.h"

iftMImage *CropMImageBorder(iftMImage *img, int border_size)
{
  iftVoxel u;
  iftMImage *crop;

  if (iftIs3DMImage(img))
  {
    crop = iftCreateMImage(img->xsize - 2 * border_size, img->ysize - 2 * border_size, img->zsize - 2 * border_size, img->m);
    int q = 0;
    for (u.z = border_size; u.z < img->zsize - border_size; u.z++)
      for (u.y = border_size; u.y < img->ysize - border_size; u.y++)
        for (u.x = border_size; u.x < img->xsize - border_size; u.x++)
        {
          int p = iftMGetVoxelIndex(img, u);
          for (int b = 0; b < img->m; b++)
            crop->val[q][b] = img->val[p][b];
          q++;
        }
  }
  else
  {
    crop = iftCreateMImage(img->xsize - 2 * border_size, img->ysize - 2 * border_size, img->zsize, img->m);
    int q = 0;
    u.z = 0;
    for (u.y = border_size; u.y < img->ysize - border_size; u.y++)
      for (u.x = border_size; u.x < img->xsize - border_size; u.x++)
      {
        int p = iftMGetVoxelIndex(img, u);
        for (int b = 0; b < img->m; b++)
          crop->val[q][b] = img->val[p][b];
        q++;
      }
  }

  return (crop);
}

void iftCreateImgRefData(char *origImgsDir, char *imgBasename, char *origImgExt, iftFileSet *fsRefData, int fileId)
{
  char filename[2048];
  sprintf(filename, "%s/%s%s", origImgsDir, imgBasename, origImgExt);
  fsRefData->files[fileId] = iftCreateFile(filename);
}

void iftWriteKernelDataSetWithFileSet(iftDataSet *Z, iftFileSet *fs, iftSampleStatus status, char *outBasename)
{
  iftSetRefData(Z, fs, IFT_REF_DATA_FILESET);
  iftSetStatus(Z, status);
  if (Z->nclasses > 0)
    iftAddStatus(Z, IFT_SUPERVISED);
  char filename[2048];
  sprintf(filename, "%s.zip", outBasename);
  iftWriteDataSet(Z, filename);
  printf("File '%s' created\n", filename);
}

int main(int argc, char *argv[])
{
  timer *tstart = NULL;

  if (argc != 4 && argc != 5)
    iftError("Usage: iftGFLIMActivDataSet <...>\n"
             "[1] input activation graphs\n"
             "[2] path of the original images\n"
             "[3] output_status: Status for the output dataset (1: Train, 2: Test)\n"
             "[4] (optional) average activations and sort superpixels (1: average all channels, filename: average per class)\n"
             "argc = %d", argc,
             "main");

  tstart = iftTic();

  iftFileSet *fs = iftLoadFileSetFromDirBySuffix(argv[1], ".json", 1);
  iftFileSet *fsAux = iftLoadFileSetFromDirOrCSV(argv[2], 1, 1);
  iftSampleStatus outStatus = atoi(argv[3]) == 1 ? IFT_TRAIN : IFT_TEST;
  //int *true_labels = NULL;
  //int nkernels = 0, minlabel = 0, maxlabel = 0;
  bool average_act = false;
  char filename[1024];

  if (argc > 4)
  {
    if(atoi(argv[4]) != 1)
    {
      iftError("Not implemented: average activations per class from file '%s'", __func__, argv[4]);
      /*if(!iftFileExists(argv[4]))
        iftError("The provided filename for true labels does not exist: %s", __func__, argv[4]);
      true_labels = iftGFLIMLoadKernelLabels(argv[4], &nkernels);
      */
    }
    average_act = true;
  }

  char *origImgsDir = NULL, *origImgExt = NULL;
  
  origImgsDir = iftDirname(fsAux->files[0]->path);
  origImgExt = iftCopyString(iftFileExt(fsAux->files[0]->path));
  
  iftDestroyFileSet(&fsAux);

  int first = 0;
  int last = fs->n - 1;
  int nclasses = iftFileSetLabelsNumber(fs);
  char *outBasename = iftBasename(argv[1]);
  iftRightTrim(outBasename, '/');

  //-----
  iftGFLIMGraph *graph0 = iftReadGFLIMGraph(fs->files[0]->path);
  int nfeats = graph0->num_feats * graph0->num_nodes;
  iftDestroyGFLIMGraph(&graph0);
  //-----

  iftDataSet *Z = iftCreateDataSet(fs->n, nfeats);

  iftFileSet *fsRefData = iftCreateFileSet(fs->n);

  Z->nclasses = nclasses;
  
  sprintf(filename, "%s_id_image.txt", argv[1]);
  FILE *fp = fopen(filename, "w");
  for (int i = first; i <= last; i++)
  {
    fprintf(fp, "id: %d image: %s\n", i, fs->files[i]->path);
  }
  fclose(fp);

  for (int i = first; i <= last; i++)
  {
    char *basename = iftFilename(fs->files[i]->path, ".json");
    //printf("Processing file %s: %d of %ld files\r", basename, i + 1, fs->n);
    fflush(stdout);

    iftGFLIMGraph *graph = iftReadGFLIMGraph(fs->files[i]->path);

    Z->sample[i].id = i;
    Z->sample[i].truelabel = fs->files[i]->label;
    
    if(average_act)
    {
      float *ch_feat = iftAllocFloatArray(graph->num_nodes);
      int *index = iftAllocIntArray(graph->num_nodes);

      for (int b = 0; b < graph->num_feats; b++)
      {
        for (int k=0; k < graph->num_nodes; k++){
          ch_feat[k]=graph->nodes[k].feats[b]; 
          index[k]=k; 
        }
        
        /* Sort the values */
        iftFHeapSort(ch_feat, index, graph->num_nodes, IFT_DECREASING);

        for (int k = 0, j = b * graph->num_nodes; k < graph->num_nodes; k++, j++)
          Z->sample[i].feat[j] = ch_feat[k];
      }

      iftFree(ch_feat);
      iftFree(index);
    } else {
      int j = 0;
      for (int p = 0; p < graph->num_nodes; p++)
      {
        for (int b = 0; b < graph->num_feats; b++, j++)
          Z->sample[i].feat[j] = graph->nodes[p].feats[b];
      }
    }
    
    iftCreateImgRefData(origImgsDir, basename, origImgExt, fsRefData, i);
    iftDestroyGFLIMGraph(&graph);
    iftFree(basename);
  }

  /* write the dataset and create the projection */
  iftWriteKernelDataSetWithFileSet(Z, fsRefData, outStatus, outBasename);

  iftDestroyFileSet(&fs);
  iftDestroyFileSet(&fsRefData);
  iftDestroyDataSet(&Z);
  iftFree(outBasename);
  iftFree(origImgsDir);
  iftFree(origImgExt);

  char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
  printf("Done ... %s\n", time);
  iftFree(time);
  return (0);
}
