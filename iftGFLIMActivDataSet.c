/**
 * \file iftGFLIMActivDataSet.c
 * \brief Create an iftDataset (a .zip file) from GFLIM activation graphs.
 * \note The class labels are read from the images names. The image names must have the format classlabel_imagename.ext (e.g, 001_0002.png, where 1 is the class label and 2 is the image name).
 * \note The graphs must be in .json format and have the same name as the original image.
 * \author Isabela Borlido
 */

#include "ift.h"

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
    iftError("Usage: iftGFLIMActivDataSet P1 P2 P3 [P4]\n"
             "P1 : input activation graphs\n"
             "P2 : path of the original images\n"
             "P3 : output_status: Status for the output dataset (1: Train, 2: Test)\n"
             "P4 : (optional) whether to sort node features based on intensity per channel (1 or 0)\n"
             "argc = %d", __func__, argc);

  tstart = iftTic();

  iftFileSet *fs = iftLoadFileSetFromDirBySuffix(argv[1], ".json", 1);
  iftFileSet *fsAux = iftLoadFileSetFromDirOrCSV(argv[2], 1, 1);
  iftSampleStatus outStatus = atoi(argv[3]) == 1 ? IFT_TRAIN : IFT_TEST;
  bool sort_feats = false;
  char filename[1024];

  if (argc > 4)
  {
    if(atoi(argv[4]) != 1)
    {
      iftError("Not implemented: average activations per class from file '%s'", __func__, argv[4]);
    }
    sort_feats = true;
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
  printf("Number of features per graph: %d (%d nodes with %d features each)\n", nfeats, graph0->num_nodes, graph0->num_feats);
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
  printf("File '%s' created\n", filename);

  for (int i = first; i <= last; i++)
  {
    char *basename = iftFilename(fs->files[i]->path, ".json");
    iftGFLIMGraph *graph = iftReadGFLIMGraph(fs->files[i]->path);
    Z->sample[i].id = i;
    Z->sample[i].truelabel = fs->files[i]->label;
    
    if(sort_feats)
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
