/**
 * \file iftGFLIMDataSetActiv.c
 * \brief Create GFLIM graph dataset from feature dataset and superpixel images.
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

iftGFLIMGraph *iftComputeGFLIMGraphFromMatrix(
                    double **matrix,
                    int nnodes,
                    int nfeats,
                    iftImage *labeled_img)
{
    iftAdjRel *A = iftCircular(sqrtf(2.0));
    iftBMap *neighbors[nnodes];
    iftGFLIMGraph *graph = iftCreateGFLIMGraph(nfeats, nnodes);

    for (int i = 0; i < nnodes; i++)
        neighbors[i] = iftCreateBMap(nnodes);

    // for each pixel
    for (int p_index = 0; p_index < labeled_img->n; p_index++)
    {
        int label = (int)(labeled_img->val[p_index] - 1); // get its label
        if(label < 0) continue; // ignore values outside the mask
        iftGFLIMGraphNode *node = &(graph->nodes[label]); // get the node in the graph
        node->numPixels++;

        // look at neighbor pixels
        iftVoxel p_voxel = iftGetVoxelCoord(labeled_img, p_index);
        for (int i = 1; i < A->n; i++)
        {
            iftVoxel q_voxel = iftGetAdjacentVoxel(A, p_voxel, i);
            if (iftValidVoxel(labeled_img, q_voxel))
            {
                // get the label of the neighbor pixel
                int q_label = (int)(labeled_img->val[iftMGetVoxelIndex(labeled_img, q_voxel)] - 1);

                // check if neighbor relationship (the edge) is already in the graph
                // since the graph is undirected, we check only one side and add both
                if (q_label >= 0 && label > q_label && !iftBMapValue(neighbors[label], q_label))
                {
                    iftGFLIMGraphInsertNeighborNode(node, q_label);
                    iftGFLIMGraphInsertNeighborNode(&(graph->nodes[q_label]), label);
                    iftBMapSet1(neighbors[label], q_label);
                    graph->num_edges++;
                }
            }
        }
    }

    for (int spx_index = 0; spx_index < nnodes; spx_index++)
    {
      for(int j = 0; j < nfeats; j++)
        graph->nodes[spx_index].feats[j] = matrix[spx_index][j];

      iftDestroyBMap(&neighbors[spx_index]);
    }
    iftDestroyAdjRel(&A);

    return graph;
}

int main(int argc, char *argv[])
{
  timer *tstart = NULL;

  if (argc != 4)
    iftError("Usage: iftActivDataSet <...>\n"
             "[1] input dataset\n"
             "[2] superpixel images directory\n"
             "[3] output_dir: Directory to save the output dataset\n",
             "main");

  tstart = iftTic();

  iftDataSet *Z = iftReadDataSet(argv[1]);
  char *spx_img_dir = argv[2];
  char *output_dir = argv[3];

  // check if superpixel images directory is valid
  if(!iftDirExists(spx_img_dir))
    iftError("Superpixel images directory does not exist: %s", __func__, spx_img_dir);

  iftFileSet *fs = Z->ref_data;
  printf("- nsamples: %d, nfeats: %d, nclasses: %d\n", Z->nsamples, Z->nfeats, Z->nclasses);

  if(!iftDirExists(output_dir))
    iftMakeDir(output_dir);
    
  printf("\n");
  for(int s = 0; s < Z->nsamples; s++)
  {
    int i = Z->sample[s].id;
    char ext[10];
    char filename[2048];
    char *basename = NULL;
    
    sprintf(ext, "%s", iftFileExt(fs->files[i]->path));
    basename = iftFilename(fs->files[i]->path, ext);

    printf("Processing sample %d/%d: %s\r", s + 1, Z->nsamples, basename);
    fflush(stdout);

    // read the superpixel image
    sprintf(filename, "%s/%s.pgm", spx_img_dir, basename);
    iftImage *spx_img = iftReadImageByExt(filename);

    int nnodes = iftMaximumValue(spx_img);
    int nfeats = Z->nfeats / nnodes;

    // compute the feature matrix with the graph features
    double **matrix = (double**)calloc(nnodes, sizeof(double*));
    for(int r = 0; r < nnodes; r++)
      matrix[r] = (double*)calloc(nfeats, sizeof(double));

    int j = 0;
    for(int r = 0; r < nnodes; r++)
    {
      for(int c = 0; c < nfeats; c++)
      {
        matrix[r][c] = Z->sample[s].feat[j];
        j++;
      }
    }

    // convert feature matrix to graph using superpixel image as reference
    iftGFLIMGraph *graph = iftComputeGFLIMGraphFromMatrix(matrix, nnodes, nfeats,spx_img);
    // free feature matrix
    for(int r = 0; r < nnodes; r++)
      free(matrix[r]);
    free(matrix);
    iftDestroyImage(&spx_img);

    // save graph
    sprintf(filename, "%s/%s.json", output_dir, basename);
    iftWriteGFLIMGraph(graph, filename);
    iftDestroyGFLIMGraph(&graph);

    iftFree(basename);
  }
  printf("\n");

  fs = NULL;
  iftDestroyDataSet(&Z);

  char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
  printf("Done ... %s\n", time);
  iftFree(time);
  return (0);
}
