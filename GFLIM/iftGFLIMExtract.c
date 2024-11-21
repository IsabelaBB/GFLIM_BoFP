#include "ift.h"

#define DEBUG 0

typedef struct iftFLIMGraphArray{
  iftFLIMGraph **val;
  int n;
} iftFLIMGraphArray;

iftFLIMGraphArray *CreateEmptyFLIMGraphArray(int n)
{
  iftFLIMGraphArray *array = (iftFLIMGraphArray *)calloc(1, sizeof(iftFLIMGraphArray));
  array->val = (iftFLIMGraph **)calloc(n, sizeof(iftFLIMGraph *));
  array->n = n;
  return array;
}

void iftDestroyLIMGraphArray(iftFLIMGraphArray **arr)
{
    iftFLIMGraphArray *aux = *arr;

    if (aux != NULL) {
        for(int i = 0; i < aux->n; i++) {
            if(aux->val[i] != NULL)
                iftDestroyFLIMGraph(&aux->val[i]);
        }
        iftFree(aux->val);
        iftFree(aux);
        *arr = NULL;
    }
}

iftFLIMGraphArray *ImageToFLIMGraphBatch(iftFileSet *fs, char *labels_dir, 
                                int first, int last, 
                                int initial_seeds, int nsuperpixels,
                                char **basename)
{
    iftFLIMGraphArray *input = CreateEmptyFLIMGraphArray(last - first + 1);

#pragma omp parallel for
    for (int k = 0; k <= last - first; k++)
    {
        int i = k + first;
        char ext[10], label_file[200];

        sprintf(ext, "%s", iftFileExt(fs->files[i]->path));
        basename[k] = iftFilename(fs->files[i]->path, ext);
        sprintf(label_file, "%s/%s.pgm", labels_dir, basename[k]);

        iftImage *img = iftReadImageByExt(fs->files[i]->path);
        iftImage *label_img = NULL;

        input->val[k] = iftImageToFLIMGraphDISF(img, initial_seeds, nsuperpixels, &label_img);
        iftWriteImageByExt(label_img, label_file);
        iftDestroyImage(&label_img);
    }
    return input;
}

iftMImage *convLayer(iftFLIMGraph *graph, iftMatrix *kernels, int kernel_size)
{
  iftMatrix *XI = iftFLIMGraphToFeatureMatrix(graph, kernel_size, 0, true);
  iftMatrix *XJ = iftMultMatrices(XI, kernels);
  iftDestroyMatrix(&XI);
  iftMImage *activ = iftMatrixToMImage(XJ, XJ->nrows,
                                       1, 1,
                                       kernels->ncols, 'c');
  iftDestroyMatrix(&XJ);
  return activ;
}

iftMImageArray *convLayerBatch(iftFLIMGraphArray *graph, iftMatrix *kernels, int kernel_size)
{
  iftMImageArray *activ = iftCreateMImageArray(graph->n);
  for(int i = 0; i < graph->n; i++){
    activ->val[i] = convLayer(graph->val[i], kernels, kernel_size);
  }
  return activ;
}

void pooling(iftMImage *activ, iftFLIMGraph *graph, 
              char *pool_type, int *pool_size, int pool_stride)
{
  iftMImage *pool;

  if (strcmp(pool_type, "no_pool") != 0)
  {
    if (strcmp(pool_type, "avg_pool") == 0)
    {
      pool = iftFLIMGraphAtrousAveragePooling(activ,
                                              graph,
                                              pool_size[0],
                                              pool_size[1],
                                              pool_size[2],
                                              1,
                                              pool_stride, true);
    }
    else
    {
      if (strcmp(pool_type, "max_pool") == 0)
      {
        pool = iftFLIMGraphAtrousMaxPooling(activ,
                                            graph,
                                            pool_size[0],
                                            pool_size[1],
                                            pool_size[2],
                                            1,
                                            pool_stride, true);
      }
      else
        iftError("Invalid pooling %s", "pooling", pool_type);
    }

#pragma omp parallel for
    for (int j = 0; j < graph->num_nodes; j++)
    {
      for (int b = 0; b < graph->num_feats; b++)
      {
        graph->nodes[j].feats[b] = (double)pool->val[j][b];
      }
    }
    iftDestroyMImage(&pool);
  }
}

void poolingArray(iftMImageArray *activ_array, iftFLIMGraphArray *graph, 
                    char *pool_type, int *pool_size, int pool_stride)
{
  for(int i = 0; i < activ_array->n; i++){
    iftFLIMGraph *pooling_graph = iftMImageToFLIMGraph(activ_array->val[i], graph->val[i]);
    iftDestroyFLIMGraph(&(graph->val[i]));
    pooling(activ_array->val[i], pooling_graph, pool_type, pool_size, pool_stride);
    graph->val[i] = pooling_graph;
    pooling_graph = NULL;
  }
}

void encodeLayerBatch(iftFLIMGraphArray *input, 
                iftFLIMLayer layer, int l, 
                char *model_dir, int first, int last,
                char **basename)
{
    int ninput_channels = input->val[0]->num_feats;
    
    if(DEBUG){
        printf("Load model\n");
        fflush(stdout);
    }
    /* Load model */
    char filename[200];
    sprintf(filename, "%s/model/conv%d-kernels.npy", model_dir, l+1);
    if(!iftFileExists(filename))
        iftError("Model file not found: %s", "encodeLayerBatch", filename);
    iftMatrix *kernelbank = iftReadMatrix(filename);
    float *mean = iftAllocFloatArray(ninput_channels);
    float *stdev = iftAllocFloatArray(ninput_channels);
    sprintf(filename, "%s/model/conv%d", model_dir, l+1);
    iftReadMeanStdev(filename, mean, stdev, ninput_channels);
    
    if(DEBUG){
        printf("Normalization\n");
        fflush(stdout);
    }
    /* Normalization (because there is no bias)*/
    for(int i = 0; i < input->n; i++)  
      iftNormalizeFLIMGraphByZScore(input->val[i], mean, stdev);
    iftFree(mean);
    iftFree(stdev);

    if(DEBUG){
        printf("Convolution\n");
        fflush(stdout);
    }
    /* convolution */
    int kernel_size = layer.kernel_size[0] * layer.kernel_size[1];
    iftMImageArray *activ = convLayerBatch(input, kernelbank, kernel_size);
    iftDestroyMatrix(&kernelbank);

    if(DEBUG){
        printf("Activation\n");
        fflush(stdout);
    }
    /* activation */
    iftMImageArray *activ_relu = iftMReLUArray(activ, NULL, layer.relu);
    if (activ_relu != activ)
        iftDestroyMImageArray(&activ);
    
    if(DEBUG){
        printf("Pooling\n");
        fflush(stdout);
    }
    /* pooling */
    poolingArray(activ_relu, input, layer.pool_type, layer.pool_size, 1);
    iftDestroyMImageArray(&activ_relu);
}

void decodeLayer(iftFLIMGraph *input, 
                char *model_dir, char *basename, 
                int nLayers)
{
    char filename[200];
    int Imax;
    
    sprintf(filename, "%s/labels0/%s.pgm", model_dir, basename);
    
    iftImage *label_img = iftReadImageByExt(filename);
    iftMImage *mimg = iftFLIMGraphToMImage(input, label_img);
    iftDestroyImage(&label_img);
    
    float scale[] = {1.0, 1.0, 1.0};

    /* Estimate one salience map per object (class) */
    iftFImage *salie = NULL;
    salie = TriStateAdaptativeDecoder(mimg);

    /* Interpolate and normalize the salience maps */
    iftFImage *smap_temp = NULL;
    if (iftIs3DMImage(mimg))
    {
      smap_temp = iftFInterp(salie, scale[0], scale[1], scale[2]);
      Imax = 4095;
    }
    else
    {
      smap_temp = iftFInterp2D(salie, scale[0], scale[1]);
      Imax = 255;
    }

    iftImage *smap = iftFImageToImage(smap_temp, Imax);
    iftDestroyFImage(&smap_temp);

    char saliency_dir[200];
    sprintf(saliency_dir, "%s/saliency%d/label%d", model_dir, nLayers, 1);
    if (!iftDirExists(saliency_dir))
      iftMakeDir(saliency_dir);

    /* save the salience map */
    if (iftIs3DMImage(mimg))
    {
      sprintf(filename, "%s/%s_layer%d.nii.gz", saliency_dir, basename, nLayers);
    }
    else
    {
      sprintf(filename, "%s/%s_layer%d.png", saliency_dir, basename, nLayers);
    }

    iftWriteImageByExt(smap, filename);
    iftDestroyImage(&smap);
    iftDestroyFImage(&salie);
    iftDestroyMImage(&mimg);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 7)
        iftError("Usage: iftGFLIMExtract <P1> <P2> <P3> <P4> <P5> <P6> \n"
                 "[1] Images directory \n"
                 "[2] Model directory \n"
                 "[3] Architecture file \n"
                 "[4] Number of layers \n"
                 "[5] Initial seeds \n"
                 "[6] Number of superpixels \n"
                 //"[7] Batch size \n"
                 ,
                 "main");

    tstart = iftTic();

    char *images_dir = argv[1];
    char *model_dir = argv[2];
    iftFLIMArch *arch = iftReadFLIMArch(argv[3]);
    int nLayers = atoi(argv[4]);
    int initial_seeds = atoi(argv[5]);
    int nsuperpixels = atoi(argv[6]);

    if (nLayers < 1)
    iftError("Invalid number of layers: %d", "main", nLayers);

    if(nLayers > arch->nlayers)
      iftError("Invalid number of layers: %d", "main", nLayers);

    if(!iftDirExists(images_dir))
        iftError("Images directory does not exist", "main");
    
    if(!iftDirExists(model_dir))
        iftError("Model directory does not exist", "main");

    if(initial_seeds < 1)
        iftError("Invalid number of initial seeds: %d", "main", initial_seeds);
    
    if(nsuperpixels < 1)
        iftError("Invalid number of superpixels: %d", "main", nsuperpixels);
    
    if(nsuperpixels > initial_seeds)
        iftError("Number of superpixels (%d) must be greater than the number of initial seeds (%d)", "main", nsuperpixels, initial_seeds);

    iftFileSet *fs_imgs = iftLoadFileSetFromDir(images_dir, 1);
    int ngraphs = fs_imgs->n;
    int batchsize = 0, nbatches = 0;
    int nremaining_graphs = fs_imgs->n;
    int ninput_channels = 3;
    int max_pixels = 0;
    int max_edges = 0;

    if (fs_imgs->n < 1) iftError("Image list is empty", __func__);
    
    /* Verify if all graphs have the same dimension for batch processing */

    if(DEBUG){
        printf("Checking the maximum number of edges\n");
    }

    // Ensure that each thread accesses a different cache line (https://wgropp.cs.illinois.edu/courses/cs598-s15/lectures/lecture18.pdf)
    typedef struct{ int edges; int pixels; char pad[128]; } tvals;

    tvals maxVals[omp_get_max_threads()];

#pragma omp parallel shared(maxVals)
    {
        int id = omp_get_thread_num();
        maxVals[id].edges = 0;
        maxVals[id].pixels = 0;

        for (int i = 0; i < fs_imgs->n; i++)
        {
            iftImage *input_image = iftReadImageByExt(fs_imgs->files[i]->path);
            maxVals[id].edges = iftMax(maxVals[id].edges, (input_image->xsize - 2) * (input_image->ysize - 2) * 8 + (input_image->ysize + input_image->xsize - 4) * 10 + 12);
            maxVals[id].pixels = iftMax(maxVals[id].pixels, input_image->n);
            iftDestroyImage(&input_image);
        }

#pragma omp flush(maxVals)
#pragma omp master
        {
            int nt = omp_get_num_threads();
            max_edges = maxVals[0].edges;
            max_pixels = maxVals[0].pixels;
            for (int i = 1; i < nt; i++)
            {
                if (maxVals[i].edges > max_edges)
                    max_edges = maxVals[i].edges;
                if (maxVals[i].pixels > max_pixels)
                    max_pixels = maxVals[i].pixels;
            }
        }
    }

    max_edges -= (max_pixels - nsuperpixels);
    if(max_edges > nsuperpixels*(nsuperpixels-1)/2) max_edges = nsuperpixels*(nsuperpixels-1)/2;

    if(DEBUG){
        printf("max_edges: %d\n", max_edges);
        fflush(stdout);
    }

    /* Select batch size for GPU/CPU processing */
    batchsize = iftFLIMGraphBatchSizeCPU(arch, nsuperpixels, ninput_channels, max_edges, nLayers);
    batchsize = iftMin(batchsize, ngraphs);
    nbatches = ngraphs / batchsize;
    nremaining_graphs = nremaining_graphs - nbatches * batchsize;

    for (int batch = 0; batch < nbatches; batch++)
    {
        int first = batch * batchsize, last = first + batchsize - 1;
        printf("Processing batch %d of %d batches\n", batch + 1, nbatches);
        fflush(stdout);

        iftFLIMGraphArray *input = NULL;
        int ngraphs_batch = last - first + 1;
        char **basename = (char **)calloc(ngraphs_batch, sizeof(char *));
        char filename[200];

        /* Create graphs */
        if(DEBUG){
          printf("Creating graphs\n");
          fflush(stdout);
        }
        sprintf(filename, "%s/labels0", model_dir);
        if(!iftDirExists(filename))
            iftMakeDir(filename);
        input = ImageToFLIMGraphBatch(fs_imgs, filename, first, last, initial_seeds, nsuperpixels, basename);

        /* For each layer do */
        for (int l = 0; l < nLayers; l++)
        {
            if(DEBUG){
              printf("Encoding layer %d\n", l+1);
              fflush(stdout);
            }
            encodeLayerBatch(input, arch->layer[l], l, model_dir, first, last, basename);
            
            /*char tmp[255], out[255], labels[255];
            sprintf(tmp, "%s/layer%d", model_dir, l + 1);
            sprintf(out, "%s/activation/val%d", model_dir, l + 1);
            iftMakeDir(out);
            sprintf(labels, "%s/labels%d", model_dir, l + 1);
            iftWriteActivationsImg(tmp, out, "pgm", labels);*/

        }

        if(DEBUG){
            printf("Dencoding\n");
            fflush(stdout);
        }

        // decode
        #pragma omp parallel for
        for (int k = 0; k < ngraphs_batch; k++)
        {
            decodeLayer(input->val[k], model_dir, basename[k], nLayers);
            iftDestroyFLIMGraph(&input->val[k]);
            iftFree(basename[k]);
        }
        iftFree(input);
        iftFree(basename);
    }

    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last)
    {
        printf("Processing remaining %d images\n", last - first + 1);
        fflush(stdout);

        iftFLIMGraphArray *input = NULL;
        int ngraphs_batch = last - first + 1;
        char **basename = (char **)calloc(ngraphs_batch, sizeof(char *));
        char filename[200];

        /* Create graphs */
        sprintf(filename, "%s/labels0", model_dir);
        if(!iftDirExists(filename))
            iftMakeDir(filename);
        input = ImageToFLIMGraphBatch(fs_imgs, filename, first, last, initial_seeds, nsuperpixels, basename);

        /* For each layer do */
        for (int l = 0; l < nLayers; l++)
        {
            encodeLayerBatch(input, arch->layer[l], l, model_dir, first, last, basename);
            
            /*char tmp[255], out[255], labels[255];
            sprintf(tmp, "%s/layer%d", model_dir, l + 1);
            sprintf(out, "%s/activation/val%d", model_dir, l + 1);
            iftMakeDir(out);
            sprintf(labels, "%s/labels%d", model_dir, l + 1);
            iftWriteActivationsImg(tmp, out, "pgm", labels);*/
        }

        // decode
      #pragma omp parallel for
        for (int k = 0; k < ngraphs_batch; k++)
        {
            decodeLayer(input->val[k], model_dir, basename[k], nLayers);
            iftDestroyFLIMGraph(&input->val[k]);
            iftFree(basename[k]);
        }
        iftFree(input);
        iftFree(basename);
    }

    iftDestroyFileSet(&fs_imgs);
    iftDestroyFLIMArch(&arch);

    printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
    return 0;
}
