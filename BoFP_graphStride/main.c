#include "ift.h"
#include <time.h>
#include <omp.h>
#include "ift/GFLIM/iftHGFLIM.h"
// cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug ..

typedef struct FLIMArgs
{
   char *operation, *method;
   char *model_dir, *arch_dir;
   char *seeds_dir;
   char *orig_dir;
   char *labels_dir, *mimg_dir;
   char *layers;
   float initial_seeds, final_spx;
   char *kernels_list;
   bool write_img;
   float alpha, alpha_incr;
   int niters;
} FLIMArgs;

FLIMArgs *initFLIMArgs()
{
#ifdef IFT_DEBUG
   printf("function %s \n", __func__);
#endif

   FLIMArgs *args = (FLIMArgs *)calloc(1, sizeof(FLIMArgs));
   args->model_dir = NULL;
   args->arch_dir = NULL;
   args->seeds_dir = NULL;
   args->orig_dir = NULL;
   args->labels_dir = NULL;
   args->mimg_dir = NULL;
   args->layers = NULL;
   args->initial_seeds = -1.0;
   args->final_spx = -1.0;
   args->kernels_list = NULL;
   args->write_img = 0;
   args->alpha = -1.0;
   args->alpha_incr = -1.0;
   args->niters = 0;
   return args;
}

void printArgs(FLIMArgs *args)
{
   printf("Args: \n");
   printf("\t Operation: %s \n", args->operation);
   printf("\t Method: %s \n", args->method);
   printf("\t Model dir: %s \n", args->model_dir);
   printf("\t Arch dir: %s \n", args->arch_dir);
   printf("\t Seeds dir: %s \n", args->seeds_dir);
   printf("\t Orig dir: %s \n", args->orig_dir);
   printf("\t Labels dir: %s \n", args->labels_dir);
   printf("\t Mimg dir: %s \n", args->mimg_dir);
   printf("\t Layers: %s \n", args->layers);
   printf("\t Initial seeds: %f \n", args->initial_seeds);
   printf("\t Final spx: %f \n", args->final_spx);
   printf("\t Kernels list: %s \n", args->kernels_list);
   printf("\t Write img: %d \n", args->write_img);
   printf("\t Alpha: %f \n", args->alpha);
   printf("\t Alpha incr: %f \n", args->alpha_incr);
   printf("\t niters: %d \n", args->niters);
}

void DestroyFLIMArgs(FLIMArgs **args_ptr)
{
#ifdef IFT_DEBUG
   printf("function %s \n", __func__);
#endif

   FLIMArgs *args = *args_ptr;
   iftFree(args->operation);
   iftFree(args->method);
   iftFree(args->model_dir);
   iftFree(args->arch_dir);
   iftFree(args->seeds_dir);
   iftFree(args->orig_dir);
   iftFree(args->labels_dir);
   iftFree(args->mimg_dir);
   iftFree(args->layers);
   iftFree(args->kernels_list);
   iftFree(args);
}

void usage()
{
   printf(" Usage: \n");
   printf("       ./GFLIM --op <operation> <args>");
   printf("    Operation : Methods (--method) : Args \n");
   printf("      --train : flim               : --in --arch --seeds --out --layers                                         [--write_img]\n");
   printf("              : gflim              : --in --arch --seeds --out --layers --n0 --nf                               [--write_img]\n");
   printf("              : hgflim             : --in --arch --seeds --out                    --alpha --alpha_incr          [--write_img]\n");
   printf("    --extract : flim               : --in --arch         --out --layers                                         [--write_img]\n");
   printf("              : gflim              : --in --arch         --out --layers --n0 --nf                               [--write_img]\n");
   printf("              : hgflim             : --in --arch         --out --layers --n0 --nf --alpha --alpha_incr --niters [--write_img]\n");
   printf("     --select :                    : --in                --out --layers \n");
   printf(" --activation : flim               : --in --out \n");
   printf("              : gflim, hgflim      : --in --out --labels \n");
   printf("      --seeds :                    : --seeds --out --labels \n");
   printf(" --grid_graph :                    : --in --labels --out \n");
   printf(" --disf_graph :                    : --in --labels --out --n0 --nf \n");
   printf("--sicle_graph :                    : --in --labels --out --n0 --nf --alpha --niters \n");
   printf(" --graph2mimg :                    : --in --labels --out \n");
   printf("     --decode : flim               : --in --mimg   --out \n");
   printf("     --decode : gflim, hgflim      : --in --labels --out \n");
   printf(" Operations description: \n");
   printf(" --disf_graph : Compute graphs of the images in 'orig_dir' using DISF, \n");
   printf("                save their pixel-superpixel maps in 'labels_dir',\n");
   printf("                and save the graphs in 'graph_dir'\n");
   printf(" --activation : Convert the .mimg or .json files in 'source_dir' to activations images, \n");
   printf("                writting them in destination directory (destination_dir). \n");
   printf("                For .json source files, a directory with pixel-superpixel labels map (labels_dir) is required.\n");
   printf("      --seeds : Convert image seeds files in 'seeds_dir' directory to graph seeds files in 'output_dir' directory.\n");
   printf("      --train : Train a FLIM or FLIMGraph model and write the output in 'model_dir'. \n");
   printf("    --extract : Extract feature from a trained FLIM or FLIMGraph model. \n");
   printf("     --select : Select kernels from a trained FLIM or FLIMGraph model. \n");
   printf("     --decode : Use an adaptative decoder to provide a unique saliency per image. \n");
   printf(" Args description: \n");
   printf("           --in : Input directory with images or graphs.\n");
   printf("       --labels : Directory with pixel-superpixel maps.\n");
   printf("           --n0 : Number of initial seeds for DISF computation.\n");
   printf("           --nf : Number of final superpixels for DISF/SICLE computation.\n");
   printf("        --seeds : Folder with seeds files.\n");
   printf("          --out : Folder for trained model.\n");
   printf("       --layers : List of layers to train. \n");
   printf("                  Example for 3 layers: 1,2,3. Example for the 2nd layer: 2\n");
   printf(" --kernels_list : A list of kernels to select. Example: 1,2,6,7,10\n");
   printf("    --write_img : {0,1} A boolean value to decide whether the activation files (just for visuzalization) \n");
   printf("                  will be written on the 'model_dir'/img_activations folder.\n");
}

int *getTrainLayersParam(int nLayers, char *param)
{
#ifdef IFT_DEBUG
   printf("function %s \n", __func__);
#endif

   int *train_layers = (int *)calloc(nLayers, sizeof(int));
   char *ptr = strtok(param, ",");
   while (ptr != NULL)
   {
      int i = 0, layer = 0;
      if (ptr[i] < 48 || ptr[i] > 57)
         break;
      while (ptr[i] != '\0')
      {
         if (ptr[i] < 48 || ptr[i] > 57)
            break;
         layer = layer * 10 + (ptr[i] - '0');
         i++;
      }
      if (layer > 0 && layer <= nLayers)
         train_layers[layer - 1] = 1;
      ptr = strtok(NULL, ",");
   }
   return train_layers;
}

char *copyString(char *toCopy)
{
   char *copy = (char *)iftAlloc(strlen(toCopy) + 1, sizeof(char));
   sprintf(copy, "%s", toCopy);
   return copy;
}

char *getValArg(char *arg, int argc, char *argv[], bool allocCopy){
   int i = 1;
   while(i < argc){
      if(strcmp(arg, argv[i]) == 0){
         if(allocCopy) return copyString(argv[i+1]);
         else return argv[i+1];
      }
      i++;
   }
   return NULL;
}

void getInputDirectory(int argc, char *argv[], FLIMArgs *args){
   iftDir *dir = NULL;
   args->orig_dir = getValArg("--in", argc, argv, true);
   if(args->orig_dir == NULL || !iftDirExists(args->orig_dir)) {
      //DestroyFLIMArgs(&args);
      iftError("Input directory missing or does not exist: %s", __func__, args->orig_dir == NULL ? "NULL" : args->orig_dir);
   }
   dir = iftLoadDir(args->orig_dir, 1);
   if(dir->nfiles < 1) {
      //DestroyFLIMArgs(&args);
      iftDestroyDir(&dir);
      iftError("No files found in input directory: %s", __func__, args->orig_dir);
   }
   iftDestroyDir(&dir);
}

void getArchFile(int argc, char *argv[], FLIMArgs *args){
   args->arch_dir = getValArg("--arch", argc, argv, true);
   if(args->arch_dir == NULL || 
      !iftFileExists(args->arch_dir)) 
   {
      //DestroyFLIMArgs(&args);
      iftError("Achitectory file (--arch) missing, invalid format (not json) or does not exist: %s", __func__, args->arch_dir == NULL ? "NULL" : args->arch_dir);
   }
}

void getSeedsDir(int argc, char *argv[], FLIMArgs *args){
   iftDir *dir = NULL;
   args->seeds_dir = getValArg("--seeds", argc, argv, true);
   if(args->seeds_dir == NULL || !iftDirExists(args->seeds_dir)) {
      //DestroyFLIMArgs(&args);
      iftError("Seeds directory (--seeds) missing or does not exist: %s", __func__, args->seeds_dir == NULL ? "NULL" : args->seeds_dir);
   }
   dir = iftLoadDir(args->seeds_dir, 1);
   if(dir->nfiles < 1) {
      //DestroyFLIMArgs(&args);
      iftDestroyDir(&dir);
      iftError("No seeds files found in directory: %s", __func__, args->seeds_dir);
   }
   iftDestroyDir(&dir);
}

void geOutputDir(int argc, char *argv[], FLIMArgs *args){
   args->model_dir = getValArg("--out", argc, argv, true);
   if(args->model_dir == NULL) {
      //DestroyFLIMArgs(&args);
      iftError("Output directory (--out) is missing: %s", __func__, args->model_dir == NULL ? "NULL" : args->model_dir);
   }
}

void getMethodName(int argc, char *argv[], FLIMArgs *args){
   args->method = getValArg("--method", argc, argv, true);
   if(args->method == NULL || 
      (strcmp(args->method, "flim") != 0 && 
      strcmp(args->method, "gflim") != 0 && 
      strcmp(args->method, "hgflim") != 0)) 
   {
      //DestroyFLIMArgs(&args);
      iftError("Method name (--method) missing, invalid or does not exist: %s", __func__, args->method == NULL ? "NULL" : args->method);
   }
}

void getLayers(int argc, char *argv[], FLIMArgs *args){
   args->layers = getValArg("--layers", argc, argv, true);
   if(args->layers == NULL) {
      //DestroyFLIMArgs(&args);
      iftError("Layers (--layers) missing", __func__);
   }
}

void getInitialSeeds(int argc, char *argv[], FLIMArgs *args, bool has_final_superpixels){
   char *tmp = getValArg("--n0", argc, argv, false);
   if(tmp == NULL || atoi(tmp) < 2 || (has_final_superpixels && atoi(tmp) <= args->final_spx)) {
      //DestroyFLIMArgs(&args);
      iftError("Number of initial seeds (--n0) missing or invalid: %s", __func__, tmp == NULL ? "NULL" : tmp);
   }
   args->initial_seeds = atoi(tmp);
   tmp = NULL;
}

void getFinalSuperpixels(int argc, char *argv[], FLIMArgs *args, bool has_initial_seeds){
   char *tmp = getValArg("--nf", argc, argv, false);
   if(tmp == NULL || atoi(tmp) < 2 || (has_initial_seeds && atoi(tmp) >= args->initial_seeds)) {
      //DestroyFLIMArgs(&args);
      iftError("Number of final superpixels (--nf) missing or invalid: %s", __func__, tmp == NULL ? "NULL" : tmp);
   }
   args->final_spx = atoi(tmp);
   tmp = NULL;
}

void getAlpha(int argc, char *argv[], FLIMArgs *args){
   char *tmp = getValArg("--alpha", argc, argv, false);
   if(tmp == NULL) args->alpha = 0;
   else args->alpha = atof(tmp);
   tmp = NULL;
}

void getAlphaIncr(int argc, char *argv[], FLIMArgs *args){
   char *tmp = getValArg("--alpha_incr", argc, argv, false);
   if(tmp == NULL) args->alpha_incr = 0;
   else args->alpha_incr = atof(tmp);
   tmp = NULL;
}

void getWriteImg(int argc, char *argv[], FLIMArgs *args){
   char *tmp = getValArg("--write_img", argc, argv, false);
   if(tmp != NULL) args->write_img = atoi(tmp);
   tmp = NULL;
}

void getMIMGArgs(int argc, char *argv[], FLIMArgs *args){
   iftDir *dir = NULL;
   args->mimg_dir = getValArg("--mimg", argc, argv, true);
   if(args->mimg_dir == NULL || !iftDirExists(args->mimg_dir)) {
      //DestroyFLIMArgs(&args);
      iftError("MIMG directory missing or does not exist: %s", __func__, args->mimg_dir == NULL ? "NULL" : args->mimg_dir);
   }
   dir = iftLoadDir(args->mimg_dir, 1);
   if(dir->nfiles < 1) {
      //DestroyFLIMArgs(&args);
      iftDestroyDir(&dir);
      iftError("No files found in images directory: %s", __func__, args->mimg_dir);
   }
   iftDestroyDir(&dir);
}

void getLabelsDir(int argc, char *argv[], FLIMArgs *args){
   args->labels_dir = getValArg("--labels", argc, argv, true);
   if(args->labels_dir == NULL || !iftDirExists(args->labels_dir)) {
      //DestroyFLIMArgs(&args);
      iftError("Labels directory (--labels) missing or does not exist: %s", __func__, args->labels_dir == NULL ? "NULL" : args->labels_dir);
   }
   iftDir *dir = NULL;
   dir = iftLoadDir(args->labels_dir, 1);
   if(dir->nfiles < 1) {
      //DestroyFLIMArgs(&args);
      iftDestroyDir(&dir);
      iftError("No files found in labels directory (--labels): %s", __func__, args->labels_dir); 
   }
   iftDestroyDir(&dir);
}

void getNumIterationsArgs(int argc, char *argv[], FLIMArgs *args){
   char *tmp = getValArg("--niters", argc, argv, false);
   if(tmp == NULL || atoi(tmp) < 1) {
      //DestroyFLIMArgs(&args);
      iftError("Number of iterations (--niters) missing or invalid: %s", __func__, tmp == NULL ? "NULL" : tmp);
   }
   args->niters = atoi(tmp);
   tmp = NULL;
}

void getTrainArgs(int argc, char *argv[], FLIMArgs *args){
   getMethodName(argc, argv, args);
   getInputDirectory(argc, argv, args);
   getArchFile(argc, argv, args);
   getSeedsDir(argc, argv, args);
   geOutputDir(argc, argv, args);
   
   if(strcmp(args->method, "flim") == 0 || strcmp(args->method, "gflim") == 0){
      getLayers(argc, argv, args);
   }

   if(strcmp(args->method, "gflim") == 0){
      getInitialSeeds(argc, argv, args, false);
      getFinalSuperpixels(argc, argv, args, true);
   }

   if(strcmp(args->method, "hgflim") == 0){
      getAlpha(argc, argv, args);
      getAlphaIncr(argc, argv, args);
   }

   getWriteImg(argc, argv, args);
}

void getExtractArgs(int argc, char *argv[], FLIMArgs *args){
   getMethodName(argc, argv, args);
   getInputDirectory(argc, argv, args);
   getArchFile(argc, argv, args);
   geOutputDir(argc, argv, args);
   
   if(strcmp(args->method, "flim") == 0 || strcmp(args->method, "gflim") == 0){
      getLayers(argc, argv, args);
   }

   if(strcmp(args->method, "gflim") == 0){
      getInitialSeeds(argc, argv, args, false);
      getFinalSuperpixels(argc, argv, args, true);
   }

   if(strcmp(args->method, "hgflim") == 0){
      getAlpha(argc, argv, args);
      getAlphaIncr(argc, argv, args);
   }

   getWriteImg(argc, argv, args);
}

void getSelectKernelsArgs(int argc, char *argv[], FLIMArgs *args){
   getInputDirectory(argc, argv, args);
   geOutputDir(argc, argv, args);
   getLayers(argc, argv, args);
   args->kernels_list = getValArg("--kernels_list", argc, argv, true);
}

void getActivationsArgs(int argc, char *argv[], FLIMArgs *args){
   getMethodName(argc, argv, args);
   getInputDirectory(argc, argv, args);
   geOutputDir(argc, argv, args);
   if(strcmp(args->method, "gflim") == 0 || strcmp(args->method, "hgflim") == 0)
      getLabelsDir(argc, argv, args);
}

void getSeedsArgs(int argc, char *argv[], FLIMArgs *args){
   getSeedsDir(argc, argv, args);
   getLabelsDir(argc, argv, args);
   geOutputDir(argc, argv, args);
}

void getDISFGraphArgs(int argc, char *argv[], FLIMArgs *args){
   getInputDirectory(argc, argv, args);
   getLabelsDir(argc, argv, args);
   geOutputDir(argc, argv, args);
   getInitialSeeds(argc, argv, args, false);
   getFinalSuperpixels(argc, argv, args, true);
}

void getGridGraphArgs(int argc, char *argv[], FLIMArgs *args){
   getInputDirectory(argc, argv, args);
   getLabelsDir(argc, argv, args);
   geOutputDir(argc, argv, args);
   getFinalSuperpixels(argc, argv, args, false);
}

void getSICLEGraphArgs(int argc, char *argv[], FLIMArgs *args){
   getInputDirectory(argc, argv, args);
   geOutputDir(argc, argv, args);
   getLabelsDir(argc, argv, args);
   getInitialSeeds(argc, argv, args, false);
   getFinalSuperpixels(argc, argv, args, true);
   getAlpha(argc, argv, args);
   getNumIterationsArgs(argc, argv, args);
}

void getGraph2MimgArgs(int argc, char *argv[], FLIMArgs *args){
   getInputDirectory(argc, argv, args);
   getLabelsDir(argc, argv, args);
   geOutputDir(argc, argv, args);
}

void getDecoderArgs(int argc, char *argv[], FLIMArgs *args){
   getMethodName(argc, argv, args);
   getInputDirectory(argc, argv, args);
   geOutputDir(argc, argv, args);
   if(strcmp(args->method, "gflim") == 0 ||
      strcmp(args->method, "hgflim") == 0)
      getLabelsDir(argc, argv, args);
   else getMIMGArgs(argc, argv, args);
   geOutputDir(argc, argv, args);
}

void prepareFLIM(char *model_dir, char *arch_dir,
                 char *seeds_dir, char *orig_dir,
                 char *graph_seeds, int layer)
{
#ifdef IFT_DEBUG
   printf("function %s \n", __func__);
   printf("\t model_dir: %s \n", model_dir);
   printf("\t arch_dir: %s \n", arch_dir);
   printf("\t seeds_dir: %s \n", seeds_dir);
   printf("\t orig_dir: %s \n", orig_dir);
   printf("\t graph_seeds: %s \n", graph_seeds);
   printf("\t layer: %d \n", layer);
#endif

   /*if (layer == 0) {
      iftMakeDir(model_dir);
      iftFLIMArch *arch = NULL;
      arch = iftReadFLIMArch(arch_dir);
      char filename[200];
      sprintf(filename, "%s/arch.json", model_dir);
      //iftWriteFLIMArch(arch, filename);
      iftDestroyFLIMArch(&arch);
   } else {*/
      char tmp[255];
      iftFileSet *fs_seeds = NULL;

      if (seeds_dir != NULL && iftDirExists(seeds_dir))
         fs_seeds = iftLoadFileSetFromDirBySuffix(seeds_dir, "-seeds.txt", 1);

      if (fs_seeds != NULL && fs_seeds->n > 0)
      {
         sprintf(tmp, "%s/markers%d", model_dir, layer);
         if (!iftDirExists(tmp))
            iftMakeDir(tmp);

         iftFileSet *fs_orig = iftLoadFileSetFromDirOrCSV(orig_dir, 1, 1);
         if (fs_orig->n == 0)
            iftError("No images found", "prepareFLIM");

         char ext[10];
         sprintf(ext, "%s", iftFileExt(fs_orig->files[0]->path));
         iftDestroyFileSet(&fs_orig);
         for (int i = 0; i < fs_seeds->n; i++)
         {
            iftImage *input;
            char *basename = NULL;
            char filename[200];
            iftLabeledSet *S, *S_inv = NULL;
            basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
            sprintf(filename, "%s/%s%s", orig_dir, basename, ext);
            input = iftReadImageByExt(filename);

            S = iftReadSeeds(input, fs_seeds->files[i]->path);
            // invert labeled set to produce the same kernels
            iftLabeledSet *s_tmp = S;
            while (s_tmp != NULL)
            {
               iftInsertLabeledSetMarkerAndHandicap(&S_inv, s_tmp->elem, s_tmp->label, s_tmp->marker, s_tmp->handicap);
               s_tmp = s_tmp->next;
            }

            sprintf(filename, "%s/markers%d/%s-seeds.txt", model_dir, layer, basename);
            iftWriteSeeds(S_inv, input, filename);
            iftDestroyLabeledSet(&S_inv);
            iftDestroyLabeledSet(&S);
            iftDestroyImage(&input);
            iftFree(basename);
         }
      }

      if (fs_seeds != NULL)
         iftDestroyFileSet(&fs_seeds);

      if (graph_seeds != NULL && iftDirExists(graph_seeds))
         fs_seeds = iftLoadFileSetFromDirBySuffix(seeds_dir, "-seeds_graph.txt", 1);

      if (fs_seeds != NULL && fs_seeds->n > 0)
      {
         sprintf(tmp, "%s/markers_graph%d", model_dir, layer);
         if (!iftDirExists(tmp))
            iftMakeDir(tmp);

         for (int i = 0; i < fs_seeds->n; i++)
         {
            char *basename = NULL;
            char filename[200];
            iftLabeledSet *S, *S_inv = NULL;
            basename = iftFilename(fs_seeds->files[i]->path, "-seeds_graph.txt");

            S = iftReadSeedsGraph(fs_seeds->files[i]->path);
            // invert labeled set to produce the same kernels
            iftLabeledSet *s_tmp = S;
            while (s_tmp != NULL)
            {
               iftInsertLabeledSetMarkerAndHandicap(&S_inv, s_tmp->elem, s_tmp->label, s_tmp->marker, s_tmp->handicap);
               s_tmp = s_tmp->next;
            }
            sprintf(filename, "%s/markers%d/%s-seeds_graph.txt", model_dir, layer, basename);
            iftWriteSeedsGraph(S, filename);
            iftDestroyLabeledSet(&S);
            iftDestroyLabeledSet(&S_inv);
            iftFree(basename);
         }
      }
      if (fs_seeds != NULL)
         iftDestroyFileSet(&fs_seeds);
   //}
}

void main_train(FLIMArgs *args)
{
#ifdef IFT_DEBUG
   printf("function %s \n", __func__);
#endif

   char graphs_dir[255], *seeds_dir = NULL, labels_dir[255];
   int nLayers = 0;
   int *train_layers = NULL;

   if(iftDirExists(args->model_dir)){
      iftWarning("Deleting the existing model directory.", "main_train");
      iftRemoveDir(args->model_dir);
   }
   iftMakeDir(args->model_dir);

   // get layers (flim and gflim only) and write arch in model dir
   if (strcmp(args->method, "gflim") == 0 || strcmp(args->method, "flim") == 0){
      iftFLIMArch *tmp_arch = iftReadFLIMArch(args->arch_dir);
      nLayers = tmp_arch->nlayers;
      char filename[200];
      sprintf(filename, "%s/arch.json", args->model_dir);
      iftWriteFLIMArch(tmp_arch, filename);
      iftDestroyFLIMArch(&tmp_arch);
      train_layers = getTrainLayersParam(nLayers, args->layers);
   }

   if (strcmp(args->method, "gflim") == 0)
   {
      seeds_dir = iftAlloc(255, sizeof(char));
      sprintf(labels_dir, "%s/labels", args->model_dir);
      iftMakeDir(labels_dir);
      sprintf(graphs_dir, "%s/graphs", args->model_dir);
      iftMakeDir(graphs_dir);
      sprintf(seeds_dir, "%s/markers_graph1", args->model_dir);
      if (!iftDirExists(seeds_dir))
         iftMakeDir(seeds_dir);

      // convert images to graph (write in tmp directory)
      iftImageToFLIMGraphDISFDir(args->orig_dir, labels_dir, graphs_dir, (int)args->initial_seeds, (int)args->final_spx);

      // convert seeds pixel to seeds graph and write in "model_dir/markers_graph1"
      iftSeedsPixelToGraphDir(labels_dir, args->seeds_dir, seeds_dir);
   }

   if (strcmp(args->method, "flim") == 0)
   {
      for (int i = 0; i < nLayers; i++)
      {
         if (train_layers[i])
         {
            prepareFLIM(args->model_dir, args->arch_dir, args->seeds_dir, args->orig_dir, seeds_dir, i + 1);
            iftFLIMTrain(i + 1, args->model_dir, args->orig_dir, args->seeds_dir);
         }
      }
      char tmp[255], tmp1[255], out[255];
      sprintf(tmp, "%s/layer%d", args->model_dir, nLayers);
      sprintf(tmp1, "%s/layer1", args->model_dir);
      sprintf(out, "%s/saliency_train%d", args->model_dir, nLayers);
      decode(tmp, tmp1, out);
      sprintf(tmp, "%s/tmp", args->model_dir);
      iftRemoveDir(tmp);
   }
   else
   {
      if (strcmp(args->method, "gflim") == 0)
      {
         //printf("Training GFLIM \n");
         for (int i = 0; i < nLayers; i++)
         {
            if (train_layers[i])
            {
               //printf("Training layer %d \n", i + 1);
               prepareFLIM(args->model_dir, args->arch_dir, args->seeds_dir, args->orig_dir, seeds_dir, i + 1);
               //printf("call FLIMGraphTrain %d \n", i + 1);
               FLIMGraphTrain(i + 1, args->model_dir, graphs_dir, seeds_dir);
               
               char tmp[255], out[255];
               sprintf(tmp, "%s/layer%d", args->model_dir, i + 1);
               sprintf(out, "%s/activation/train%d", args->model_dir, i + 1);
               iftMakeDir(out);
               iftWriteActivationsImg(tmp, out, "pgm", labels_dir);
            }
         }
         //printf("Decoding \n");
         char tmp[255];
         sprintf(tmp, "%s/layer%d", args->model_dir, nLayers);
         iftDecodeLayer(tmp, labels_dir, args->model_dir, nLayers, IFT_HGFLIM_TRAIN);
         iftRemoveDir(graphs_dir);
         iftRemoveDir(labels_dir);
      }
      else
      {
         if (strcmp(args->method, "hgflim") == 0)
         {
            runHGFLIM(args->orig_dir, args->model_dir, args->seeds_dir,
                        args->alpha, args->alpha_incr, args->arch_dir, IFT_HGFLIM_TRAIN);
            printf("\r THE END \\o/ \n");
         }else
               iftWarning("Method not implemented", __func__);
      }
   }
   
   if(seeds_dir != NULL) free(seeds_dir);
   if(train_layers != NULL) free(train_layers);
}

void main_extract(FLIMArgs *args)
{
#ifdef IFT_DEBUG
   printf("function %s \n", __func__);
#endif

   iftFLIMArch *arch;
   char input_list[255], feat_dir[255];
   int max_edges = 0;
   
   sprintf(input_list, "%s/input.csv", args->model_dir);
   
   if (strcmp(args->method, "flim") == 0)
   {
      arch = iftReadFLIMArch(args->arch_dir);
      sprintf(feat_dir, "%s/activation", args->model_dir);
      if (!iftDirExists(feat_dir))
         iftMakeDir(feat_dir);

      // Creating list of input files
      iftWriteCSVFiles(input_list, args->orig_dir);
      iftFLIMExtractFeatures(args->orig_dir, input_list, arch, args->model_dir,
                             feat_dir, NULL, -1);

      char tmp[255], out[255];
      sprintf(tmp, "%s/layer1", args->model_dir);
      sprintf(out, "%s/saliency_%s%d", args->model_dir, args->operation, arch->nlayers);
      decode(feat_dir, feat_dir, out);
      iftRemoveDir(feat_dir);
      iftDestroyFLIMArch(&arch);
   }
   else
   {
      if (strcmp(args->method, "gflim") == 0)
      {
         char labels_dir[255], tmp[255]/*, graphs_dir[255], mimg_activation_dir[255]*/;

         arch = iftReadFLIMArch(args->arch_dir);
         //sprintf(feat_dir, "%s/activation_graph", args->model_dir);
         sprintf(feat_dir, "%s/saliency_%s%d", args->model_dir, args->operation, arch->nlayers);
         if (!iftDirExists(feat_dir))
            iftMakeDir(feat_dir);

         // create tmp folder
         sprintf(labels_dir, "%s/labels", args->model_dir);
         iftMakeDir(labels_dir);
         /*sprintf(graphs_dir, "%s/graphs", args->model_dir);
         iftMakeDir(graphs_dir);*/

         //max_edges = iftImageToFLIMGraphDISFDir(args->orig_dir, labels_dir, graphs_dir, (int)args->initial_seeds, (int)args->final_spx);

         // Creating list of input files
         iftWriteCSVFiles(input_list, args->orig_dir); // iftWriteCSVFiles(input_list, graphs_dir);

         /*iftFLIMGraphExtractFeatures(graphs_dir, input_list, arch, args->model_dir,
                                     feat_dir, NULL, -1, max_edges);*/

         iftFLIMGraphExtractFeatures_batch(args->orig_dir, input_list, arch, args->model_dir,
                                 feat_dir, labels_dir, (int)args->initial_seeds, (int)args->final_spx, -1, 0);

         /*char out[255];
         sprintf(out, "%s/saliency_%s%d", args->model_dir, args->operation, arch->nlayers);
         iftDecodeLayer(feat_dir, labels_dir, args->model_dir, arch->nlayers, strcmp(args->operation, "val") == 0 ? IFT_HGFLIM_VAL : IFT_HGFLIM_TEST);
         sprintf(tmp, "%s/tmp", args->model_dir);
         iftRemoveDir(tmp);*/

         iftRemoveDir(labels_dir);
         //iftRemoveDir(graphs_dir);
         //iftRemoveDir(feat_dir);
         iftDestroyFLIMArch(&arch);
      }
      else
      {
         if (strcmp(args->method, "hgflim") == 0)
         {
            /*extractHGFLIM(args->orig_dir, args->arch_dir,
                               args->model_dir, 1, args->write_img, args->initial_seeds, args->final_spx,
                               args->alpha, args->alpha_incr, args->niters);*/
            printf("EXTRACTING FEATURES FROM HGFLIM \n");
            runHGFLIM(args->orig_dir, args->model_dir, args->seeds_dir,
                        args->alpha, args->alpha_incr, args->arch_dir, strcmp(args->operation, "val") == 0 ? IFT_HGFLIM_VAL : IFT_HGFLIM_TEST);
            printf("\r THE END \\o/ \n");
         }else iftWarning("Method not implemented", __func__);
      }
   }
   
}

int main_GFLIM(int argc, char *argv[])
{

#ifdef IFT_DEBUG
   printf("function %s \n", __func__);
#endif

   FLIMArgs *args = initFLIMArgs();
   args->operation = getValArg("--op", argc, argv, false);
   if(args->operation == NULL) {
      usage();
      //DestroyFLIMArgs(&args);
      iftError("Operation (--op) missing", __func__);
   }

   if(strcmp(args->operation, "train") == 0){
      getTrainArgs(argc, argv, args);
      main_train(args);
   }else if (strcmp(args->operation, "val") == 0 || strcmp(args->operation, "test") == 0){
      getExtractArgs(argc, argv, args);
      main_extract(args);
   }
   else if (strcmp(args->operation, "select") == 0){
      getSelectKernelsArgs(argc, argv, args);
      char *kernels_list = (char *)malloc(255 * sizeof(char));
      sprintf(kernels_list, "%s", args->kernels_list);
      int *train_layers = getTrainLayersParam(1, args->layers);
      iftSelectKernelsManual(args->model_dir, args->orig_dir, train_layers[0], kernels_list, args->method);
      free(kernels_list);
      free(train_layers);
   }
   else if (strcmp(args->operation, "activation") == 0){
      getActivationsArgs(argc, argv, args);
      iftWriteActivationsImg(args->orig_dir, args->model_dir, "pgm", args->labels_dir);
   }else if (strcmp(args->operation, "seeds") == 0){
      getSeedsArgs(argc, argv, args);
      iftSeedsPixelToGraphDir(args->labels_dir, args->seeds_dir, args->model_dir);
   }else if (strcmp(args->operation, "disf_graph") == 0){
      getDISFGraphArgs(argc, argv, args);
      iftImageToFLIMGraphDISFDir(args->orig_dir, args->labels_dir, args->model_dir, args->initial_seeds, args->final_spx);
   }else if (strcmp(args->operation, "grid_graph") == 0){
      getGridGraphArgs(argc, argv, args);
      iftImageToGridFLIMGraphs(args->orig_dir, args->labels_dir, args->model_dir);
   }else if (strcmp(args->operation, "sicle_graph") == 0){
      getSICLEGraphArgs(argc, argv, args);
      iftSICLEArgs *sicle_args = iftCreateSICLEArgs(args->initial_seeds, args->final_spx, args->niters, args->alpha);
      reduceFLIMGraphSICLEDir(args->orig_dir, NULL, args->model_dir, args->labels_dir, NULL, sicle_args);
      iftFree(args);
   }else if (strcmp(args->operation, "graph2mimg") == 0){
      getGraph2MimgArgs(argc, argv, args);
      iftFLIMGraphToMIMG(args->orig_dir, args->labels_dir, args->model_dir);
   }else if (strcmp(args->operation, "decode") == 0){
      getDecoderArgs(argc, argv, args);
      if(strcmp(args->method, "flim") == 0)
         decode(args->orig_dir, args->mimg_dir, args->model_dir);
      else 
         iftDecode(args->orig_dir, args->labels_dir, args->model_dir);
   }else usage();

#ifdef IFT_DEBUG
   printf("end main_GFLIM \n");
#endif

   return 0;
}

void texture_test(){
   iftImage *img = iftCreateColorImage(5, 5, 1, 3);
   
   /*for(int i = 0; i < img->n; i++){
      img->val[i] = 235;
      img->Cb[i] = 128;
      img->Cr[i] = 128;
   }

   for(int i = 6; i <= 8; i++){
      for(int j = 5; j < 8; j++){
         img->val[10*j-1+i] = 125;
         img->Cb[10*j-1+i] = 128;
         img->Cr[10*j-1+i] = 128;   
      }
   }

   for(int i = 3; i <= 5; i++){
      for(int j = 5; j < 8; j++){
         img->val[10*j-1+i] = 16;
         img->Cb[10*j-1+i] = 128;
         img->Cr[10*j-1+i] = 128;   
      }
   }*/

  // seed for random values
   srand(0);

  // fill the image with random values
   for(int i = 0; i < img->n; i++){
      img->val[i] = rand() % 256;
      img->Cb[i] = rand() % 256;
      img->Cr[i] = rand() % 256;
   }
   
   printf("Image: \n");
   for(int i = 0; i < 5; i++){
      for(int j=0; j < 5; j++){
         printf("%d,%d,%d \t", img->val[i*5+j], img->Cb[i*5+j], img->Cr[i*5+j]);
      }
      printf("\n");
   }
   printf("\n\n");

   iftColorSpace cspace = GRAY_CSPACE;
   iftImage *label_img_tmp = NULL, *mask = NULL;
   iftMImage *mimg = iftImageToMImage(img, LABNorm_CSPACE);
   iftAdjRel *A = iftCircular(sqrtf(2.0));
   label_img_tmp = runDISF(mimg, 10, 3, A, mask);
   
   iftMImage *gray_mimg = iftImageToMImage(img, cspace);
   iftDestroyImage(&img);

   printf("Label Image: \n");
   for(int i = 0; i < 5; i++){
      for(int j=0; j < 5; j++){
         printf("%d \t", label_img_tmp->val[i*5+j]);
      }
      printf("\n");
   }
   printf("\n");

   printf("GRAY Image: \n");
   for(int i = 0; i < 5; i++){
      for(int j=0; j < 5; j++){
         printf("%.2f \t", gray_mimg->val[i*5+j][0]);
      }
      printf("\n");
   }
   printf("\n\n");

   // print adjacency
   /*printf("Adjacency: \n");
   int pixel = 6;
   iftVoxel u = iftMGetVoxelCoord(mimg, pixel);
   for(int j = 1; j < A->n; j++){
            
      iftVoxel v = iftGetAdjacentVoxel(A,u,j);
            
      if (iftMValidVoxel(mimg, v))
      {
         int neigh = iftMGetVoxelIndex(mimg, v);
         printf("Pixel %d Adj %d \n", j, neigh);
      }
   }*/

   iftFLIMGraph *graph = iftSuperpixelImageToFLIMGraph_haralick2(3, A, mimg, gray_mimg, label_img_tmp);

   printf("Graph: \n");
   for(int i = 0; i < graph->num_nodes; i++){
      printf("Node %d: ", i);
      for(int j = 0; j < graph->num_feats; j++){
         printf("\t %.4f ", graph->nodes[i].feats[j]);
      }
      printf("\n");
   }

   iftDestroyAdjRel(&A);
   iftDestroyImage(&label_img_tmp);
   iftDestroyMImage(&mimg);
   iftDestroyMImage(&gray_mimg);
   iftDestroyFLIMGraph(&graph);
}

int main(int argc, char *argv[])
{
#ifdef IFT_DEBUG
   printf("DEBUG TRUE \n");
#endif
   main_GFLIM(argc, argv);
   //texture_test();
   
   return 0;
}