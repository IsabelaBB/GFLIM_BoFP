/**
 * \file iftGFLIMMergeScales.c
 * \brief Merge layers or scales based on a .txt file with the kernels mapping
 */

#include "ift.h"

int main(int argc, char *argv[]) {
    timer *tstart;
    
    if (argc != 5)
      iftError("Usage: iftGFLIMMergeScales <P1> <P2> <P3> <P4> \n"
		"P1: model folder\n"
		"P2: images folder \n"
		"P3: arch file (.json) \n"
		"P4: current layer \n"
		,"main");

    tstart = iftTic();

    char *model_dir = argv[1];
    char *images_folder = argv[2];
    char *arch_file = argv[3];
    int current_layer = atoi(argv[4]);
	  char filename[512];

    if(!iftDirExists(model_dir))
      iftError("Model directory does not exist: %s", "main", model_dir);
    if(!iftDirExists(images_folder))
      iftError("Images directory does not exist: %s", "main", images_folder);

    iftFileSet *fs = iftLoadFileSetFromDir(images_folder, 1);
    if(fs->n < 1)
      iftError("No files found in the images directory: %s", "main", images_folder);
    
    iftGFLIMArch *arch = iftReadGFLIMArch(arch_file);
    
    if(current_layer < 1 || current_layer > arch->nlayers)
      iftError("Invalid layer: %d", "main", current_layer);

	// compute the kernel maps for all layers
	sprintf(filename, "%s/model/kernels_map.txt", model_dir);
    
	iftGFLIMSkipConnection(arch, current_layer, filename, model_dir, fs);
	iftDestroyGFLIMArch(&arch);
    iftDestroyFileSet(&fs);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
