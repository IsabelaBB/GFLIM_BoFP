/**
 * \file iftGFLIMWriteKernelsCorrespondence.c
 * \brief Given the markers and superpixel images of two scales, it creates the correspondence between the kernels of these scales.
 * \warning The markers coordinates must be in "*-seeds_graph.txt" files.
 */


#include "ift.h"

int main(int argc, char *argv[]) {
    timer *tstart;

    if (argc != 5 && argc != 6)
      iftError("Usage: iftGFLIMMergeScales <P1> <P2> <P3> <P4> [<P5>]\n"
	        "P1: markers folder scale 0\n"
	        "P2: superpixels folder scale 0\n"
            "P3: superpixels folder scale i\n"
            "P4: save correspondence file\n"
            "P5: save markers folder (optional)\n"
	       ,"main");

    tstart = iftTic();

    char *markers_folder_0 = argv[1];
    char *spx_folder_0 = argv[2];
    char *spx_folder_i = argv[3];
    char *save_file = argv[4];
    char *save_markers_folder = NULL;
    if(argc == 6)
        save_markers_folder = argv[5];

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(markers_folder_0,"-seeds_graph.txt", 1);
    if(fs->n < 1)
      iftError("No files found in the markers directory: %s", "main", markers_folder_0);

    if (!iftDirExists(save_markers_folder))
        iftMakeDir(save_markers_folder);

    // Count the number of seed in scale 0
    int nseeds = 0;
    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, "-seeds_graph.txt");
        iftLabeledSet *S0 = iftReadSeedsGraph(fs->files[i]->path);
        nseeds += iftLabeledSetSize(S0);
        iftDestroyLabeledSet(&S0);
        iftFree(basename);
    }

    iftIntArray *kernel_map = iftCreateIntArray(nseeds);
    int last_kernel = 0;
    for (int i = 0; i < fs->n; i++)
    {
        char filename[512];
        char *basename = iftFilename(fs->files[i]->path, "-seeds_graph.txt");
        
        iftLabeledSet *S0 = iftReadSeedsGraph(fs->files[i]->path);

        // read the superpixel image
        sprintf(filename, "%s/%s%s", spx_folder_0, basename, ".pgm");
        if(!iftFileExists(filename)) iftError("Labels file %s does not exist", __func__, filename);
        iftImage *spx_img0 = iftReadImageByExt(filename);
        
        sprintf(filename, "%s/%s%s", spx_folder_i, basename, ".pgm");
        if(!iftFileExists(filename)) iftError("Labels file %s does not exist", __func__, filename);
        iftImage *spx_imgi = iftReadImageByExt(filename);
        
        int nnodes0 = iftMaximumValue(spx_img0);
        int nnodesi = iftMaximumValue(spx_imgi);
        
        // create a superpixel mapping from scale 0 to scale i
        iftIntArray *spx_map = iftCreateIntArray(nnodes0);
        for (int p = 0; p < spx_img0->n; p++) {
            spx_map->val[spx_img0->val[p] - 1] = spx_imgi->val[p] - 1;
        } 

        iftDestroyImage(&spx_img0);
        iftDestroyImage(&spx_imgi);

        // map from spxi to kernel id (from 1) to assigm the same kernel id to the same superpixel
        iftIntArray *spx_id_map = iftCreateIntArray(nnodesi); 
        iftLabeledSet *Si = NULL;
        iftLabeledSet *s = S0;
        iftIntArray *labels = iftCreateIntArray(nnodesi); 
        while (s != NULL) {
            int spxi = spx_map->val[s->elem];
            int id = spx_id_map->val[spxi];
            last_kernel++;
            if(id == 0) // not visited spx
            {
                spx_id_map->val[spxi] = last_kernel;
                id = last_kernel;
                if(save_markers_folder != NULL){
                    iftInsertLabeledSetMarkerAndHandicap(&Si, spxi, s->label, s->marker, s->handicap);
                    labels->val[spxi] = s->label;
                }
            }else{
                if(s->label > 0) {
                    labels->val[spxi] = s->label;
                }
            }
            kernel_map->val[last_kernel] = id; 
            s = s->next;
        }

        if(save_markers_folder != NULL)
        {
            // save the new markers file
            sprintf(filename, "%s/%s%s", save_markers_folder, basename, "-seeds_graph.txt");
            iftWriteSeedsGraph(Si, filename);
            iftDestroyLabeledSet(&Si);
        }
        iftDestroyIntArray(&labels);

        iftDestroyLabeledSet(&S0);
        iftDestroyIntArray(&spx_map);
        iftDestroyIntArray(&spx_id_map);
        iftFree(basename);
    }
    iftDestroyFileSet(&fs);

    // save the kernel correspondence
    FILE *fp = fopen(save_file, "a+");
    for (int i = 1; i <= last_kernel; i++)
        fprintf(fp, "%d ", kernel_map->val[i]);
    fprintf(fp, "\n");
    fclose(fp);
    iftDestroyIntArray(&kernel_map);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
