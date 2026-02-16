/**
 * \file iftGFLIMMergeScales.c
 * \brief Merge layers or scales based on a .txt file with the kernels mapping
 * \author Isabela Borlido
 */

#include "ift.h"

char* read_file_scales(const char* filename) { 
    FILE* file = fopen(filename, "r"); 
    if (!file) { 
        perror("Failed to open file"); 
        return NULL; 
    } 
 
    // Allocate initial memory for the file content 
    size_t capacity = 1024; 
    char* content = malloc(capacity); 
    if (!content) { 
        perror("Failed to allocate memory"); 
        fclose(file); 
        return NULL; 
    } 
 
    size_t length = 0; 
    size_t bytesRead; 
 
    // Read the file in chunks 
    while ((bytesRead = fread(content + length, sizeof(char), 1024, file)) > 0) { 
        length += bytesRead; 
 
        // If we reach capacity, resize the buffer 
        if (length + 1024 > capacity) { 
            capacity *= 2; // Double the capacity 
            content = realloc(content, capacity); 
            if (!content) { 
                perror("Failed to reallocate memory"); 
                fclose(file); 
                return NULL; 
            } 
        } 
    } 
 
    // Null-terminate the string 
    if (length < capacity) { 
        content[length] = '\0'; 
    } else { 
        content = realloc(content, length + 1); 
        content[length] = '\0'; 
    } 
 
    fclose(file); 
    return content; 
} 

iftIntArray *parseStringToIntArray(char *str)
{
    // count the number of integers in the string
    int n = 1;
    for(char *p = str; *p != '\0'; p++)
    {
        if(*p == ',')
            n++;
    }

    char *token = strtok(str, ",");
    if(token == NULL) return NULL;

    iftIntArray *arr = iftCreateIntArray(n);
    int i = 0;
    while(token != NULL)
    {
        arr->val[i] = atoi(token);
        token = strtok(NULL, ",");
        i++;
    }
    return arr;
}

iftIntArray **iftGFLIMCompute_kernel_maps_scales(char *filename)
{   
    char *buf = read_file_scales(filename);

    int nlines = 100, line_count = 0;
    const char line_delimiter[] = "\n";
    const char number_delimiter[] = " ";
    char *line, *endptr; 
    int nkernels = 100, nkernels0 = 0, kernel_count = 0;

    char* line_saveptr = NULL;
    char* number_saveptr = NULL;

    int **kernel_maps_scale0 = iftAlloc(nlines, sizeof(int *));

    // Get the first line 
    line = strtok_r(buf, line_delimiter, &line_saveptr); 

    while(line != NULL)
    {
        // break into numbers
        char* inner_token = strtok_r(line, number_delimiter, &number_saveptr);

        // allocate memory for more lines if needed
        if(line_count >= nlines)
        {
            nlines += 100;
            kernel_maps_scale0 = realloc(kernel_maps_scale0, nlines * sizeof(int *));
            if(kernel_maps_scale0 == NULL)
                iftError("out of memory, nlines=%d", __func__, nlines);
        }

        kernel_count = 0, nkernels = 100;
        kernel_maps_scale0[line_count] = iftAllocIntArray(nkernels);
        
        // get each number
        while (inner_token != NULL) {

            // allocate memory for more kernels if needed
            if(kernel_count >= nkernels)
            {
                nkernels += 100;
                kernel_maps_scale0[line_count] = realloc(kernel_maps_scale0[line_count], nkernels * sizeof(int));
                if(kernel_maps_scale0[line_count] == NULL)
                    iftError("out of memory, nkernels=%d", __func__, nkernels);
            }
            kernel_maps_scale0[line_count][kernel_count] = (int) strtol(inner_token,&endptr,10);
            kernel_count++;
            inner_token = strtok_r(NULL, number_delimiter, &number_saveptr);
        }
        if(nkernels != kernel_count)
            kernel_maps_scale0[line_count] = realloc(kernel_maps_scale0[line_count], kernel_count * sizeof(int));
        if(line_count == 0)
            nkernels0 = kernel_count;
        line_count++;
        line = strtok_r(NULL, line_delimiter, &line_saveptr);
    }
    nlines = line_count;
    free(buf);

    printf("nlines: %d, nkernels0: %d\n", nlines, nkernels0);

    if(line_count == 0)
        iftError("No lines found in the file", __func__);
    if(nkernels0 == 0)
        iftError("No kernels found in the first line of the file", __func__);

    iftIntArray **kernel_maps = iftAlloc(nlines+1, sizeof(iftIntArray *));
    
    // compute the kernel maps for all scales
    // from scale 0 to scale i
    
    kernel_maps[0] = iftCreateIntArray(nkernels0); 
    for(int i=0; i < nkernels0; i++)
        kernel_maps[0]->val[i] = i;
    for(int l = 0; l < nlines; l++)
    {
        kernel_maps[l+1] = iftCreateIntArray(nkernels0); 
        int nkernelsi = 0;
        for(int i=0; i < nkernels0; i++)
        {
            int k0 = kernel_maps_scale0[l][i] - 1;
            if(k0 == i)
            {
                kernel_maps[l+1]->val[i] = nkernelsi;
                nkernelsi++;
            }else
                kernel_maps[l+1]->val[i] = kernel_maps[l+1]->val[k0];
        }
        iftFree(kernel_maps_scale0[l]);
    }   
    iftFree(kernel_maps_scale0);

    return kernel_maps;
}

void iftGFLIMMergeScales(int layer, char *kernels_file, char *model_dir, iftFileSet *fs, iftIntArray *list_superpixels, char *merge_option)
{
    // compute the kernel maps for the layers that will be used in the skip connection
    iftIntArray **kernels_maps = iftGFLIMCompute_kernel_maps_scales(kernels_file); // compute the kernel maps for all layers

    char filename[512];
    sprintf(filename, "%s/layer%d", model_dir, layer);
    if(!iftDirExists(filename)) iftMakeDir(filename);
    sprintf(filename, "%s/labels%d", model_dir, layer);
    if(!iftDirExists(filename)) iftMakeDir(filename);

    for (int file_id = 0; file_id < fs->n; file_id++)
    {
        char ext[10], filename[512];
        sprintf(ext, "%s", iftFileExt(fs->files[file_id]->path));
        char *basename = iftFilename(fs->files[file_id]->path, ext);

        sprintf(filename, "%s/scales/%d/labels%d/%s.pgm", model_dir, list_superpixels->val[0], layer, basename);
        iftImage *spx_img0 = iftReadImageByExt(filename);

        int **node_maps = iftAlloc(list_superpixels->n, sizeof(int *));
        iftMatrix **matrices = iftAlloc(list_superpixels->n, sizeof(iftMatrix *));

        sprintf(filename, "%s/scales/%d/layer%d/%s.json", model_dir, list_superpixels->val[0], layer, basename);
        iftGFLIMGraph *graph_finer = iftReadGFLIMGraph(filename);
        iftGFLIMMaxNormGraph(graph_finer);

        node_maps[0] = iftAlloc(graph_finer->num_nodes, sizeof(int));
        for(int i = 0; i < graph_finer->num_nodes; i++)
            node_maps[0][i] = i;

        matrices[0] = iftGFLIMGraphToNodeFeaturesMatrix(graph_finer);

        for(int j = 1; j < list_superpixels->n; j++)
        {
            sprintf(filename, "%s/scales/%d/layer%d/%s.json", model_dir, list_superpixels->val[j], layer, basename);
            iftGFLIMGraph *graph_skip = iftReadGFLIMGraph(filename);
            iftGFLIMMaxNormGraph(graph_skip);
            matrices[j] = iftGFLIMGraphToNodeFeaturesMatrix(graph_skip);
            iftDestroyGFLIMGraph(&graph_skip);

            sprintf(filename, "%s/scales/%d/labels%d/%s.pgm", model_dir, list_superpixels->val[j], layer, basename);
            iftImage *spx_img = iftReadImageByExt(filename);

            node_maps[j] = iftGFLIMComputeNodeMap(spx_img, spx_img0);
            iftDestroyImage(&spx_img);

            iftMatrix *tmp = iftGFLIMUpsampleFeatureMatrix(matrices[j], node_maps[j], kernels_maps[j], matrices[0]->nrows, matrices[0]->ncols);
            iftDestroyMatrix(&matrices[j]);
            matrices[j] = tmp;
            tmp = NULL;
        }

        // merge the feature matrices
        iftMatrix *merged = NULL;
        if(strcmp(merge_option, "dotprod") == 0)
            merged = iftGFLIMMergeMatrices_dotProd(matrices, list_superpixels->n, 0, false, false);
        else if(strcmp(merge_option, "dotprod_sumscale0") == 0)
            merged = iftGFLIMMergeMatrices_dotProd_sumScale0(matrices, list_superpixels->n, 0, false, false);
        else if(strcmp(merge_option, "dotprod_sumscalei") == 0)
            merged = iftGFLIMMergeMatrices_dotProd_sumScalei(matrices, list_superpixels->n, 0, false, false);
        else if(strcmp(merge_option, "sum") == 0)
            merged = iftGFLIMMergeMatrices_sum(matrices, list_superpixels->n, 0, false, false, -1);
        else if(strcmp(merge_option, "sum_sumscale0") == 0)
            merged = iftGFLIMMergeMatrices_sum(matrices, list_superpixels->n, 0, false, false, 0);
        else if(strcmp(merge_option, "sum_sumscalei") == 0)
            merged = iftGFLIMMergeMatrices_sum(matrices, list_superpixels->n, 0, false, false, list_superpixels->n-1);
        else
            iftError("Unknown merge function: %s",  __func__, merge_option);

        for(int j = 0; j < list_superpixels->n; j++)
        {
            iftDestroyMatrix(&matrices[j]);
            iftFree(node_maps[j]);
        }
        iftFree(matrices);
        iftFree(node_maps);

        // copy the merged features to the graph of the current layer
        if(graph_finer->num_nodes != merged->nrows || graph_finer->num_feats != merged->ncols)
            iftError("Graph and merged matrix dimensions do not match: (%ld, %ld) != (%d, %d)", __func__, 
                    graph_finer->num_nodes, graph_finer->num_feats, merged->nrows, merged->ncols); 
        for(int i = 0; i < graph_finer->num_nodes; i++)
        {
            for(int j = 0; j < graph_finer->num_feats; j++)
                graph_finer->nodes[i].feats[j] = iftMatrixElem(merged, j, i);
        }
        // min-max normalization per channel
        /*for(int j = 0; j < graph_finer->num_feats; j++)
        {
            float min = iftMatrixElem(merged, j, 0);
            float max = iftMatrixElem(merged, j, 0);
            for(int i = 1; i < graph_finer->num_nodes; i++)
            {
                if(iftMatrixElem(merged, j, i) < min)
                    min = iftMatrixElem(merged, j, i);
                if(iftMatrixElem(merged, j, i) > max)
                    max = iftMatrixElem(merged, j, i);
            }
            for(int i = 0; i < graph_finer->num_nodes; i++)
                graph_finer->nodes[i].feats[j] = (iftMatrixElem(merged, j, i) - min) / (max - min);
        }*/
        iftDestroyMatrix(&merged);

        // save the graph with the skip connections
        sprintf(filename, "%s/layer%d/%s.json", model_dir, layer, basename);
        iftWriteGFLIMGraph(graph_finer, filename);
        iftDestroyGFLIMGraph(&graph_finer);

        // save the superpixel image of the finer layer
		sprintf(filename, "%s/labels%d/%s.pgm", model_dir, layer, basename);
		iftWriteImageByExt(spx_img0, filename);
        iftDestroyImage(&spx_img0);
        iftFree(basename);
   }
    
    for(int j = 0; j < list_superpixels->n; j++){
        iftDestroyIntArray(&kernels_maps[j]);
    }
    iftFree(kernels_maps);
}

int main(int argc, char *argv[]) {
    timer *tstart;
    
    if (argc != 7)
      iftError("Usage: iftGFLIMMergeScales P1 P2 P3 P4 P5 P6\n"
		"P1: images folder\n"
        "P2: list of superpixels\n"
        "P3: model folder\n"
		"P4: kernel maps file \n"
		"P5: current layer \n"
        "P6: merge option \n"
		,"main");

    tstart = iftTic();

    char *images_folder = argv[1];
    iftIntArray *list_superpixels = parseStringToIntArray(argv[2]);
    char *model_dir = argv[3];
    char *kernel_maps_file = argv[4];
    int current_layer = atoi(argv[5]);
    char *merge_option = argv[6];
	char filename[1024];

    if(!iftDirExists(model_dir))
      iftError("Model directory does not exist: %s", "main", model_dir);
    if(!iftDirExists(images_folder) && !iftFileExists(images_folder))
      iftError("Images directory does not exist: %s", "main", images_folder);
    if(!iftFileExists(kernel_maps_file))
      iftError("Kernel maps file does not exist: %s", "main", kernel_maps_file);
    if(current_layer < 1)
      iftError("Invalid layer: %d", "main", current_layer);
    if(list_superpixels == NULL || list_superpixels->n == 0)
      iftError("Invalid list of superpixels: %s", "main", argv[2]);

    iftFileSet *fs = iftLoadFileSetFromDirOrCSV(images_folder, 1, false);
    if(fs->n < 1)
      iftError("No files found in the images directory: %s", "main", images_folder);
    
    for(int i =0; i < list_superpixels->n; i++){
        sprintf(filename, "%s/scales/%d/labels%d", model_dir, list_superpixels->val[i], current_layer);
        if(!iftDirExists(filename))
          iftError("Labels directory does not exist: %s", "main", filename);
        sprintf(filename, "%s/scales/%d/layer%d", model_dir, list_superpixels->val[i], current_layer);
        if(!iftDirExists(filename))
          iftError("Layer directory does not exist: %s", "main", filename);
    }

	iftGFLIMMergeScales(current_layer, kernel_maps_file, model_dir, fs, list_superpixels, merge_option);
    iftDestroyIntArray(&list_superpixels);
    iftDestroyFileSet(&fs);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
