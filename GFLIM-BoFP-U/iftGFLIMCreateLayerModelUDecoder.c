#include "ift.h"

/**
 * \file iftGFLIMCreateLayerModel.c
 * \brief Create the kernel weights, bias and true labels for a layer on the second half of GFLIM-BoFP-U.
 * \note The kernels, biases, and true labels are saved in the model directory per training image. To merge them use iftGFLIMMergeLayerModel.
 * \note Differs from iftGFLIMCreateLayerModel on the unpooling operation.
 */

iftMatrix *iftGFLIMUCreateLayerModel(iftGFLIMGraph *graph, iftLabeledSet *S,
                                    iftFLIMLayer layer, float stdev_factor,
                                    float **bias, int **truelabel,
                                    iftGFLIMGraphAttention attention_opt,
                                    iftGFLIMAggregationType aggregation_opt)
{
    iftDataSet *Z = iftGFLIMExtractLabelDataSet(graph, S, layer.kernel_size[0] * layer.kernel_size[1], attention_opt, aggregation_opt);

#ifdef PRINT_VALS
    printf("Dataset %s\n", basename);
    for (int i = 0; i < Z->nsamples; i++)
    {
        printf("Sample %d: ", i);
        for (int j = 0; j < Z->nfeats; j++)
            printf("%f ", Z->sample[i].feat[j]);
        printf("\n");
    }
    printf("\n");
#endif

    iftNormalizeDataSetByZScoreInPlace(Z, NULL, stdev_factor);

#ifdef PRINT_VALS
    printf("Z-norm mean: ");
    for (int i = 0; i < Z->nfeats; i++)
        printf("%f ", Z->fsp.mean[i]);
    printf("\n");
    printf("Z-norm stdev: ");
    for (int i = 0; i < Z->nfeats; i++)
        printf("%f ", Z->fsp.stdev[i]);
    printf("\n");

    printf("Kernels after z-norm: \n");
    for (int i = 0; i < Z->nsamples; i++)
    {
        for (int j = 0; j < Z->nfeats; j++)
            printf("%f ", Z->sample[i].feat[j]);
        printf("\n");
    }
    printf("\n");
#endif

    (*truelabel) = iftAllocIntArray(Z->nsamples);
    iftMatrix *kernels = iftGFLIMComputeKernelsAndTrueLabels(Z, *truelabel);

#ifdef PRINT_VALS
    printf("Kernels after unit-norm: \n");
    for (int i = 0; i < kernels->nrows; i++)
    {
        for (int j = 0; j < kernels->ncols; j++)
            printf("%f ", iftMatrixElem(kernels, i, j));
        printf("\n");
    }
    printf("\n");
#endif

    /* compute biases and update filters */
    (*bias) = iftGFLIMComputeBias(kernels, Z);

#ifdef PRINT_VALS
    printf("Kernels after bias: \n");
    for (int i = 0; i < kernels->nrows; i++)
    {
        for (int j = 0; j < kernels->ncols; j++)
            printf("%f ", iftMatrixElem(kernels, i, j));
        printf("\n");
    }
    printf("\n");

    printf("Bias: \n");
    for (int i = 0; i < kernels->ncols; i++)
        printf("%f ", (*bias)[i]);
    printf("\n");
#endif

    iftDestroyDataSet(&Z);

    return kernels;
}

iftSet **createLabelDict(iftImage *input_label_map, iftImage *skipConn_label_map)
{
	if(input_label_map->n != skipConn_label_map->n) iftError("Label maps must have the same size", __func__);
	
	int input_max_label = -1, skipConn_max_label = -1;
	for(int p = 0; p < input_label_map->n; p++)
	{
		input_max_label = iftMax(input_max_label, input_label_map->val[p]);
		skipConn_max_label = iftMax(skipConn_max_label, skipConn_label_map->val[p]);
	}
	
	iftSet **label_dict = (iftSet**)iftAlloc(input_max_label, sizeof(iftSet*)); // dictionary from the current labels to the previous ones
	iftBMap **match_matrix = (iftBMap**)iftAlloc(input_max_label, sizeof(iftBMap*));
    
    for(int i = 0; i < input_max_label; i++)
	{
        match_matrix[i] = iftCreateBMap(skipConn_max_label);
		label_dict[i] = NULL;
	}
	
	for(int p = 0; p < input_label_map->n; p++)
	{
		int input_label = input_label_map->val[p] - 1;
		int skipConn_label = skipConn_label_map->val[p] - 1;
		if(!iftBMapValue(match_matrix[input_label], skipConn_label))
		{
            iftBMapSet1(match_matrix[input_label], skipConn_label);
			iftInsertSet(&(label_dict[input_label]), skipConn_label);
		}
	}
	for(int i = 0; i < input_max_label; i++) 
        iftDestroyBMap(&match_matrix[i]);
    iftFree(match_matrix);

	return label_dict;
}

iftGFLIMGraph *unpool(iftGFLIMGraph *input_graph, iftGFLIMGraph *skipConn_graph, iftSet **label_dict, char *skip_operation)
{
	int new_nfeatures = input_graph->num_feats;
	
    if(strcmp(skip_operation, "concat") == 0) {
		new_nfeatures += skipConn_graph->num_feats;
    }

    if(strcmp(skip_operation, "sum") == 0) {
        new_nfeatures = iftMax(skipConn_graph->num_feats, new_nfeatures);
    }
    
	iftGFLIMGraph *unpool_graph = iftCopyFLIMGraph(new_nfeatures, skipConn_graph, false); 
	
	for(int input_node=0; input_node < input_graph->num_nodes; input_node++)
	{
    	double *input_features = input_graph->nodes[input_node].feats;
		int input_size = input_graph->nodes[input_node].numPixels;
		iftSet *dict_vals = label_dict[input_node];
		while(dict_vals != NULL)
		{
			int skipConn_node = dict_vals->elem;
            if(skipConn_node >= skipConn_graph->num_nodes || skipConn_node < 0)
            {
                iftError("Skip connection node %d is out of bounds for the skip connection graph with %d nodes", __func__, skipConn_node, skipConn_graph->num_nodes);
            }

			double *skipConn_features = skipConn_graph->nodes[skipConn_node].feats;
			int skipConn_size = skipConn_graph->nodes[skipConn_node].numPixels;
			
			if(strcmp(skip_operation,"sum") == 0)
			{
                memcpy(unpool_graph->nodes[skipConn_node].feats, input_features, input_graph->num_feats  * sizeof(double));
                memcpy(unpool_graph->nodes[skipConn_node].feats + input_graph->num_feats, skipConn_features + input_graph->num_feats, (skipConn_graph->num_feats - input_graph->num_feats) * sizeof(double));
				/*for(int i = 0; i < input_graph->num_feats; i++)
					unpool_graph->nodes[skipConn_node].feats[i] = input_features[i];//*(skipConn_size/input_size);
                for(int i = input_graph->num_feats; i < skipConn_graph->num_feats; i++)
					unpool_graph->nodes[skipConn_node].feats[i] = skipConn_features[i];*/
                
			}else{
				if(strcmp(skip_operation,"concat") == 0)
				{
					for(int i = 0; i < input_graph->num_feats; i++)
						unpool_graph->nodes[skipConn_node].feats[i] = input_features[i]*(skipConn_size/input_size) + skipConn_features[i];
					memcpy(unpool_graph->nodes[skipConn_node].feats + input_graph->num_feats, skipConn_features, (int)skipConn_graph->num_feats);
				}else{
					iftError("Skip operation %s not supported", __func__, skip_operation);
				}
			}
			
			dict_vals = dict_vals->next;
		}
	}
	
	return unpool_graph;
}


int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 7)
        iftError("Usage: iftGFLIMGSCreateLayerModel <P1> <P2> <P3> <P4> <P5> <P6> \n"
                 "P1: model directory\n"
                 "P2: network architecture (.json)\n"
                 "P3: layer number (1, 2, 3, etc)\n"
                 "P4: attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE,...) \n"
                 "P5: aggregation strategy (CONCAT, MEAN, WMEAN) \n"
                 "P6: level (depth of the layer that's being created) \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftFLIMArch *arch = iftReadFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    iftGFLIMGraphAttention attention_opt = iftGLIMGraphGetAttentionType(argv[4]);
    iftGFLIMAggregationType aggregation_opt = iftGLIMGraphGetAggregationType(argv[5]);
    int level = atoi(argv[6]); // indicate the depth of the layer that's being created

    char input_graph_dir[200], input_labels_dir[200], skipConn_graph_dir[200], skipConn_labels_dir[200];

    if (layer < 1)
        iftError("Invalid layer number: %d", "main", layer);

    if (layer > arch->nlayers)
        iftError("Layer number %d is greater than the number of layers in the network (%d)", "main", layer, arch->nlayers);

    char markers_dir[200];
    sprintf(markers_dir, "%s/markers", model_dir);
    if (!iftDirExists(markers_dir))
        iftError("Image markers directory does not exist: %s", "main", markers_dir);

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(markers_dir, "-seeds_graph.txt", 1);
    if (fs->n < 1)
        iftError("Image markers directory is empty: %s", "main", markers_dir);

    sprintf(input_graph_dir, "%s/layer%d", model_dir, layer - 1);
    if (!iftDirExists(input_graph_dir))
        iftError("Graph directory does not exist: %s", "main", input_graph_dir);

    sprintf(skipConn_graph_dir, "%s/noPool%d", model_dir, level);
    if (!iftDirExists(skipConn_graph_dir))
        iftError("Graph directory does not exist: %s", "main", skipConn_graph_dir);

    sprintf(input_labels_dir, "%s/labels%d", model_dir, level+1);
    if (!iftDirExists(input_labels_dir))
        iftError("Labels directory does not exist: %s", "main", input_labels_dir);

    sprintf(skipConn_labels_dir, "%s/labels%d", model_dir, level);
    if (!iftDirExists(skipConn_labels_dir))
        iftError("Labels directory does not exist: %s", "main", skipConn_labels_dir);

    char *filename = iftAllocCharArray(512);
    sprintf(filename, "%s/model", model_dir);
    iftMakeDir(filename);

    for (int i = 0; i < fs->n; i++)
    {
        iftLabeledSet *S = iftReadSeedsGraph(fs->files[i]->path);
        if (S == NULL)
            continue;

        char *basename = iftFilename(fs->files[i]->path, "-seeds_graph.txt");

        // convert seeds from initial layer to the current layer
        iftImage *label_img0 = NULL, *input_labels = NULL;
        sprintf(filename, "%s/labels0/%s%s", model_dir, basename, ".pgm");
        if (!iftFileExists(filename))
            iftError("Labels file %s does not exist", __func__, filename);
        label_img0 = iftReadImageByExt(filename);

        // read labels of the provided level
        iftImage *skipConn_labels = NULL;
        sprintf(filename, "%s/%s%s", skipConn_labels_dir, basename, ".pgm");
        if (!iftFileExists(filename))
            iftError("Labels file %s does not exist", __func__, filename);
        skipConn_labels = iftReadImageByExt(filename);        

        //iftLabeledSet *T = iftGFLIMUpdateGraphMarkers(S, skipConn_labels, label_img0);
        //iftDestroyLabeledSet(&S);
        //S = T;
        iftGFLIMUpdateGraphMarkers(S, skipConn_labels, label_img0);
        iftDestroyImage(&label_img0);

        sprintf(filename, "%s/%s%s", input_labels_dir, basename, ".pgm");
        if (!iftFileExists(filename))
            iftError("Labels file %s does not exist", __func__, filename);
        input_labels = iftReadImageByExt(filename);

        // create a dictionary from the current labels to the previous ones (the current level/depth is always skipConn_level-1)
        iftSet **label_dict = createLabelDict(input_labels, skipConn_labels);
        

        // read the current graph
        sprintf(filename, "%s/%s%s", input_graph_dir, basename, ".json");
        if (!iftFileExists(filename))
            iftError("Graph file %s does not exist", "main", filename);
        iftGFLIMGraph *input_graph = iftReadFLIMGraph(filename);

        // read the current graph
        sprintf(filename, "%s/%s%s", skipConn_graph_dir, basename, ".json");
        if (!iftFileExists(filename))
            iftError("Graph file %s does not exist", "main", filename);
        iftGFLIMGraph *skipConn_graph = iftReadFLIMGraph(filename);

        // push seeds that did not exist in the previous layer at the end of the seed set
        /*int *seed_nodes = iftAlloc(skipConn_graph->num_nodes, sizeof(int*));
        iftLabeledSet *T2 = iftGFLIMUpdateGraphMarkers(S, input_labels, label_img0);
        iftLabeledSet *s = T2;
        for(int j = 0; j < skipConn_graph->num_nodes; j++) seed_nodes[j] = -1;
        // For each seed in T2 (input_graph), mark in seed_nodes all possible seeds in T
        while(s != NULL)
        {
            iftSet *dict_vals = label_dict[s->elem];
            while(dict_vals != NULL)
            {
                seed_nodes[dict_vals->elem] = s->elem;
                dict_vals = dict_vals->next;
            }
            s = s->next;
        }

        iftFree(seed_nodes);*/
        iftDestroyImage(&input_labels);
        iftDestroyImage(&skipConn_labels);

        // unpool and skip connection
        iftGFLIMGraph *unpool_graph = unpool(input_graph, skipConn_graph, label_dict, "sum");
        iftDestroyFLIMGraph(&skipConn_graph);
        for(int i = 0; i < input_graph->num_nodes; i++)
            iftDestroySet(&label_dict[i]);
        iftFree(label_dict);
        iftDestroyFLIMGraph(&input_graph);
        

        // create the kernels, bias and truelabels for the current graph
        float *bias;
        int *truelabel;
        iftMatrix *kernels = iftGFLIMUCreateLayerModel(unpool_graph, S, arch->layer[layer - 1], arch->stdev_factor, &bias, &truelabel, attention_opt, aggregation_opt);

        /* save kernels, biases, and truelabels of the kernels */
        sprintf(filename, "%s/model/%s-conv%d-kernels.npy", model_dir, basename, layer);
        iftWriteMatrix(kernels, filename);
        sprintf(filename, "%s/model/%s-conv%d", model_dir, basename, layer);
        iftGFLIMSaveBias(filename, bias, kernels->ncols);
        iftGFLIMSaveKernelLabels(filename, truelabel, kernels->ncols);

        iftFree(bias);
        iftFree(truelabel);
        iftDestroyMatrix(&kernels);
        iftFree(basename);
        iftDestroyFLIMGraph(&unpool_graph);
        iftDestroyLabeledSet(&S);
    }

    iftDestroyFileSet(&fs);
    iftDestroyFLIMArch(&arch);
    iftFree(filename);

    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
