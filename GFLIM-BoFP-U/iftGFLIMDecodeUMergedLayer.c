#include "ift.h"

/**
 * \file iftGFLIMGSEncodeMergedLayer.c
 * \brief Encode a layer using the merged model created by iftGFLIMMergeLayerModels.
 * Used to create the first half of the GFLIM-BoFP-U model.
 * \note The second half of the model is created by iftGFLIMCreateLayerModelUDecoder.
 */


iftFLIMGraph *iftGFLIMUEncodeLayer(iftFLIMGraph *graph,
                                  iftMatrix *Kernels, float *bias,
                                  iftFLIMLayer layer, 
                                  iftFLIMGraphAttention attention_opt, 
                                  int *kernel_labels,
                                  iftGFLIMAggregationType aggregation_opt,
                                  bool reduce)
{

    iftMImage *activ = iftGFLIMConvLayer(graph, Kernels, layer.kernel_size[0] * layer.kernel_size[1], attention_opt, kernel_labels, aggregation_opt, reduce);

    iftMImage *activ_relu = NULL;
    if (layer.relu)
        activ_relu = iftMReLU(activ, bias, true);
    else
    {
        if (bias != NULL)
            activ_relu = iftGFLIMSumMImgToFloatArray(activ, bias, true);
        else
            activ_relu = activ;
    }
    activ = NULL;

#ifdef PRINT_VALS
    printf("activ_relu: %d %d %d %ld \n", activ_relu->xsize, activ_relu->ysize, activ_relu->zsize, activ_relu->m);
    for (int i = 0; i < activ_relu->n; i++)
    {
        for (int j = 0; j < activ_relu->m; j++)
            printf("%f ", activ_relu->val[i][j]);
        printf("\n");
    }
    printf("\n");
#endif

    iftFLIMGraph *output_graph = iftMImageToFLIMGraph(activ_relu, graph);
    iftDestroyMImage(&activ_relu);
    
    return output_graph;
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
			iftInsertSet(&label_dict[input_label], skipConn_label);
		}
	}
	for(int i = 0; i < input_max_label; i++) 
        iftDestroyBMap(&match_matrix[i]);
    iftFree(match_matrix);

	return label_dict;
}

iftFLIMGraph *unpool(iftFLIMGraph *input_graph, iftFLIMGraph *skipConn_graph, iftSet **label_dict, char *skip_operation)
{
	int new_nfeatures = input_graph->num_feats;
	
    if(strcmp(skip_operation, "concat") == 0) {
		new_nfeatures += skipConn_graph->num_feats;
    }

    if(strcmp(skip_operation, "sum") == 0) {
        new_nfeatures = iftMax(skipConn_graph->num_feats, new_nfeatures);
    }
    
	iftFLIMGraph *unpool_graph = iftCopyFLIMGraph(new_nfeatures, skipConn_graph, false); 
	
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
					unpool_graph->nodes[skipConn_node].feats[i] = input_features[i]*(skipConn_size/input_size);
                for(int i = 0; i < skipConn_graph->num_feats; i++)
					unpool_graph->nodes[skipConn_node].feats[i] += skipConn_features[i];*/
                
			}else{
				if(strcmp(skip_operation,"concat") == 0)
				{
					for(int i = 0; i < input_graph->num_feats; i++)
						unpool_graph->nodes[skipConn_node].feats[i] = input_graph->nodes[input_node].feats[i]*(skipConn_size/input_size) + skipConn_graph->nodes[skipConn_node].feats[i];
					memcpy(unpool_graph->nodes[skipConn_node].feats + input_graph->num_feats, skipConn_graph->nodes[skipConn_node].feats, (int)skipConn_graph->num_feats);
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

    if (argc != 8 && argc != 7)
        iftError("Usage: iftGFLIMGSEncodeMergedLayer <P1> <P2> <P3> <P4> <P5> [<P6>] \n"
                 "[1] model directory \n"
                 "[2] architecture of the network (.json) \n"
                 "[3] layer number (1, 2, 3) \n"
                 "[4] attention (NONE, NODE_SIZE, NEIGHB_SIMIL, NEIGHB_SIZE, ...) \n"
                 "[5] aggregation (CONCAT, MEAN, WMEAN) \n"
                 "[6] depth of the layer that's being created (>= 0) \n"
                 "[7] reduce output channels based on attention values (1/0). It is necessary a compatible attention mechanism \n",
                 "main");

    tstart = iftTic();

    char *model_dir = argv[1];
    iftFLIMArch *arch = iftReadFLIMArch(argv[2]);
    int layer = atoi(argv[3]);
    iftFLIMGraphAttention attention_opt = iftGLIMGraphGetAttentionType(argv[4]);
    iftGFLIMAggregationType aggregation_opt = iftGLIMGraphGetAggregationType(argv[5]);
    int level = atoi(argv[6]); // indicate the depth of the layer that's being created
    bool kernel_removal = false;
    if (argc == 8)
        kernel_removal = atoi(argv[7]);

    if (layer < 1)
        iftError("Invalid layer number: %d", "main", layer);

    if (layer > arch->nlayers)
        iftError("Invalid layer number: %d", "main", layer);

    char input_graph_dir[200], input_labels_dir[200], skipConn_graph_dir[200], skipConn_labels_dir[200], output_graph_dir[200];
    
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

    sprintf(output_graph_dir, "%s/layer%d", model_dir, layer);
    iftMakeDir(output_graph_dir);

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(input_graph_dir, ".json", 1);
    if (fs->n < 1)
        iftError("Graph directory is empty: %s", "main", input_graph_dir);

    iftMatrix *Kmerged = NULL;
    float *bias_merged = NULL;
    char *filename = iftAllocCharArray(512);

    // read kernels and bias
    sprintf(filename, "%s/model/conv%d", model_dir, layer);
    iftGFLIMLoadMergedModel(filename, &Kmerged, &bias_merged);

    // read the kernel labels
    int *true_labels = NULL;
    sprintf(filename, "%s/model/conv%d-labels.txt", model_dir, layer);
    if (!iftFileExists(filename))
        iftError("File %s does not exist", "main", filename);
    int nkernels = 0;
    true_labels = iftGFLIMLoadKernelLabels(filename, &nkernels);

    /* Encode layer with merged model */

    for (int i = 0; i < fs->n; i++)
    {
        char *basename = iftFilename(fs->files[i]->path, ".json");

        // read input labels
        iftImage *input_labels = NULL;
        sprintf(filename, "%s/%s%s", input_labels_dir, basename, ".pgm");
        if (!iftFileExists(filename))
            iftError("Labels file %s does not exist", __func__, filename);
        input_labels = iftReadImageByExt(filename);

        // read labels of the provided level
        iftImage *skipConn_labels = NULL;
        sprintf(filename, "%s/%s%s", skipConn_labels_dir, basename, ".pgm");
        if (!iftFileExists(filename))
            iftError("Labels file %s does not exist", __func__, filename);
        skipConn_labels = iftReadImageByExt(filename);

        // create a dictionary from the current labels to the previous ones (the current level/depth is always skipConn_level-1)
        iftSet **label_dict = createLabelDict(input_labels, skipConn_labels);
        iftDestroyImage(&input_labels);
        iftDestroyImage(&skipConn_labels);

        // read the current graph
        sprintf(filename, "%s/%s%s", input_graph_dir, basename, ".json");
        if (!iftFileExists(filename))
            iftError("Graph file %s does not exist", "main", filename);
        iftFLIMGraph *input_graph = iftReadFLIMGraph(filename);

        // read the current graph
        sprintf(filename, "%s/%s%s", skipConn_graph_dir, basename, ".json");
        if (!iftFileExists(filename))
            iftError("Graph file %s does not exist", "main", filename);
        iftFLIMGraph *skipConn_graph = iftReadFLIMGraph(filename);

        // unpool and skip connection
        iftFLIMGraph *unpool_graph = unpool(input_graph, skipConn_graph, label_dict, "sum");
        //iftFLIMGraph *unpool_graph = skipConn_graph;
        printf("unpool nodes %ld\n", unpool_graph->num_nodes);
        iftDestroyFLIMGraph(&skipConn_graph);
        for(int i = 0; i < input_graph->num_nodes; i++)
            iftDestroySet(&label_dict[i]);
        iftFree(label_dict);
        iftDestroyFLIMGraph(&input_graph);

        /* conv + att + bias */
        iftFLIMGraph *conv_graph = iftGFLIMUEncodeLayer(unpool_graph, Kmerged, bias_merged, arch->layer[layer - 1], attention_opt, true_labels, aggregation_opt, kernel_removal);
        iftDestroyFLIMGraph(&unpool_graph);

        /* save output */
        sprintf(filename, "%s/%s.json", output_graph_dir, basename);
        iftWriteFLIMGraph(conv_graph, filename);
        iftDestroyFLIMGraph(&conv_graph);
        iftFree(basename);
    }

    iftFree(true_labels);
    iftDestroyMatrix(&Kmerged);
    iftFree(bias_merged);
    iftFree(filename);
    iftDestroyFileSet(&fs);
    iftDestroyFLIMArch(&arch);
    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return (0);
}
