# Graph-based feature learning from image markers

This repository contain the programs for GFLIM and GFLIM-BoFP

> [!IMPORTANT]  
> This repository is part of the IFT library. Please download the remaining files from <https://github.com/IsabelaBB/LibIFT>

## Compile

Compile LibIFT project

```shell
cd build; cmake ..; make; cp libift.a ../lib/; make install;
```

Alternatively, you can compile only the programs at demo/GFLIM or demo/Classification:

1. Set `NEWIFT_DIR` to the path to LibIFT: `export NEWIFT_DIR="<pathToLibIFTdir>";`

2. Compile GFLIM programs: `cd demo/GFLIM/; bash compile.sh 0;`

In both cases, executable files will be created at LibIFT/bin

## Programs

There are three version of GFLIM:

- GFLIM
- GFLIM Bag of Feature Points (GFLIM BoFP)
- Recurrent GFLIM BoFP (RGFLIM BoFP)
- Multiscale GFLIM BoFP (MGFLIM BoFP)

In GFLIM BoFP, you can either extract neighborhood information based on activation or color similarity. This option can be used setting to the variables `kernel_neighbors`, `pool_neighbors`, and/or `kernel_neighbors` the value `color` instead of the standard `act`. The programs that handle this option are at [GFLIM_BoFP_precompGraph](GFLIM_BoFP_precompGraph/) folder. This version along with RGFLIM BoFP and MGFLIM BoFP (folder [M-GFLIM-BoFP](M-GFLIM-BoFP/)) are beta version that might not work well.

### Pre-processing

For all versions, use [iftCreateGFLIMGraph](iftCreateGFLIMGraph.c) to convert images into graphs using superpixel segmentation. Or you can use your own tool for segmentation and convert the superpixel maps into graphs using [iftCreateGFLIMGraphFromLabels](iftCreateGFLIMGraphFromLabels.c). Alternatively, [iftGFLIMCreateNeighbMatricesAndGraphs](GFLIM_BoFP_precompGraph/iftGFLIMCreateNeighbMatricesAndGraphs.c) create graphs from images and their neighborhood matrices for all layers based on color similarity (i.e., when `kernel_neighbors`, `pool_neighbors`, and/or `kernel_neighbors` are set to `color`).

### Seeds creation

All GFLIM versions pixel labeled information to compute the kernels. The labeled information can be provided by the used through manually labeled markers or all pixels can be automatically extracted and labeled using ground-truth information.

#### Seeds from user-drawn markers

You can either use [iDSIF](https://github.com/IsabelaBB/iDISF) or other tool to manually draw markers on the training images and genate text with the markers information. The markers text files must have `ncoords width height` on the first row and `xcoord ycoord marker label handicap` on the remaining rows, in which `xcoord` and `ycoord` are the pixel coordinates, `label` is the class that pixel belongs, and `marker` and `handicap` are unused values maintained for compatibility with FLIM, set to -1 and 0, respectively.

#### Seeds from all pixels

Use [CreateSeedsFromAllPix](CreateSeedsFromAllPix.py) to create text files from the training images with all pixels labeled based on the class information on the image filenames.

#### Creating graph seeds

With the training graphs (and their respective superpixel images), the image seeds, and a architecture file (see [arch.json](https://github.com/IsabelaBB/LibIFT/blob/main/Datasets/archs/arch.json)), graph seed can be created using [iftGFLIMConvertSeeds](iftGFLIMConvertSeeds.c), [iftGFLIMConvertSeeds_v2](GFLIM-BoFP/iftGFLIMConvertSeeds_v2.c), or [iftGFLIMPrecConvertSeeds](GFLIM_BoFP_precompGraph/iftGFLIMPrecConvertSeeds.c). The standard option is [iftGFLIMConvertSeeds](iftGFLIMConvertSeeds.c).

### Training

> Note: Before trainig, you must to follow the previous steps.

To train a model using GFLIM, simply use [iftGFLIMTrainLayer](GFLIM/iftGFLIMTrainLayer.c) for each layer.

For the other versions, you must to follow the steps for each layer:

- Create a layer model: Creates kernels, biases, and true labels per training image.
- Merge layer models: Merge the kernels, bias, and labels in unified files.
- Encode merged layer: Extract features from a training graphs using the merged files.

Alternatively, you can also extract features from the training graphs before the merging using the "Encode layer" option.

#### Create a layer model

- [iftGFLIMCreateLayerModel](GFLIM-BoFP/iftGFLIMCreateLayerModel.c)
- [iftGFLIMPrecCreateLayerModel](GFLIM_BoFP_precompGraph/iftGFLIMPrecCreateLayerModel.c)

#### Encode layer

- [iftGFLIMEncodeLayer](GFLIM-BoFP/iftGFLIMEncodeLayer.c)
- [iftGFLIMEncodeLayerV3](GFLIM_BoFP_precompGraph/iftGFLIMEncodeLayerV3.c)

#### Merge layer models

- [iftGFLIMMergeLayerModels](GFLIM-BoFP/iftGFLIMMergeLayerModels.c)

#### Encode merged models
 
- [iftGFLIMEncodeMergedLayer](GFLIM-BoFP/iftGFLIMEncodeMergedLayer.c)
- [iftGFLIMEncodeMergedLayerBatch](GFLIM-BoFP/iftGFLIMEncodeMergedLayerBatch.c)
- [iftGFLIMEncodeMergedLayerV3](GFLIM_BoFP_precompGraph/iftGFLIMEncodeMergedLayerV3.c)
- [iftGFLIMEncodeMergedLayerBatchV3](GFLIM_BoFP_precompGraph/iftGFLIMEncodeMergedLayerBatchV3.c)

#### Handling skip layers and multiscale layers

- [iftGFLIMSkipConnection](M-GFLIM-BoFP/iftGFLIMSkipConnection.c): Handle skip connections from a layer X to a layer X+Y. Merge the output of the layers based on the file `<model_dir>/model/kernels_map.txt` that contains the kernels mapping. The layers to be merged are specified in the architecture file. Used in RGFLIM and MGFLIM BoFP.
- [iftGFLIMWriteKernelsCorrespondence](M-GFLIM-BoFP/iftGFLIMWriteKernelsCorrespondence.c): Writes a text file with the correspondence between the kernels of two layers. Used in RGFLIM and MGFLIM BoFP.
- [iftGFLIMMergeScales](M-GFLIM-BoFP/iftGFLIMMergeScales.c): Merge scales based on a .txt file with the kernels mapping. Used in MGFLIM BoFP.

### Visualize a layer's output

Use [iftGFLIMWriteActivations](iftGFLIMWriteActivations.c) to write the images activations from the output graphs of any GFLIM version.

### Decoding

There are a few automatic decoders to use with [iftGFLIMDecodeLayer](iftGFLIMDecodeLayer.c) for any GFLIM model. The decoders aggregate the information from the output graphs in one channel per class, creating saliency maps. These are usually used for object segmentation / detection.

### Extracting features

After training use the following programs to extract information from the validation/test data. These programs also allows to write the activation images and decode the output graphs.

#### GFLIM, GFLIM BoFP, or RGFLIM BoFP

- [iftGFLIMExtract](iftGFLIMExtract.c): Extract features from a set of images.
- [iftGFLIMExtract2](iftGFLIMExtract2.c): Extract features from a set of graphs and superpixel maps.

#### GFLIM BoFP with neighbors based on color similarity

- [iftGFLIMExtractV3](GFLIM_BoFP_precompGraph/iftGFLIMExtractV3.c): Extract features from a set of images. This version does NOT support residual connections and stride `IFT_GFLIM_IMAGESTRIDE`.

#### MGFLIM BoFP

- [iftGFLIMExtractMultiscale](M-GFLIM-BoFP/iftGFLIMExtractMultiscale.c): Extract features from a set of graphs and superpixel images, each with multiple segmentation scales.

### Train an SVM and using it to classify

Use [iftGFLIMActivDataSet](iftGFLIMActivDataSet.c) to convert the graphs created with a GFLIM-based model into vectors and write them in a .zip file in `iftDataset` format. Then, train an SVM with [iftSupTrainBySVM](https://github.com/IsabelaBB/Classification/blob/main/iftSupTrainBySVM.c) and classify using [iftClassifyBySVM](https://github.com/IsabelaBB/Classification/blob/main/iftClassifyBySVM.c), both from [Classification](https://github.com/IsabelaBB/Classification) module. The training and classification use the aforementioned .zip file.
