# Graph-based feature learning from image markers

This repository contain the programs for GFLIM and GFLIM-BoFP

> [!IMPORTANT]  
> This repository is part of the IFT library. Please download the remaining files from <https://github.com/IsabelaBB/LibIFT>

## Compile

Before compiling, set `NEWIFT_DIR` to the path to LibIFT

```shell
export NEWIFT_DIR="~/Codes/LibIFT"; 
```

Then, compile GFLIM programs

```bash
bash compile.sh 0
```

The executable files will be created at LibIFT/bin

## Programs

### Pre-processing

- [iftCreateGFLIMGraph](iftCreateGFLIMGraph.c): Convert images into graphs for a GFLIM or GFLIM-BoFP model.
- [iftGFLIMConvertSeeds](iftGFLIMConvertSeeds.c): Convert image-based markers to graph-based markers. Optionally, it also performs k-means on the graph markers if a number of feature points per marker > 0 is defined. The markers coordinates must be in "*-seeds.txt" files. 
- [iftGFLIMConvertSeeds_v2](GFLIM-BoFP/iftGFLIMConvertSeeds_v2.c): Convert image-based markers to graph-based markers, performs a k-means per graph marker, and another k-means per label. The markers coordinates must be in "*-seeds.txt" files. This program is only used for GFLIM-BoFP.

### Training

- [iftGFLIMTrainLayer](GFLIM/iftGFLIMTrainLayer.c): Train a layer model using GFLIM. It performs marker-based normalization on the graph-based markers, compute the kernel bank and apply convolution to extract features from the training graphs.
- [iftGFLIMCreateLayerModel](GFLIM-BoFP/iftGFLIMCreateLayerModel.c): Create the model for a given layer using GFLIM-BoFP. It creates kernels, biases, and true labels per training image.
- [iftGFLIMMergeLayerModels](GFLIM-BoFP/iftGFLIMMergeLayerModels.c): Merge the kernels, bias, and labels of a layer created by iftGFLIMCreateLayerModel.
- [iftGFLIMEncodeLayer](GFLIM-BoFP/iftGFLIMEncodeLayer.c): Encode a layer using the non-merged kernels, bias, and labels created by iftGFLIMCreateLayerModel.
- [iftGFLIMEncodeMergedLayer](GFLIM-BoFP/iftGFLIMEncodeMergedLayer.c): Encode a layer using the merged model created by iftGFLIMMergeLayerModels.

### Visualize a layer's output

- [iftGFLIMWriteActivations](iftGFLIMWriteActivations.c): Write the activations of a GFLIM or a GFLIM-BoFP layer

### Decoding

- [iftGFLIMDecodeLayer](iftGFLIMDecodeLayer.c): Decode a layer of a GFLIM or GFLIM-BoFP model

### Extracting features and decoding from a trained model

- [iftGFLIMExtract](iftGFLIMExtract.c): Extract features from a set of images and decode from a GFLIM or a GFLIM-BoFP trained model.


## Python scripts

The following files automate the training and inference of a model. 

- [GFLIM-train](GFLIM/GFLIM-train.py): Train a GFLIM model and decode the features of each layer. It optionally writes the activation images of each layer.
- [GFLIM-BoFP-train](GFLIM-BoFP/GFLIM-BoFP-train.py): Train a GFLIM-BoFP model and decode the features of each layer. It optionally writes the activation images of each layer.
- [GFLIM-extract](GFLIM-extract.py): Extract features from a set of images and decode from a GFLIM or a GFLIM-BoFP trained model. 
