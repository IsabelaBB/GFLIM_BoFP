import os
import shutil
import json
import argparse
import sys

parser = argparse.ArgumentParser(
    prog="GFLIM-BoFP-train",
    description="Train a model using GFLIM-BoFP",
    epilog="Thanks for using %(prog)s! :)",
)

parser.add_argument("img_dir", help="Image directory", type=str)
parser.add_argument("markers_dir", help="Image markers directory", type=str)
parser.add_argument("layer", help="Layer number", type=int)
parser.add_argument("model_dir", help="Model directory", type=str)
parser.add_argument("stride", help="Stride type", type=str, choices=["none", "image-based", "graph-based"])
parser.add_argument("final_spx", help="Final superpixels", type=int)
parser.add_argument("graph_feat", help="Graph feature (see iftGFLIMGraph.h for more information)", type=str, choices=["MEAN", "SUM", "SUM_MULT_RELSIZE", "MEAN_MULT_RELSIZE", 
                                                                                                                      "MEAN_CONCAT_RELSIZE", "MEAN_CONCAT_RELSIZE_NEIGHB", 
                                                                                                                      "MEAN_CONCAT_RELSIZE_HARALICK", 
                                                                                                                      "SUM_MULT_RELSIZE_CONCAT_HARALICK", "SUM_CONCAT_HARALICK"])
parser.add_argument("spx_alg", help="Superpixel algorithm.", type=str, choices=["DISF", "ERGC"])
parser.add_argument("decoder", help="Decoder type.", type=str, choices=["tristate", "hybrid", "graph-adaptive", "gaussian", "test"])
parser.add_argument("attention", help="Attention option.", type=str, choices=["NONE", "NODE_RELSIZE", "NODE_SIZE", "NEIGHB_SIMIL", "COSINE_SIMILARITY", #"NEIGHB_SIZE", 
                                                                                "CLASS_DOMINANCE_TEST", "CLASS_DOMINANCE", 
                                                                                "CLASS_DOMINANCE_POWER2", "CLASS_DOMINANCE_SOFTMAX_NEIGHB"])
parser.add_argument("aggregation", help="Aggregation option.", type=str, choices=["CONCAT", "MEAN", "WMEAN"])

parser.add_argument("-arch", "--arch_file", help="Arch file", type=str, default="")
parser.add_argument("-init_seeds", "--init_seeds", help="Initial number of seeds in DISF. Required only if spx_alg is DISF", type=int, default=0)
parser.add_argument("-act", "--activation", help="Write activations", action="store_true")
parser.add_argument("-reduce_kernels", "--reduce_kernels", help="Use another k-means to reduce the number of kernels to 64 per layer maximum.", type=int, default=0)
parser.add_argument("-dropout", "--dropout", help=" ", action="store_true")
parser.add_argument("-path", "--path", help=" ", type=str, default="../bin")
parser.add_argument("-graphdir", "--graphdir", help=" ", type=str, default="")
parser.add_argument("-labelsdir", "--labelsdir", help=" ", type=str, default="")


def setFunctions(path):
    return {
        'CreateGraph': '{}/iftCreateGFLIMGraph'.format(path),
        'ConvertSeeds': '{}/iftGFLIMConvertSeeds'.format(path),
        'ConvertSeeds2': '{}/iftGFLIMConvertSeeds_v2'.format(path),
        'CreateLayerModel': '{}/iftGFLIMCreateLayerModel'.format(path),
        'EncodeLayer': '{}/iftGFLIMEncodeLayer'.format(path),
        'WriteActivations': '{}/iftGFLIMWriteActivations'.format(path),
        'MergeLayerModels': '{}/iftGFLIMMergeLayerModels'.format(path),
        'EncodeMergedLayer': '{}/iftGFLIMEncodeMergedLayer'.format(path),
        'DecodeLayer': '{}/iftGFLIMDecodeLayer'.format(path)
    }

def verifyArgs(path):
    if(args.graphdir != ""):
        if(os.path.exists(args.graphdir) == False):
            print("Graph directory does not exist")
            exit()

    if(args.labelsdir != ""):
        if(os.path.exists(args.labelsdir) == False):
            print("Labels directory does not exist")
            exit()

    if(os.path.exists(args.path) == False):
        print("Path to exec files does not exist")
        exit()

    if(os.path.exists(args.img_dir) == False):
        print("Image directory does not exist")
        exit()

    if(os.path.exists(args.markers_dir) == False):
        print("Image markers directory does not exist")
        exit()

    if(args.layer < 1):
        print("Layer must be greater than 0")
        exit()

    if(args.final_spx < 1):
        print("Final superpixels must be greater than 0")
        exit()

    if(args.spx_alg == "DISF"):
        if(args.init_seeds < 1):
            print("Initial seeds must be greater than 0")
            exit()
        if(args.final_spx > args.init_seeds):
            print("Final superpixels must be smaller than initial seeds")
            exit()

def rmdir(path):
    if(os.path.exists(path)):
        shutil.rmtree(path, ignore_errors=True)

def CallFunction(function, args, msg=None):
    if(msg is not None):
        sys.stdout.write(msg)
        sys.stdout.flush()
    os.system("{} {}".format(function, " ".join([str(i) for i in args])))
    #print("{} {}".format(function, " ".join([str(i) for i in args])))


args = parser.parse_args()
print("Model directory: {}".format(args.model_dir))
#print("Arguments: {}".format(args))
functions = setFunctions(args.path)

if(args.layer == 1):
    if(args.arch_file == ""):
        print("Arch file is required for layer 1")
        exit()

    rmdir(args.model_dir)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs("{}/model".format(args.model_dir), exist_ok=True)
    shutil.copyfile(args.arch_file, "{}/model/arch.json".format(args.model_dir))

else:
    if(args.arch_file == ""):
        args.arch_file = "{}/model/arch.json".format(args.model_dir)
        if(os.path.isfile(args.arch_file) == False):
            print("Arch file does not exist")
            exit()
    else:
        shutil.copyfile(args.arch_file, "{}/model/arch.json".format(args.model_dir))

verifyArgs(args)

if(args.layer == 1):

    if(args.graphdir == "" and args.labelsdir == ""):

        with open(args.arch_file) as f:
            data = json.load(f)
            kernel_size = data['layer1']['conv']['kernel_size'][0]*data['layer1']['conv']['kernel_size'][1]
            nfeat_markers = data['layer1']['conv']["nkernels_per_marker"]

        CallFunction(functions['CreateGraph'], [args.img_dir, args.final_spx, args.model_dir, args.graph_feat, args.spx_alg, args.init_seeds], 'Creating graphs:\t\t')

    else:
        if(args.graphdir == "" and args.labelsdir != ""):
            print("Graph directory is required")
            exit()
        if(args.graphdir != "" and args.labelsdir == ""):
            print("Labels directory is required")
            exit()
        shutil.copytree(args.graphdir, "{}/layer0".format(args.model_dir), dirs_exist_ok=True)
        shutil.copytree(args.labelsdir, "{}/labels0".format(args.model_dir), dirs_exist_ok=True)
        

    if(args.reduce_kernels > 0):
        CallFunction(functions['ConvertSeeds2'], [args.model_dir, args.markers_dir, nfeat_markers, kernel_size, args.reduce_kernels, args.attention, args.aggregation], 'Converting seeds:\t\t')
    else:
        CallFunction(functions['ConvertSeeds'], [args.model_dir, args.markers_dir, nfeat_markers, kernel_size, args.attention, args.aggregation], 'Converting seeds:\t\t')
    

CallFunction(functions['CreateLayerModel'], [args.model_dir, args.arch_file, args.layer, args.stride, args.attention, args.aggregation], "Creating weights layer {}:\t".format(args.layer))
CallFunction(functions['MergeLayerModels'], [args.model_dir, args.layer], "Merging weights layer {}:\t".format(args.layer))
CallFunction(functions['EncodeMergedLayer'], [args.model_dir, args.arch_file, args.layer, args.stride, args.attention, args.aggregation, 1 if args.dropout else 0], "Encoding layer {}:\t\t".format(args.layer))

if(args.activation):
    CallFunction(functions['WriteActivations'], [args.model_dir, args.layer], "Writing activations layer {}:\t".format(args.layer))
CallFunction(functions['DecodeLayer'], [args.model_dir, args.layer, args.arch_file, args.decoder, args.stride], "Decoding layer {}:\t\t".format(args.layer))

#for layer in range(0,args.nlayers+1):
#    line = "{}/layer{}".format(args.model_dir, layer)
#    rmdir(line)
#    line = "{}/labels{}".format(args.model_dir, layer)
#    rmdir(line)

