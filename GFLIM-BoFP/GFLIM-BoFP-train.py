import os
import shutil
import json
import argparse

path_to_exec = "/home/isabela/Codes/svnift/trunk/bin"
if(os.path.exists(path_to_exec) == False):
    print("Path to exec files does not exist")
    exit()

functions = {
    'CreateGraph': '{}/iftCreateGFLIMGraph'.format(path_to_exec),
    'ReduceSeeds': '{}/iftReduceSeedSet'.format(path_to_exec),
    'ReduceSeeds2': '{}/iftReduceSeedSet_v2'.format(path_to_exec),
    'CreateLayerModel': '{}/iftGFLIMCreateLayerModel'.format(path_to_exec),
    'EncodeLayer': '{}/iftGFLIMEncodeLayer'.format(path_to_exec),
    'WriteActivations': '{}/iftGFLIMWriteActivations'.format(path_to_exec),
    'MergeLayerModels': '{}/iftGFLIMMergeLayerModels'.format(path_to_exec),
    'EncodeMergedLayer': '{}/iftGFLIMEncodeMergedLayer'.format(path_to_exec),
    'DecodeLayer': '{}/iftGFLIMDecodeLayer'.format(path_to_exec)
}

def verifyPath(path):
    if(os.path.exists(args.img_dir) == False):
        print("Image directory does not exist")
        exit()

    if(os.path.exists(args.markers_dir) == False):
        print("Image markers directory does not exist")
        exit()

    if(os.path.isfile(args.arch_file) == False):
        print("Arch file does not exist")
        exit()

    if(args.nlayers < 1):
        print("Number of layers must be greater than 0")
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
        print(msg)
    os.system("{} {}".format(function, " ".join([str(i) for i in args])))
    #print("{} {}".format(function, " ".join([str(i) for i in args])))

parser = argparse.ArgumentParser(
    prog="exec-gflim",
    description="Train a model using GFLIM",
    epilog="Thanks for using %(prog)s! :)",
)

parser.add_argument("img_dir", help="Image directory", type=str)
parser.add_argument("markers_dir", help="Image markers directory", type=str)
parser.add_argument("arch_file", help="Arch file", type=str)
parser.add_argument("nlayers", help="Number of layers (if negative, do not encode layers again)", type=int)
parser.add_argument("model_dir", help="Model directory", type=str)
parser.add_argument("stride", help="Stride type", type=str, choices=["none", "image-based", "graph-based"])
parser.add_argument("final_spx", help="Final superpixels", type=int)
parser.add_argument("graph_feat", help="Graph feature (see iftGFLIMGraph.h for more information)", type=str, choices=["MEAN", "SUM", 
                                                                                                                      "SUM_MULT_RELSIZE", "MEAN_MULT_RELSIZE", 
                                                                                                                      "MEAN_CONCAT_RELSIZE", "MEAN_CONCAT_RELSIZE_NEIGHB", "MEAN_CONCAT_RELSIZE_HARALICK", 
                                                                                                                      "SUM_MULT_RELSIZE_CONCAT_HARALICK", "SUM_CONCAT_HARALICK"])
parser.add_argument("spx_alg", help="Superpixel algorithm.", type=str, choices=["DISF", "ERGC"])
parser.add_argument("decoder", help="Decoder type.", type=str, choices=["tristate", "hybrid", "graph-adaptive", "gaussian"])
parser.add_argument("attention", help="Attention option.", type=str, choices=["NONE", "NODE_RELSIZE", "NODE_SIZE", "NEIGHB_SIMIL", "NEIGHB_SIZE", 
                                                                                "CLASS_DOMINANCE", "CLASS_DOMINANCE_WINNER", "CLASS_DOMINANCE_WINNER_POWER2", "CLASS_DOMINANCE_SOFTMAX_NEIGHB",
                                                                                "COSINE_SIMILARITY"])
parser.add_argument("aggregation", help="Aggregation option.", type=str, choices=["CONCAT", "MEAN", "WMEAN"])


parser.add_argument("-init_seeds", "--init_seeds", help="Initial number of seeds in DISF. Required only if spx_alg is DISF", type=int, default=0)
parser.add_argument("-act", "--activation", help="Write activations", action="store_true")
parser.add_argument("-reduce_kernels", "--reduce_kernels", help="Use another k-means to reduce the number of kernels to 64 per layer maximum.", type=int, default=0)
parser.add_argument("-dropout", "--dropout", help=" ", action="store_true")

args = parser.parse_args()
print("Arguments: {}".format(args))

rmdir(args.model_dir)
verifyPath(args)

if(args.kmeans):
    with open(args.arch_file) as f:
        data = json.load(f)
        kernel_size = data['layer1']['conv']['kernel_size'][0]*data['layer1']['conv']['kernel_size'][1]
        nfeat_markers = data['layer1']['conv']["nkernels_per_marker"]

os.makedirs(args.model_dir, exist_ok=True)

CallFunction(functions['CreateGraph'], [args.img_dir, args.final_spx, args.model_dir, args.graph_feat, args.spx_alg, args.init_seeds], 'CreateGraph')

# alterar os arquivos para executar pca para obj e bkg e salvar as matrizes pca em arquivo
if(args.kmeans):
    if(args.reduce_kernels > 0):
        CallFunction(functions['ReduceSeeds2'], [args.model_dir, args.markers_dir, kernel_size, nfeat_markers, args.attention, args.aggregation, args.reduce_kernels], 'ReduceSeeds')
    else:
        CallFunction(functions['ReduceSeeds'], [args.model_dir, args.markers_dir, kernel_size, nfeat_markers, args.attention, args.aggregation], 'ReduceSeeds')

for layer in range(1,args.nlayers+1):
    print("Layer {}".format(layer))

    CallFunction(functions['CreateLayerModel'], [args.model_dir, args.arch_file, layer, args.stride, args.attention, args.aggregation], "\t Creating layer model ")
    CallFunction(functions['MergeLayerModels'], [args.model_dir, layer], "\t Merging layer models ")
    CallFunction(functions['EncodeMergedLayer'], [args.model_dir, args.arch_file, layer, args.stride, args.attention, args.aggregation, 1 if args.dropout else 0], "\t Encoding merged layer ")

    if(args.activation):
        CallFunction(functions['WriteActivations'], [args.model_dir, layer], "\t Writing activations ")
    CallFunction(functions['DecodeLayer'], [args.model_dir, layer, args.arch_file, args.decoder, args.stride], "\t Decoding layer ")

for layer in range(0,args.nlayers+1):
    line = "{}/layer{}".format(args.model_dir, layer)
    rmdir(line)
    line = "{}/labels{}".format(args.model_dir, layer)
    rmdir(line)

