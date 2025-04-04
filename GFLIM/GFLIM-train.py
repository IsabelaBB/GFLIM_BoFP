import os
import shutil
import argparse

parser = argparse.ArgumentParser(
    prog="GFLIM-train",
    description="Train a model using GFLIM",
    epilog="Thanks for using %(prog)s! :)",
)

parser.add_argument("img_dir", help="Image directory", type=str)
parser.add_argument("markers_dir", help="Image markers directory", type=str)
parser.add_argument("arch_file", help="Arch file", type=str)
parser.add_argument("nlayers", help="Number of layers (if negative, do not encode layers again)", type=int)
parser.add_argument("model_dir", help="Model directory", type=str)
parser.add_argument("final_spx", help="Final superpixels", type=int)
parser.add_argument("init_seeds", help="Initial number of seeds in DISF", type=int, default=0)
parser.add_argument("-act", "--activation", help="Write activations", action="store_true")
parser.add_argument("-path", "--path", help=" ", type=str, default="~/Codes/LibIFT")

def setFunctions(path):
    return {
        'CreateGraph': '{}/iftCreateGFLIMGraph'.format(path),
        'ConvertSeeds': '{}/iftGFLIMConvertSeeds'.format(path),
        'TrainLayer': '{}/iftGFLIMTrainLayer'.format(path),
        'DecodeLayer': '{}/iftGFLIMDecodeLayer'.format(path),
    }

def verifyArgs(args):
    if(os.path.exists(args.path) == False):
        print("Path to exec files does not exist")
        exit()

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


args = parser.parse_args()
print("Arguments: {}".format(args))
functions = setFunctions(args.path)
rmdir(args.model_dir)
verifyArgs(args)

os.makedirs(args.model_dir, exist_ok=True)

CallFunction(functions['CreateGraph'], [args.img_dir, args.final_spx, args.model_dir, args.graph_feat, args.spx_alg, args.init_seeds], 'Create the graphs')
CallFunction(functions['ConvertSeeds'], [args.model_dir, args.markers_dir, -1], 'Convert seeds from image to graph')

for layer in range(1,args.nlayers+1):
    CallFunction(functions['TrainLayer'], [args.model_dir, args.arch_file, layer], 'Train layer {}'.format(layer))
    if(args.activation):
        CallFunction(functions['WriteActivations'], [args.model_dir, layer], "\t Write activations ")
    CallFunction(functions['DecodeLayer'], [args.model_dir, layer, args.arch_file, 'tristate', 'none'], 'Decode layer {}'.format(layer))

for layer in range(1,args.nlayers+1):
    line = "{}/layer{}".format(args.model_dir, layer-1)
    rmdir(line)

line = "{}/labels0".format(args.model_dir)
rmdir(line)

