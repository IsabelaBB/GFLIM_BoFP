import os
import shutil
import argparse

path_to_exec = "/home/isabela/Codes/svnift/trunk/bin"
if(os.path.exists(path_to_exec) == False):
    print("Path to exec files does not exist")
    exit()

functions = {
    'CreateGraph': '{}/iftCreateGFLIMGraph'.format(path_to_exec),
    'ConvertSeeds': '{}/iftGFLIMConvertSeeds'.format(path_to_exec),
    'TrainLayer': '{}/iftGFLIMTrainLayer'.format(path_to_exec),
    'DecodeLayer': '{}/iftGFLIMDecodeLayer'.format(path_to_exec),
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
parser.add_argument("final_spx", help="Final superpixels", type=int)
parser.add_argument("init_seeds", help="Initial number of seeds in DISF", type=int, default=0)

args = parser.parse_args()
print("Arguments: {}".format(args))

rmdir(args.model_dir)
verifyPath(args)

os.makedirs(args.model_dir, exist_ok=True)

CallFunction(functions['CreateGraph'], [args.img_dir, args.final_spx, args.model_dir, args.graph_feat, args.spx_alg, args.init_seeds], 'Create the graphs')
CallFunction(functions['ConvertSeeds'], [args.model_dir, args.markers_dir, -1], 'Convert the seed from image to graph')

for layer in range(1,args.nlayers+1):
    CallFunction(functions['TrainLayer'], [args.model_dir, args.arch_file, layer], 'Train the layer {}'.format(layer))
    CallFunction(functions['DecodeLayer'], [args.model_dir, layer, args.arch_file, 'tristate', 'none'], 'Decode the layer {}'.format(layer))

for layer in range(1,args.nlayers+1):
    line = "{}/layer{}".format(args.model_dir, layer-1)
    rmdir(line)

line = "{}/labels0".format(args.model_dir)
rmdir(line)

