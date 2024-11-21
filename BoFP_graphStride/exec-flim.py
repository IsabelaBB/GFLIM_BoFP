import os
import shutil
import json
import argparse

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

    if(args.init_seeds < 1):
        print("Initial seeds must be greater than 0")
        exit()

    if(args.final_spx < 1 or args.final_spx > args.init_seeds):
        print("Final superpixels must be greater than 0 and smaller than initial seeds")
        exit()


def rmdir(path):
    if(os.path.exists(path)):
        shutil.rmtree(path, ignore_errors=True)

def deleteModel(args):
    rmdir("{}/model".format(args.model_dir))
    rmdir("{}/markers".format(args.model_dir))
    rmdir("{}/markers_img".format(args.model_dir))
    rmdir("{}/img_markers".format(args.model_dir))
    for layer in range(0,args.nlayers+1):
        rmdir("{}/layer{}".format(args.model_dir, layer))
        rmdir("{}/labels{}".format(args.model_dir, layer))
        rmdir("{}/activation{}".format(args.model_dir, layer))
        rmdir("{}/activation{}_before".format(args.model_dir, layer))
        rmdir("{}/saliency{}_before".format(args.model_dir, layer))


parser = argparse.ArgumentParser(
    prog="exec-gflim",
    description="Executes GFLIM for training from graph's creation to layer decoding",
    epilog="Thanks for using %(prog)s! :)",
)

parser.add_argument("img_dir", help="Image directory", type=str)
parser.add_argument("markers_dir", help="Image markers directory", type=str)
parser.add_argument("arch_file", help="Arch file", type=str)
parser.add_argument("nlayers", help="Number of layers (if negative, do not encode layers again)", type=int)
parser.add_argument("model_dir", help="Model directory", type=str)
parser.add_argument("init_seeds", help="Initial seeds", type=int)
parser.add_argument("final_spx", help="Final superpixels", type=int)
parser.add_argument("--kmeans", help="Optional: reduce seeds using kmeans", action="store_true")
parser.add_argument("-gt", "--gt_dir", help="Optional: ground truth directory to reduce layer depth", type=str)
parser.add_argument("-act", "--activation", help="Optional: Write activations", action="store_true")
args = parser.parse_args()

verifyPath(args)
deleteModel(args)

path_to_exec = "/home/isabela/Codes/svnift/trunk/bin"
if(os.path.exists(path_to_exec) == False):
    print("Path to exec files does not exist")
    exit()

if(args.kmeans):
    with open(args.arch_file) as f:
        data = json.load(f)
        kernel_size = data['layer1']['conv']['kernel_size'][0]*data['layer1']['conv']['kernel_size'][1]
        nfeat_background = data['layer1']['conv']["nkernels_per_marker"]
        nfeat_foreground = data['layer1']['conv']["nkernels_per_marker"]

os.makedirs(args.model_dir, exist_ok=True)

line = "{}/iftCreateGFLIMGraph {} {} {} {}".format(path_to_exec, args.img_dir, args.init_seeds, args.final_spx, args.model_dir)
os.system(line)

if(args.kmeans):
    line = "{}/iftReduceSeedSet {} {} {} {} {}".format(path_to_exec, args.model_dir, args.markers_dir, kernel_size, nfeat_background, nfeat_foreground)
    os.system(line)
    args.markers_dir = "{}/img_markers".format(args.model_dir) # Update markers folder when reducing seed set
    os.rename("{}/markers".format(args.model_dir), args.markers_dir)

for layer in range(1,args.nlayers+1):
    print("Layer {}".format(layer))
    print("\t Creating layer model ")
    line = "{}/iftGFLIMGSCreateLayerModel {} {} {} {}".format(path_to_exec, args.model_dir, args.arch_file, layer, args.markers_dir)
    os.system(line)
    
    print("\t Merging layer models ")
    line = "{}/iftGFLIMMergeLayerModels {} {}".format(path_to_exec, args.model_dir, layer)
    os.system(line)

    print("\t Encoding merged layer ")
    line = "{}/iftGFLIMGSEncodeMergedLayer {} {} {}".format(path_to_exec, args.model_dir, args.arch_file, layer)
    os.system(line)

    if(args.activation):
        print("\t Writing activations ")
        line = "{}/iftGFLIMWriteActivations {} {}".format(path_to_exec, args.model_dir, layer)
        os.system(line)

    if(args.gt_dir is not None):
        if(args.activation):
            os.rename("{}/activation{}".format(args.model_dir, layer), "{}/activation{}_before".format(args.model_dir, layer))
        
        print("\t Decoding layer ")
        line = "{}/iftGFLIMGSDecodeLayer {} {} {}".format(path_to_exec, args.model_dir, layer, args.arch_file)
        os.system(line)
        os.rename("{}/saliency{}".format(args.model_dir, layer), "{}/saliency{}_before".format(args.model_dir, layer))

        print("\t Reducing merged layer weights ")
        labels_dir = "{}/labels{}".format(args.model_dir, layer)
        line = "{}/iftReduceLayerWeights {} {} {} {}".format(path_to_exec, args.model_dir, layer, labels_dir, args.gt_dir)
        os.system(line)

        print("\t Encoding reduced merged layer ")
        line = "{}/iftGFLIMGSEncodeMergedLayer {} {} {}".format(path_to_exec, args.model_dir, args.arch_file, layer)
        os.system(line)

        if(args.activation):
            print("\t Writing activations ")
            line = "{}/iftGFLIMWriteActivations {} {}".format(path_to_exec, args.model_dir, layer)
            os.system(line)

    print("\t Decoding layer ")
    line = "{}/iftGFLIMGSDecodeLayer {} {} {}".format(path_to_exec, args.model_dir, layer, args.arch_file)
    os.system(line)

for layer in range(0,args.nlayers+1):
    line = "{}/layer{}".format(args.model_dir, layer)
    rmdir(line)
    line = "{}/labels{}".format(args.model_dir, layer)
    rmdir(line)

