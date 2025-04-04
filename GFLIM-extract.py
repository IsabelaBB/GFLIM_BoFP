import os
import shutil
import argparse

parser = argparse.ArgumentParser(
    prog="gflim extract",
    description="Produce saliency maps using a model trained by GFLIM",
    epilog="Thanks for using %(prog)s! :)",
)

def verifyPath(path):
    if(os.path.exists(args.model_dir) == False):
        print("Model directory does not exist")
        exit()

    if(os.path.exists(args.img_dir) == False):
        print("Image directory does not exist")
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


parser.add_argument("img_dir", help="Image directory", type=str)
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
parser.add_argument("batch_size", help="Batch size (<1 for automatic size)", type=int)
parser.add_argument("aggregation", help="Aggregation option.", type=str, choices=["CONCAT", "MEAN", "WMEAN"])

parser.add_argument("-init_seeds", "--init_seeds", help="Initial number of seeds in DISF. Required only if spx_alg is DISF", type=int, default=0)
parser.add_argument("-dropout", "--dropout", help=" ", action="store_true")

args = parser.parse_args()
print("Arguments: {}".format(args))
verifyPath(args)

path_to_exec = "/home/isabela/Codes/svnift/trunk/bin"
if(os.path.exists(path_to_exec) == False):
    print("Path to exec files does not exist")
    exit()

# alterar para ler as matrizes pca quando for o caso e usá-las nos grafos de entrada
vals = [args.img_dir, args.model_dir, args.arch_file, args.nlayers, args.graph_feat, args.spx_alg, args.final_spx, args.decoder, args.batch_size, args.stride, args.attention, args.aggregation, 1 if args.dropout else 0, args.init_seeds]
line = "{} {}".format("{}/iftGFLIMExtract".format(path_to_exec), " ".join([str(i) for i in vals]))
os.system(line)

for layer in range(1,args.nlayers+1):
    line = "{}/labels{}".format(args.model_dir, layer-1)
    if(os.path.exists(line)):
        shutil.rmtree(line)

