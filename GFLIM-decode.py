import os
import shutil
import argparse
import sys

parser = argparse.ArgumentParser(
    prog="GFLIM-decoder",
    description="Produce saliency maps using a model trained with GFLIM or GFLIM-BoFP",
    epilog="Thanks for using %(prog)s! :)",
)

parser.add_argument("arch_file", help="Arch file", type=str)
parser.add_argument("layer", help="Layer number", type=int)
parser.add_argument("model_dir", help="Model directory", type=str)
parser.add_argument("stride", help="Stride type", type=str, choices=["none", "image-based", "graph-based"])
parser.add_argument("decoder", help="Decoder type.", type=str, choices=["tristate", "hybrid", "graph-adaptive", "gaussian", "test", 'mean'])

parser.add_argument("-path", "--path", help=" ", type=str, default="../bin")


def verifyArgs(args):
    if(os.path.exists(args.path) == False):
        print("Path to exec files does not exist")
        exit()

    if(os.path.exists(args.model_dir) == False):
        print("Model directory does not exist")
        exit()

    if(args.layer < 1):
        print("Layer must be greater than 0")
        exit()
    
    if(os.path.isfile(args.arch_file) == False):
        print("Arch file does not exist")
        exit()


args = parser.parse_args()
print("Model directory: {}".format(args.model_dir))
#print("Arguments: {}".format(args))
verifyArgs(args)


sys.stdout.write("Decoding layer {}:\t\t".format(args.layer))
sys.stdout.flush()
vals = [args.model_dir, args.layer, args.arch_file, args.decoder, args.stride]
line = "{} {}".format("{}/iftGFLIMDecodeLayer".format(args.path), " ".join([str(i) for i in vals]))
os.system(line)



