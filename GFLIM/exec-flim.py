import os
import sys
import shutil
import json

if (len(sys.argv) != 8):
    print("python exec-flim.py <P1> <P2> <P3> <P4> <P5> <P6> <P7>")
    print("P1: Image directory")
    print("P2: Image markers directory")
    print("P3: Arch file")
    print("P4: Number of layers ")
    print("P5: Model directory")
    print("P6: Initial seeds")
    print("P7: Final superpixels")
    exit()

image_folder = sys.argv[1]
markers_folder = sys.argv[2]
arch_file = sys.argv[3]
nlayers = int(sys.argv[4])
model_folder = sys.argv[5]
init_seeds = int(sys.argv[6])
final_spx = int(sys.argv[7])

if(os.path.isfile(arch_file) == False):
    print("Arch file does not exist")
    exit()

path_to_exec = "/home/isabela/Codes/svnift/trunk/bin"
if(os.path.exists(path_to_exec) == False):
    print("Path to exec files does not exist")
    exit()
    
os.makedirs(model_folder, exist_ok=True)

line = "{}/iftCreateGFLIMGraph {} {} {} {}".format(path_to_exec, image_folder, init_seeds, final_spx, model_folder)
os.system(line)

line = "{}/iftImageToGraphSeeds {} {}".format(path_to_exec, model_folder, markers_folder)
os.system(line)

for layer in range(1,nlayers+1):
    print("Layer {}".format(layer))
    print("\t Training layer ")
    line = "{}/iftGFLIMTrainLayer {} {} {}".format(path_to_exec, model_folder, arch_file, layer)
    os.system(line)
    
    print("\t Decoding layer ")
    line = "{}/iftGFLIMDecodeLayer {} {} {}".format(path_to_exec, model_folder, layer, arch_file)
    os.system(line)

for layer in range(1,nlayers+1):
    line = "{}/layer{}".format(model_folder, layer-1)
    shutil.rmtree(line)

line = "{}/labels0".format(model_folder)
shutil.rmtree(line)

