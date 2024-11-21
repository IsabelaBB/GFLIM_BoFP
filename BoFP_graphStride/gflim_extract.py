import os
import sys
import shutil

if (len(sys.argv) != 7):
    print("python exec-flim.py <P1> <P2> <P3> <P4> <P5> <P6> ")
    print("P1: Image directory")
    print("P2: Arch file")
    print("P3: Number of layers")
    print("P4: Model directory")
    print("P5: Initial seeds")
    print("P6: Final superpixels")
    exit()

image_folder = sys.argv[1]
arch_file = sys.argv[2]
nlayers = int(sys.argv[3])
model_folder = sys.argv[4]
init_seeds = int(sys.argv[5])
final_spx = int(sys.argv[6])

path_to_exec = "/home/isabela/Codes/svnift/trunk/bin"
if(os.path.exists(path_to_exec) == False):
    print("Path to exec files does not exist")
    exit()
    
if(os.path.exists(model_folder) == False):
    print("Model folder does not exist")
    exit()

line = "{}/iftGFLIMGSExtract {} {} {} {} {} {}".format(path_to_exec, image_folder, model_folder, arch_file, nlayers, init_seeds, final_spx)
os.system(line)

for layer in range(1,nlayers+1):
    line = "{}/labels{}".format(model_folder, layer-1)
    if(os.path.exists(line)):
        shutil.rmtree(line)

