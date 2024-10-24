import os
import sys
import shutil
    
if (len(sys.argv) != 8):
    print("python exec-flim.py <P1> <P2> <P3>")
    print("P1: Arch file")
    print("P2: Number of layers (if negative, do not encode layers again)")
    print("P3: Model directory")
    exit()

arch_file = sys.argv[1]
nlayers = int(sys.argv[2])
model_folder = sys.argv[3]

path_to_exec = "/home/isabela/Codes/svnift/trunk/bin"

for layer in range(1,nlayers+1):
    print("\t Decoding layer ")
    line = "{}/iftGFLIMISDecodeLayer {} {} {}".format(path_to_exec, model_folder, layer, arch_file)
    os.system(line)

for layer in range(1,nlayers+1):
    line = "{}/layer{}".format(model_folder, layer-1)
    shutil.rmtree(line)
    line = "{}/labels{}".format(model_folder, layer-1)
    shutil.rmtree(line)

