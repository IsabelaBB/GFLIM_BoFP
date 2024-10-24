import os
import sys
import shutil
    
if (len(sys.argv) != 8):
    print("python exec-flim.py <P1> <P2> <P3> <P4> <P5> <P6> <P7>")
    print("P1: Image directory")
    print("P2: Image markers directory")
    print("P3: Arch file")
    print("P4: Number of layers")
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

path_to_exec = "/home/isabela/Codes/svnift/trunk/bin"
os.makedirs(model_folder, exist_ok=True)

line = "{}/iftCreateGFLIMGraph {} {} {} {}".format(path_to_exec, image_folder, init_seeds, final_spx, model_folder)
os.system(line)

line = "{}/iftGFLIMFeatPoints {} {} {}".format(path_to_exec, markers_folder, model_folder)
os.system(line)

for layer in range(1,nlayers+1):
    print("Layer {}".format(layer))
    print("\t Creating layer model ")
    line = "{}/iftGFLIMCreateLayerModel {} {} {} {}".format(path_to_exec, model_folder, arch_file, layer, markers_folder)
    os.system(line)
    
    print("\t Merging layer models ")
    line = "{}/iftGFLIMMergeLayerModels {} {}".format(path_to_exec, model_folder, layer)
    os.system(line)

    print("\t Encoding merged layer ")
    line = "{}/iftGFLIMEncodeMergedLayer {} {} {}".format(path_to_exec, model_folder, arch_file, layer)
    os.system(line)


