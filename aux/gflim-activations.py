import os
import sys
import shutil

"""
for s in 1 2 3; do
        img_dir="../datasets/baseline_flim/schisto/image-splits2d/split${s}/orig/train/"
        markers_dir="../datasets/baseline_flim/schisto/markers/split${s}"
        for spx in 5000 4000 3000 2000 1000 8000 7000 6000; do
                model_dir="schisto/split${s}/stride-image_20000-${spx}/"
                python3 ../svnift/trunk/demo/GFLIM/aux/gflim-activations.py ${img_dir} ${markers_dir} arch1.json 3 ${model_dir} 20000 ${spx} imageStride
                model_dir="schisto/split${s}/noStride_20000-${spx}/"
                python3 ../svnift/trunk/demo/GFLIM/aux/gflim-activations.py ${img_dir} ${markers_dir} arch1_noStride.json 3 ${model_dir} 20000 ${spx} noStride
        done;
done;
"""

def validate_args(path_to_exec, gflim_type, model_folder, nlayers,):
    if(os.path.exists(path_to_exec) == False):
        print("Path to exec files does not exist")
        exit()

    if(gflim_type != "gflim" and gflim_type != "noStride" and gflim_type != "imageStride" and gflim_type != "graphStride"):
        print("Invalid gflim type {}".format(gflim_type))
        exit()

    if(os.path.exists(model_folder) == False): # Check if model folder exists
        print("Model folder does not exist: {}".format(model_folder))
        exit()
    elif(os.path.exists(model_folder+"/model") == False): # Check if model folder exists
        print("Model folder does not exist: {}/model".format(model_folder))
        exit()
    else: # Check if all layers were trained
        i = 1
        while(i <= nlayers):
            line = "{}/model/conv{}-kernels.npy".format(model_folder, i)
            if(os.path.exists(line) == False):
                print("File {} does not exist".format(line))
                exit()
            if(gflim_type != "gflim"):
                line = "{}/model/conv{}-bias.txt".format(model_folder, i)
                if(os.path.exists(line) == False):
                    print("File {} does not exist".format(line))
                    exit()
                line = "{}/model/conv{}-labels.txt".format(model_folder, i)
                if(os.path.exists(line) == False):
                    print("File {} does not exist".format(line))
                    exit()
            else:
                line = "{}/model/conv{}-mean.txt".format(model_folder, i)
                if(os.path.exists(line) == False):
                    print("File {} does not exist".format(line))
                    exit()
                line = "{}/model/conv{}-stdev.txt".format(model_folder, i)
                if(os.path.exists(line) == False):
                    print("File {} does not exist".format(line))
                    exit()
            i = i + 1

def prepare_gflim(path_to_exec, image_folder, init_seeds, final_spx, model_folder, markers_folder):
    line = "{}/iftCreateGFLIMGraph {} {} {} {}".format(path_to_exec, image_folder, init_seeds, final_spx, model_folder)
    os.system(line)
    line = "{}/iftImageToGraphSeeds {} {}".format(path_to_exec, model_folder, markers_folder)
    os.system(line)

def prepare_gflim_noStride(path_to_exec, image_folder, init_seeds, final_spx, model_folder, markers_folder):
    line = "{}/iftCreateGFLIMGraph {} {} {} {}".format(path_to_exec, image_folder, init_seeds, final_spx, model_folder)
    os.system(line)
    line = "{}/iftGFLIMFeatPoints {} {}".format(path_to_exec, markers_folder, model_folder)
    os.system(line)

def prepare_gflim_imageStride(path_to_exec, image_folder, init_seeds, final_spx, model_folder):
    line = "{}/iftCreateGFLIMGraph {} {} {} {}".format(path_to_exec, image_folder, init_seeds, final_spx, model_folder)
    os.system(line)

def prepare_gflim_graphStride(path_to_exec, image_folder, init_seeds, final_spx, model_folder):
    line = "{}/iftCreateGFLIMGraph {} {} {} {}".format(path_to_exec, image_folder, init_seeds, final_spx, model_folder)
    os.system(line)

if (len(sys.argv) != 9):
    print("python exec-flim.py <P1> <P2> <P3> <P4> <P5> <P6> <P7> <P8>")
    print("P1: Image directory")
    print("P2: Image markers directory")
    print("P3: Arch file")
    print("P4: Number of layers (if negative, do not encode layers again)")
    print("P5: Model directory")
    print("P6: Initial seeds")
    print("P7: Final superpixels")
    print("P8: gflim type (gflim, noStride, imageStride, graphStride)")
    exit()

image_folder = sys.argv[1]
markers_folder = sys.argv[2]
arch_file = sys.argv[3]
nlayers = int(sys.argv[4])
model_folder = sys.argv[5]
init_seeds = int(sys.argv[6])
final_spx = int(sys.argv[7])
gflim_type = sys.argv[8]

path_to_exec = "/home/isabela/Codes/svnift/trunk/bin"

validate_args(path_to_exec, gflim_type, model_folder, nlayers)

if(gflim_type == "gflim"):
    prepare_gflim(path_to_exec, image_folder, init_seeds, final_spx, model_folder, markers_folder)
elif(gflim_type == "noStride"):
    prepare_gflim_noStride(path_to_exec, image_folder, init_seeds, final_spx, model_folder, markers_folder)
elif(gflim_type == "imageStride"):
    prepare_gflim_imageStride(path_to_exec, image_folder, init_seeds, final_spx, model_folder)
else: # graphStride
    prepare_gflim_graphStride(path_to_exec, image_folder, init_seeds, final_spx, model_folder)

for layer in range(1,nlayers+1):
    print("Layer {}".format(layer))

    print("\t Encoding merged layer ")
    if(gflim_type == "noStride"):
        line = "{}/iftGFLIMEncodeMergedLayer {} {} {}".format(path_to_exec, model_folder, arch_file, layer)
    elif(gflim_type == "imageStride"):
        line = "{}/iftGFLIMISEncodeMergedLayer {} {} {}".format(path_to_exec, model_folder, arch_file, layer)
    else:
        line = "{}/iftGFLIMGSEncodeMergedLayer {} {} {}".format(path_to_exec, model_folder, arch_file, layer)
    os.system(line)

    print("\tWrite activations")
    line = "{}/iftGFLIMWriteActivations {} {}".format(path_to_exec, model_folder, layer)
    os.system(line)

# Remove temporary files
for layer in range(0,nlayers+1):
    line = "{}/layer{}".format(model_folder, layer)
    shutil.rmtree(line)
    line = "{}/labels{}".format(model_folder, layer)
    if(os.path.exists(line)):
        shutil.rmtree(line)

