import os
import shutil
import json
import argparse
import sys
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
import tkinter.messagebox
import numpy as np
from PIL import Image, ImageTk


def rmdir(path):
    if(os.path.exists(path)):
        shutil.rmtree(path, ignore_errors=True)

# Set up the functions to be called
def CallFunction(function, args, msg=None):
    if(msg is not None):
        sys.stdout.write(msg)
        sys.stdout.flush()
    print("{} {}".format(function, " ".join([str(i) for i in args])))
    os.system("{} {}".format(function, " ".join([str(i) for i in args])))


class GFLIMModel:
    def __init__(self, dataset_name, markers_name, arch_file, split, img_set,
                 stride, attention, aggregation, 
                 nsuperpixels, graph_feats, superpixel_alg, init_seeds):
        
        path = "/home/isabela/Codes/LibIFT/bin"
        self.functions = {
            'CreateGraph': '{}/iftCreateGFLIMGraph'.format(path),
            'ConvertSeeds': '{}/iftGFLIMConvertSeeds'.format(path),
            'ConvertSeeds2': '{}/iftGFLIMConvertSeeds_v2'.format(path),
            'CreateLayerModel': '{}/iftGFLIMCreateLayerModel'.format(path),
            'EncodeLayer': '{}/iftGFLIMEncodeLayer'.format(path),
            'WriteActivations': '{}/iftGFLIMWriteActivations'.format(path),
            'MergeLayerModels': '{}/iftGFLIMMergeLayerModels'.format(path),
            'EncodeMergedLayer': '{}/iftGFLIMEncodeMergedLayer'.format(path),
            'DecodeLayer': '{}/iftGFLIMDecodeLayer'.format(path),
            'SelectKernels': '{}/iftGFLIMSelectKernels'.format(path)
        }

        self.dataset_name = dataset_name
        self.markers_name = markers_name
        self.arch_file = arch_file
        self.img_set = img_set
        self.split = split
        self.layer = 1
        self.stride = stride
        self.attention = attention
        self.aggregation = aggregation
        self.nsuperpixels = nsuperpixels
        self.graph_feats = graph_feats
        self.superpixel_alg = superpixel_alg
        self.init_seeds = init_seeds
        
        self.readytoRun = False
        self.setupDirectories()
    
    def trainLayer(self, layer, img_set, markers_dir):
        if(layer == 1):

            if(img_set != self.img_set and layer == 1):
                self.img_set = img_set
                self.setupDirectories()
                rmdir("{}/layer0".format(self.model_dir))
                rmdir("{}/labels0".format(self.model_dir))
                self.createGraphs()
                
            elif (os.path.exists("{}/layer0".format(self.model_dir)) == False or 
                os.path.exists("{}/labels0".format(self.model_dir)) == False):
                self.createGraphs()

        elif(os.path.exists("{}/layer{}".format(self.model_dir, self.layer-1)) == False or 
                os.path.exists("{}/labels{}".format(self.model_dir, self.layer-1)) == False):
            print("Layer {} does not exist".format(self.layer-1))
            exit()

        self.convertSeeds(markers_dir)
        self.createModel(layer)
        self.mergeLayerModels(layer)
        self.encodeMergedLayer(layer)

    def trainLayer_noMerge(self, layer, img_set, markers_dir):
        print("called trainLayer_noMerge with args: ", [layer, img_set, markers_dir])

        if self.readytoRun == False:
            tkinter.messagebox.showerror("Error", "Model not configured / invalid. Please set the model configuration.")
            return

        if(layer == 1):
            if os.path.exists(self.model_dir) == False:
                os.makedirs("{}/model".format(self.model_dir), exist_ok=True)
                shutil.copyfile(self.arch_file, "{}/model/arch.json".format(self.model_dir))

            if(img_set != self.img_set):
                self.img_set = img_set
                self.setupDirectories()
                rmdir("{}/layer0".format(self.model_dir))
                rmdir("{}/labels0".format(self.model_dir))
                self.createGraphs()

            elif (os.path.exists("{}/layer0".format(self.model_dir)) == False or 
                os.path.exists("{}/labels0".format(self.model_dir)) == False):
                self.createGraphs()
            else:
                layer0_list = os.listdir("{}/layer0".format(self.model_dir))
                labels0_list = os.listdir("{}/labels0".format(self.model_dir))
                create_graphs = False
                for img_name in os.listdir(markers_dir):
                    img_name = img_name.split("-")[0]
                    print("img_name: ", img_name+".json")
                    if img_name+".json" not in layer0_list or img_name+".pgm" not in labels0_list:
                        print("Layer 0 does not exist for image {}".format(img_name))
                        create_graphs = True
                        break
                if create_graphs:
                    rmdir("{}/layer0".format(self.model_dir))
                    rmdir("{}/labels0".format(self.model_dir))
                    self.createGraphs()

        elif(os.path.exists("{}/layer{}".format(self.model_dir, self.layer-1)) == False or 
                os.path.exists("{}/labels{}".format(self.model_dir, self.layer-1)) == False):
            print("Layer {} does not exist".format(self.layer-1))
            exit()

        self.convertSeeds(markers_dir)
        self.createModel(layer)
        self.encodeLayer(layer)

    def setupDirectories(self):
        if self.stride == "image-based":
            stride_name = "imageStride"
        elif self.stride == "graph-based":
            stride_name = "graphStride"
        else:
            stride_name = self.stride

        aggregation_name = ""
        if aggregation != "CONCAT":
            aggregation_name = "-{}".format(self.aggregation)

        dataset_dir = "/home/isabela/Codes/LibIFT/Datasets/{}/".format(self.dataset_name)
        self.orig_img_markers_dir = "{}/markers/{}/{}/".format(dataset_dir, self.markers_name, self.split)
        self.img_dir = "{}/splits/{}/orig/{}".format(dataset_dir, self.split, self.img_set)
        self.model_dir = "/home/isabela/Codes/LibIFT/Results/{}/{}/gt_markers_erosion5_interative/gflim-BoFP/{}/{}/{}-{}/kmeans{}/{}-{}".format(
            self.dataset_name, self.split, stride_name, self.superpixel_alg, self.init_seeds, self.nsuperpixels, aggregation_name, self.graph_feats, self.attention)
        self.model_img_markers_dir = "{}/markers_img/".format(self.model_dir)

        self.readytoRun = True
        if os.path.exists(self.orig_img_markers_dir) == False:
            tkinter.messagebox.showerror("Error", "Markers directory does not exist: {}".format(self.orig_img_markers_dir))
            self.readytoRun = False
        if os.path.exists(self.img_dir) == False:
            tkinter.messagebox.showerror("Error", "Image directory does not exist: {}".format(self.img_dir))
            self.readytoRun = False
        if os.path.isfile(self.arch_file) == False:
            tkinter.messagebox.showerror("Error", "Arch file does not exist: {}".format(self.arch_file))
            self.readytoRun = False

            

    # basic calls
    def createGraphs(self):
        CallFunction(self.functions['CreateGraph'], [self.img_dir, self.nsuperpixels, self.model_dir, self.graph_feats, self.superpixel_alg, self.init_seeds], 'Creating graphs:\t\t')

    def convertSeeds(self, markers_dir):
        with open(self.arch_file) as f:
            data = json.load(f)
            kernel_size = data['layer1']['conv']['kernel_size'][0]*data['layer1']['conv']['kernel_size'][1]
            nfeat_markers = data['layer1']['conv']["nkernels_per_marker"]
        CallFunction(self.functions['ConvertSeeds'], [self.model_dir, markers_dir, nfeat_markers, kernel_size, self.attention, self.aggregation], 'Converting seeds:\t\t')

    def createModel(self, layer):
        CallFunction(self.functions['CreateLayerModel'], [self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation], "Creating weights layer {}:\t".format(layer))

    def encodeLayer(self, layer):
        CallFunction(self.functions['EncodeLayer'], [self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation], "Encoding non merger layer {}:\t\t".format(layer))
    
    def mergeLayerModels(self, layer):
        CallFunction(self.functions['MergeLayerModels'], [self.model_dir, layer], "Merging weights layer {}:\t".format(layer))
    
    def encodeMergedLayer(self, layer):
        CallFunction(self.functions['EncodeMergedLayer'], [self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation, 0], "Encoding layer {}:\t\t".format(layer))

    def writeActivations(self, layer):
        CallFunction(self.functions['WriteActivations'], [self.model_dir, layer], "Writing activations layer {}:\t".format(layer))

    def selectKernels(self, layer, img_markers_dir, kernels_list, img_name):
        print("função selectKernels, args: ", [self.model_dir, img_markers_dir, layer, ",".join([str(k) for k in kernels_list]), img_name])
        CallFunction(self.functions['SelectKernels'], [self.model_dir, img_markers_dir, ",".join([str(k) for k in kernels_list]), img_name], "Selecting kernels layer {}:\t".format(layer))
        
class ConfigModel(tk.Toplevel):
    def __init__(self, root, dataset_name, markers_name, arch_file, split, img_set, stride, attention, aggregation, nsuperpixels, graph_feats, superpixel_alg, init_seeds, fn_callback):
        super().__init__(root)
        self.root = self
        self.root.title("Model Configuration")

        self.dataset_name = dataset_name
        self.markers_name = markers_name
        self.nsuperpixels = nsuperpixels
        self.init_seeds = init_seeds

        self.arch_file_var = tk.StringVar()
        if arch_file is not None:
            self.arch_file_var.set(arch_file)
        self.split_var = tk.StringVar()
        if split is not None:
            self.split_var.set(split)
        self.img_set_var = tk.StringVar()
        if img_set is not None:
            self.img_set_var.set(img_set)
        self.stride_var = tk.StringVar()
        if stride is not None:
            self.stride_var.set(stride)
        self.attention_var = tk.StringVar()
        if attention is not None:
            self.attention_var.set(attention)
        self.aggregation_var = tk.StringVar()
        if aggregation is not None:
            self.aggregation_var.set(aggregation)
        self.graph_feats_var = tk.StringVar()
        if graph_feats is not None:
            self.graph_feats_var.set(graph_feats)
        self.superpixel_alg_var = tk.StringVar()
        if superpixel_alg is not None:
            self.superpixel_alg_var.set(superpixel_alg)
        self.init_seeds_var = tk.StringVar()
        if init_seeds is not None:
            self.init_seeds_var.set(init_seeds)
        
        self.fn = fn_callback

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        frame = tk.Frame(main_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Dataset name")
        label.pack(side=tk.LEFT)
        self.dataset_entry = tk.Entry(frame)
        self.dataset_entry.pack(side=tk.LEFT)
        self.dataset_entry.insert(0, str(self.dataset_name))

        frame = tk.Frame(main_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Markers name")
        label.pack(side=tk.LEFT)
        self.markers_entry = tk.Entry(frame)
        self.markers_entry.pack(side=tk.LEFT)
        self.markers_entry.insert(0, str(self.markers_name))

        frame = tk.Frame(main_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Arch file")
        label.pack(side=tk.LEFT)
        entry = tk.Entry(frame, width=15, textvariable=self.arch_file_var, state=tk.DISABLED, relief=tk.SUNKEN)
        entry.insert(0, str(self.arch_file_var.get()))
        entry.pack(side=tk.LEFT)
        button = tk.Button(frame, text="...", command=self.load_arch_file, width=1, height=1)
        button.pack(side=tk.LEFT)

        frame = tk.Frame(main_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Split")
        label.pack(side=tk.LEFT)
        self.split_combobox = ttk.Combobox(frame, state="readonly", width=10, textvariable=self.split_var)
        self.split_combobox['values'] = ["split"+str(l+1) for l in range(5)]
        self.split_combobox.pack(side=tk.LEFT)
        if(self.split_var.get() in self.split_combobox['values']):
            self.split_combobox.current(self.split_combobox['values'].index(self.split_var.get()))

        frame = tk.Frame(main_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Image set")
        label.pack(side=tk.LEFT)
        combobox = ttk.Combobox(frame, state="readonly", width=10, textvariable=self.img_set_var, values=["train", "val", "val-train", "test"])
        combobox.pack(side=tk.LEFT)
        if(self.img_set_var.get() in combobox['values']):
            combobox.current(combobox['values'].index(self.img_set_var.get()))

        frame = tk.Frame(main_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Stride")
        label.pack(side=tk.LEFT)
        self.stride_combobox = ttk.Combobox(frame, state="readonly", width=10, textvariable=self.stride_var)
        self.stride_combobox.pack(side=tk.LEFT)
        self.stride_combobox['values'] = ["none", "image-based", "graph-based"]
        if(self.stride_var.get() in self.stride_combobox['values']):
            self.stride_combobox.current(self.stride_combobox['values'].index(self.stride_var.get()))

        frame = tk.Frame(main_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Attention")
        label.pack(side=tk.LEFT)
        self.attention_combobox = ttk.Combobox(frame, state="readonly", width=10, textvariable=self.attention_var)
        self.attention_combobox.pack(side=tk.LEFT)
        self.attention_combobox['values'] = ["NONE", "CLASS_DOMINANCE_TEST", "CLASS_DOMINANCE", "CLASS_DOMINANCE_POWER2"]
        if(self.attention_var.get() in self.attention_combobox['values']):
            self.attention_combobox.current(self.attention_combobox['values'].index(self.attention_var.get()))

        frame = tk.Frame(main_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Aggregation")
        label.pack(side=tk.LEFT)
        self.aggregation_combobox = ttk.Combobox(frame, state="readonly", width=10, textvariable=self.aggregation_var)
        self.aggregation_combobox.pack(side=tk.LEFT)
        self.aggregation_combobox['values'] = ["CONCAT", "MEAN", "WMEAN"]
        if(self.aggregation_var.get() in self.aggregation_combobox['values']):
            self.aggregation_combobox.current(self.aggregation_combobox['values'].index(self.aggregation_var.get()))

        frame = tk.Frame(main_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Superpixels")
        label.pack(side=tk.LEFT)
        self.nsuperpixels_entry = tk.Entry(frame, width=10)
        self.nsuperpixels_entry.pack(side=tk.LEFT)
        self.nsuperpixels_entry.insert(0, str(self.nsuperpixels))

        frame = tk.Frame(main_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Graph feats")
        label.pack(side=tk.LEFT)
        self.graph_feats_combobox = ttk.Combobox(frame, state="readonly", width=10)
        self.graph_feats_combobox.pack(side=tk.LEFT)
        self.graph_feats_combobox['values'] = ["SUM", "MEAN", "SUM_MULT_RELSIZE", "MEAN_MULT_RELSIZE",
                                                "MEAN_CONCAT_RELSIZE", "MEAN_CONCAT_RELSIZE_NEIGHB", 
                                                "MEAN_CONCAT_RELSIZE_HARALICK", 
                                                "SUM_MULT_RELSIZE_CONCAT_HARALICK", "SUM_CONCAT_HARALICK"]
        if(self.graph_feats_var.get() in self.graph_feats_combobox['values']):
            self.graph_feats_combobox.current(self.graph_feats_combobox['values'].index(self.graph_feats_var.get()))

        frame = tk.Frame(main_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Superpixel alg")
        label.pack(side=tk.LEFT)
        self.superpixel_alg_combobox = ttk.Combobox(frame, state="readonly", width=10)
        self.superpixel_alg_combobox.pack(side=tk.LEFT)
        self.superpixel_alg_combobox['values'] = ["DISF", "ERGC"]
        if(self.superpixel_alg_var.get() in self.superpixel_alg_combobox['values']):
            self.superpixel_alg_combobox.current(self.superpixel_alg_combobox['values'].index(self.superpixel_alg_var.get()))

        frame = tk.Frame(main_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Init seeds")
        label.pack(side=tk.LEFT)
        self.init_seeds_entry = tk.Entry(frame, width=10)
        self.init_seeds_entry.pack(side=tk.LEFT)
        self.init_seeds_entry.insert(0, str(self.init_seeds))
        
        button = tk.Button(main_frame, text="Save", command=self.sendData)
        button.pack(padx=5, pady=10)

    def sendData(self):
        dataset_name = self.dataset_entry.get()
        markers_name = self.markers_entry.get()
        arch_file = self.arch_file_var.get()
        split = self.split_combobox.get()
        img_set = self.img_set_var.get()
        stride = self.stride_combobox.get()
        attention = self.attention_combobox.get()
        aggregation = self.aggregation_combobox.get()
        nsuperpixels = int(self.nsuperpixels_entry.get())
        superpixel_alg = self.superpixel_alg_combobox.get()
        init_seeds = int(self.init_seeds_entry.get())
        graph_feats = self.graph_feats_combobox.get()

        self.fn(dataset_name, markers_name, arch_file, split, img_set,
                stride, attention, aggregation, 
                nsuperpixels, graph_feats, superpixel_alg, init_seeds)
        self.destroy()

    def load_arch_file(self):
        arch_file = tkinter.filedialog.askopenfile(title="Select arch file", filetypes=[("JSON files", "*.json")])
        if arch_file:
            self.arch_file_var.set(arch_file.name)
            print("Arch file loaded: {}".format(arch_file.name))
            with open(arch_file.name) as f:
                data = json.load(f)
                self.split_combobox['values'] = ["split"+str(l+1) for l in range(int(data['nlayers']))]
        else:
            print("No arch file selected")


class IterativeGFLIMTrain:
    def __init__(self, root, dataset_name, markers_name, arch_file, split, img_set,
                 stride, attention, aggregation, 
                 nsuperpixels, graph_feats, superpixel_alg, init_seeds):
        
        self.root = root
        self.root.title("Iterative GFLIM Training")
        self.root.geometry("1200x800")
        self.model = None
        
        self.train_img_name = tk.StringVar()
        self.train_img_name.set("1")

        self.markers_var = tk.IntVar()
        self.markers_var.set(0)
        self.progress_var = tk.IntVar()
        self.progress_var.set(0)
        
        self.layer = 1
        self.tabs = {}
        self.selected_kernels = []
        self.create_widgets()

        self.getModelConfig(dataset_name, markers_name, arch_file, split, img_set,
                            stride, attention, aggregation,
                            nsuperpixels, graph_feats, superpixel_alg, init_seeds)
        

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        main_area = tk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        main_area.pack(fill=tk.BOTH, expand=True)

        model_config_frame = ttk.LabelFrame(main_area, text="Model Configuration")
        model_config_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_area.add(model_config_frame)

        button = tk.Button(model_config_frame, text="Set model configuration", command=self.setModelConfig)
        button.pack(padx=5, pady=5)

        button = tk.Button(model_config_frame, text="Train Layer no merge", command=self.train_layer_no_merge)
        button.pack(padx=5, pady=5)

        button = tk.Button(model_config_frame, text="Select kernels", command=self.select_kernels)
        button.pack(padx=5, pady=5)

        frame = tk.Frame(model_config_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Image name")
        label.pack(side=tk.LEFT)
        self.train_images_combobox = ttk.Combobox(frame, state="readonly", width=10, textvariable=self.train_img_name)
        self.train_images_combobox.pack(side=tk.LEFT)
        self.train_images_combobox['values'] = [""]
        

        frame = tk.LabelFrame(model_config_frame, text="Markers")
        frame.pack(padx=5, pady=5)

        button = tk.Radiobutton(frame, text="Original", variable=self.markers_var, value=0)
        button.pack(side=tk.LEFT)
        self.model_markers_radiobutton = tk.Radiobutton(frame, text="Reduced", variable=self.markers_var, value=1)
        self.model_markers_radiobutton.pack(side=tk.LEFT)

        frame = tk.Frame(model_config_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Image size")
        label.pack(side=tk.LEFT)
        self.image_size_spinbox = tk.Spinbox(frame, from_=200, to=1000, width=5, relief="sunken", 
                                             repeatdelay=500, repeatinterval=100, increment=50,
                                             command=self.show_activations)
        self.image_size_spinbox.pack(side=tk.LEFT)

        frame = tk.Frame(model_config_frame)
        frame.pack(padx=5, pady=5)
        label = tk.Label(frame, text="Images per column")
        label.pack(side=tk.LEFT)
        self.ncols_spinbox = tk.Spinbox(frame, from_=1, to=100, width=5, relief="sunken", repeatdelay=500, repeatinterval=100, command=self.show_activations)
        self.ncols_spinbox.pack(side=tk.LEFT)


        self.display_frame = ttk.Notebook(main_area)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_area.add(self.display_frame)

        self.progress_bar = ttk.Progressbar(self.main_frame, 
                                            variable=self.progress_var, 
                                            mode='determinate', value=0)

    def setModelConfig(self):
        if self.model == None:
            self.model = GFLIMModel("", "", "", "", "",
                                    "", "", "", 
                                    0, "", "", 0)
            
        model_config = ConfigModel(self.root, self.model.dataset_name, self.model.markers_name, 
                                   self.model.arch_file, self.model.split, self.model.img_set, 
                                   self.model.stride, self.model.attention, self.model.aggregation, 
                                   self.model.nsuperpixels, self.model.graph_feats, self.model.superpixel_alg, 
                                   self.model.init_seeds, self.getModelConfig)
        model_config.create_widgets()

    def getModelConfig(self, dataset_name, markers_name, arch_file, split, img_set,
                        stride, attention, aggregation, 
                        nsuperpixels, graph_feats, superpixel_alg, init_seeds):
        run_again = False
        if (self.model == None or
            self.model.dataset_name != dataset_name or 
            self.model.markers_name != markers_name or
            self.model.arch_file != arch_file or
            self.model.split != split or
            self.model.img_set != img_set or
            self.model.stride != stride or
            self.model.attention != attention or
            self.model.aggregation != aggregation or
            int(self.model.nsuperpixels) != int(nsuperpixels) or
            self.model.graph_feats != graph_feats or
            self.model.superpixel_alg != superpixel_alg or
            int(self.model.init_seeds) != int(init_seeds)):
            run_again = True

        print("Model configuration updated")

        if run_again or (self.model != None and self.model.readytoRun == False):
            self.cleanWindow()
            self.model = GFLIMModel(dataset_name, markers_name, arch_file, split, img_set,
                                stride, attention, aggregation,
                                nsuperpixels, graph_feats, superpixel_alg, init_seeds)
            self.train_images_combobox['values'] = [name.split('-')[0] for name in os.listdir(self.model.orig_img_markers_dir) if name.endswith("-seeds.txt")]
            self.train_images_combobox.current(0)
            if os.path.exists(self.model.model_img_markers_dir):
                self.model_markers_radiobutton.config(state=tk.NORMAL)
            else:
                self.model_markers_radiobutton.config(state=tk.DISABLED)
                self.markers_var.set(0)
            self.show_activations()
            
    def cleanWindow(self):
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        self.tabs = {}
        self.selected_kernels = []
  
    def create_tab(self, frame, label):
        tab = tk.Frame(frame)
        frame.add(tab, text=label)
        
        canvas = tk.Canvas(tab)
        vscroll = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        hscroll = ttk.Scrollbar(tab,orient="horizontal",command=canvas.xview)
        hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        canvas.configure(yscrollcommand=vscroll.set,
                         xscrollcommand=hscroll.set)

        content = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        return (tab, canvas, content)

    def show_activations(self):
        if self.model == None:
            tkinter.messagebox.showerror("Error", "Model not configured / invalid. Please set the model configuration.")
            return
        
        activations_dir = "{}/activationNoMerge{}/".format(self.model.model_dir, self.layer)
        if os.path.exists(activations_dir):
            self.cleanWindow()
            for label in os.listdir(activations_dir):
                self.tabs[label] = (self.create_tab(self.display_frame, label))
                label_dir = os.path.join(activations_dir, label, self.train_img_name.get())
                self.loadImages(label_dir, self.tabs[label])
        else:
            label = tk.Label(self.display_frame, text="No activations found. Train a model or set a new configuration.", fg="gray", font=("Arial", 12))
            label.pack(fill=tk.BOTH, expand=True)
            return

    def display_image(self, img_path, frame, r, c):
        basename = os.path.basename(img_path).split(".")[0].split("-")[1]
        img = Image.open(img_path)
        height = int(self.image_size_spinbox.get())
        img.thumbnail((height, int(height*(img.width/img.height))), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        
        button = tk.Label(frame, image=img, text=basename, compound=tk.TOP)
        button.image = img
        button.grid(row=r, column=c)
        button.bind("<Button-1>", lambda event, kernel_name=basename, frame=button: self.on_image_click(kernel_name, button))
        #button.config(relief=tk.RAISED)
        

    def on_image_click(self, kernel_name, button):
        print("Image clicked:", kernel_name)
        kernel_val = int(kernel_name)
        if(kernel_val not in self.selected_kernels):
            self.selected_kernels.append(kernel_val)
            button.config(bg="gray")
        else:
            self.selected_kernels.remove(kernel_val)
            button.config(bg='#d9d9d9')
        print("Selected kernels:", self.selected_kernels)
        
    def loadImages(self, img_dir, frame_data):
        frame, canvas, content = frame_data
        img_files = os.listdir(img_dir)
        ncols = int(self.ncols_spinbox.get())
        r, c = 0, 0
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            if os.path.isfile(img_path):
                self.display_image(img_path, content, r, c)
                c += 1
                if c >= ncols:
                    c = 0
                    r += 1
        content.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    def select_kernels(self):
        """ select kernels and train again"""
        if len(self.selected_kernels) == 0:
            print("No kernels selected")
            return
        
        print('chamando funcção selectKernels')
        if self.model == None:
            tkinter.messagebox.showerror("Error", "Model not configured / invalid. Please set the model configuration.")
            return
        
        img_markers = self.model.orig_img_markers_dir if self.markers_var.get() == 0 else self.model.model_img_markers_dir
        self.model.selectKernels(self.layer, img_markers, self.selected_kernels, str(self.train_img_name.get()))
        self.markers_var.set(1)
        self.model.trainLayer_noMerge(self.layer, self.model.img_set, img_markers)
        self.selected_kernels = []
        rmdir(self.model.model_dir + "/activationNoMerge{}/".format(self.layer))
        self.model.writeActivations(self.layer)
        self.show_activations()
        

    def train_layer(self):
        if self.model == None:
            tkinter.messagebox.showerror("Error", "Model not configured / invalid. Please set the model configuration.")
            return
        
        img_markers = self.model.orig_img_markers_dir if self.markers_var.get() == 0 else self.model.model_img_markers_dir
        self.model.trainLayer(self.layer, self.model.img_set, img_markers)

    def train_layer_no_merge(self):
        if self.model == None:
            tkinter.messagebox.showerror("Error", "Model not configured / invalid. Please set the model configuration.")
            return
        
        img_markers = self.model.orig_img_markers_dir if self.markers_var.get() == 0 else self.model.model_img_markers_dir
        self.model.trainLayer_noMerge(self.layer, self.model.img_set, img_markers)
        self.model.writeActivations(self.layer)
        self.show_activations()


if __name__ == "__main__":
    # model configuration
    layer = "1"
    stride = "graph-based"
    attention = "NONE"
    aggregation = "CONCAT"
    nsuperpixels = "20000"
    graph_feats = "SUM"
    split = "split1"
    img_set = "train"

    superpixel_alg = "DISF"
    init_seeds = "50000"

    # directories
    dataset_name = "cells_dataset"
    markers_name = "gt_markers_erosion5_interative2"
    arch_file = "/home/isabela/Codes/LibIFT/Datasets/cells_dataset/arch_cells.json"

    root = tk.Tk()
    app = IterativeGFLIMTrain(root, dataset_name, markers_name, arch_file, split, img_set,
                              stride, attention, aggregation,
                              nsuperpixels, graph_feats, superpixel_alg, init_seeds)
    root.mainloop()
    