import os
import sys

if (len(sys.argv) != 4):
    print("python3 rename_folders.py <P1> <P2> <P3>")
    print("P1: Parent directory")
    print("P2: Folder name")
    print("P3: New folder name")
    exit()

parent_dir = sys.argv[1]
folder_name = sys.argv[2]
new_folder_name = sys.argv[3]

if(os.path.exists(parent_dir) == False):
    print("Folder {} does not exist".format(parent_dir))
    exit()

if(os.path.exists(parent_dir+"/"+folder_name) == False):
    print("Folder {} does not exist".format(parent_dir+"/"+folder_name))
    exit()

if(os.path.exists(parent_dir+"/"+new_folder_name) == True):
    print("Folder {} already exists".format(parent_dir+"/"+new_folder_name))
    exit()

os.rename(parent_dir+"/"+folder_name, parent_dir+"/"+new_folder_name)


