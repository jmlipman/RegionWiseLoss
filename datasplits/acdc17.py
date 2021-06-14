import os, json
from shutil import copyfile
import nibabel as nib

# Source data:
path_source = "/media/miguelv/HD1/Datasets/ACDC17/explore_data/baum/training/"
# Target folder (it should not exist!)
path_output = "/media/miguelv/HD1/Datasets/nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Task103_ACDC17aaa/"

if os.path.isdir(path_output):
    raise Exception("The folder already exists!")
os.makedirs(os.path.join(path_output, "imagesTr"))
os.makedirs(os.path.join(path_output, "imagesTs"))
os.makedirs(os.path.join(path_output, "labelsTr"))
os.makedirs(os.path.join(path_output, "labelsTs"))

train_list = [str(x) for x in range(2, 101, 2)]
val_list = [str(x) for x in range(1, 101, 2)]

train_files = []

# For training
for ll, split in zip([train_list, val_list], ["Tr", "Ts"]):
    for subject in ll:
        path = path_source + "patient" + subject.zfill(3) + "/"
        for f in os.listdir(path):
            if f.endswith("_gt.nii.gz"):
                # Images
                f_img = f.replace("_gt", "")
                orig_f1 = path + f_img
                dest_f1 = path_output + "images" + split + "/" + f_img.replace(".nii.gz", "_0000.nii.gz")
                copyfile(orig_f1, dest_f1)

                # Labels
                orig_f2 = path + f
                dest_f2 = path_output + "labels" + split + "/" + f_img
                copyfile(orig_f2, dest_f2)

                if split == "Tr":
                    train_files.append({"image": dest_f1.replace("_0000", ""), "label": dest_f2})


# Creating the JSON file
d = {"name": "ACDC17",
    "description": "ACDC17 Dataset",
    "reference": "",
    "license": "",
    "release": "",
    "tensorImageSize": "3D",
    "modality": {"0": "T2"},
    "labels": {
        "0": "background",
        "1": "RV_cavity",
        "2": "myocardium",
        "3": "LV_cavity"
        },
    "numTraining": len(train_files),
    "numTest": 0,
    "training": train_files,
    "test": []
    }


with open(path_output + "dataset.json", "w") as f:
    f.write(json.dumps(d))
