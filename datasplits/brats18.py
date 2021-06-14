import os, json
from shutil import copyfile
import nibabel as nib

# Source data:
path_source = "/media/miguelv/HD1/Datasets/BraTS/2018/MICCAI_BraTS_2018_Data_Training/HGG/"
# Target folder (it should not exist!)
path_output = "/media/miguelv/HD1/Datasets/nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Task102_Brats18/"


if os.path.isdir(path_output):
    raise Exception("The folder already exists!")
os.makedirs(os.path.join(path_output, "imagesTr"))
os.makedirs(os.path.join(path_output, "imagesTs"))
os.makedirs(os.path.join(path_output, "labelsTr"))
os.makedirs(os.path.join(path_output, "labelsTs"))

lista = sorted(os.listdir(path_source))

train_list = lista[::2] # [:2]
val_list = lista[1::2] # [:2]


modalities = ["t2", "t1", "t1ce", "flair"]
# For training
for subject in train_list:
    for i, mod in enumerate(modalities):
        orig_f = path_source + subject + "/" + subject + "_" + mod + ".nii.gz"
        dest_f = path_output + "imagesTr/" + subject + "_" + str(i).zfill(4) + ".nii.gz"
        copyfile(orig_f, dest_f)

    orig_f = path_source + subject + "/" + subject + "_seg.nii.gz"
    dest_f = path_output + "labelsTr/" + subject + ".nii.gz"

    img = nib.load(orig_f)
    data = img.get_data()
    data[data==4] = 3 # This step is actually important, other nnUNet won't work
    nib.save(nib.Nifti1Image(data, affine=img.affine, header=img.header), dest_f)
    #copyfile(orig_f, dest_f)

# For testing
for subject in val_list:
    for i, mod in enumerate(modalities):
        orig_f = path_source + subject + "/" + subject + "_" + mod + ".nii.gz"
        dest_f = path_output + "imagesTs/" + subject + "_" + str(i).zfill(4) + ".nii.gz"
        copyfile(orig_f, dest_f)

    orig_f = path_source + subject + "/" + subject + "_seg.nii.gz"
    dest_f = path_output + "labelsTs/" + subject + ".nii.gz"
    #copyfile(orig_f, dest_f)

    img = nib.load(orig_f)
    data = img.get_data()
    data[data==4] = 3 # This step is actually important, other nnUNet won't work
    nib.save(nib.Nifti1Image(data, affine=img.affine, header=img.header), dest_f)



# Creating the JSON file
files = []
for subject in train_list:
    x = path_output + "imagesTr/" + subject + ".nii.gz"
    y = path_output + "labelsTr/" + subject + ".nii.gz"
    files.append({"image": x, "label": y})

    #files.append({"image": (path_output + "imagesTr/" + f).replace("_0000", ""),
    #    "label": os.path.abspath(f).replace("imagesTr", "labelsTr").replace("_0000", "")})

d = {"name": "Brats18",
    "description": "Brats18 Dataset",
    "reference": "",
    "license": "",
    "release": "",
    "tensorImageSize": "3D",
    "modality": {"0": "T2", "1": "T1", "2": "T1CE", "3": "FLAIR"},
    "labels": {
        "0": "background",
        "1": "ncr_net",
        "2": "edema",
        "3": "et"
        },
    "numTraining": len(train_list),
    "numTest": 0,
    "training": files,
    "test": []
    }


with open(path_output + "dataset.json", "w") as f:
    f.write(json.dumps(d))
