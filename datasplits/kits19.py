import os, json
from shutil import copyfile
import nibabel as nib
import numpy as np
from PIL import Image

def imresize(im, shape):
    return np.array(Image.fromarray(im).resize(shape))

# Source data:
path_source = "/media/miguelv/HD1/Datasets/KiTS19/kits19/data/"
# Target folder (it should not exist!)
path_output = "/media/miguelv/HD1/Datasets/nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Task103_Kits19/"

if os.path.isdir(path_output):
    raise Exception("The folder already exists!")
os.makedirs(os.path.join(path_output, "imagesTr"))
os.makedirs(os.path.join(path_output, "imagesTs"))
os.makedirs(os.path.join(path_output, "labelsTr"))
os.makedirs(os.path.join(path_output, "labelsTs"))

# Training: patient0, 2, 4, ...
train_list = [str(x) for x in range(0, 210, 2)]
val_list = [str(x) for x in range(1, 210, 2)]

train_files = []

# For training
for ll, split in zip([train_list, val_list], ["Tr", "Ts"]):
    for subject in ll:
        print(subject)
        path = path_source + "case_" + subject.zfill(5) + "/"

        # Images
        orig_f1 = path + "imaging.nii.gz"
        dest_f1 = path_output + "images" + split + "/case_" + subject.zfill(5) + "_0000.nii.gz"
        Img = nib.load(orig_f1)
        Img = Img.as_reoriented([[2, 1], [0, 1], [1, 1]])
        #copyfile(orig_f1, dest_f1)

        # Labels
        orig_f2 = path + "segmentation.nii.gz"
        dest_f2 = path_output + "labels" + split + "/case_" + subject.zfill(5) + ".nii.gz"
        #copyfile(orig_f2, dest_f2)

        Labels = nib.load(orig_f2)
        slices = np.sum(Labels.get_data(), axis=(1,2)) != 0
        Labels = Labels.as_reoriented([[2, 1], [0, 1], [1, 1]])
        labels_data = Labels.get_data()[:, :, slices]
        image_data = Img.get_data()[:, :, slices]

        new_image = np.zeros((256, 256, image_data.shape[-1]))
        for j in range(image_data.shape[-1]):
            new_image[:,:,j] = imresize(image_data[:,:,j], (256, 256))

        new_labels = np.zeros((256, 256, labels_data.shape[-1]))
        for j in range(labels_data.shape[-1]):
            new_labels[:,:,j] = imresize(labels_data[:,:,j], (256, 256))

        #nib.save(Img, dest_f1)
        nib.save(nib.Nifti1Image(new_image,
            affine=Img.affine, header=Img.header), dest_f1)
        nib.save(nib.Nifti1Image(new_labels,
            affine=Img.affine, header=Img.header), dest_f2)

        if split == "Tr":
            train_files.append({"image": dest_f1.replace("_0000.nii", ".nii"), "label": dest_f2})


# Creating the JSON file
d = {"name": "Kits19",
    "description": "Kits19 Dataset",
    "reference": "",
    "license": "",
    "release": "",
    "tensorImageSize": "3D",
    "modality": {"0": "CT"},
    "labels": {
        "0": "background",
        "1": "kidney",
        "2": "tumor"
        },
    "numTraining": len(train_files),
    "numTest": 0,
    "training": train_files,
    "test": []
    }


with open(path_output + "dataset.json", "w") as f:
    f.write(json.dumps(d))
