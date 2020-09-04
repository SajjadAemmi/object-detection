import os
import glob
import random
import shutil

images_path = 'images'
dataset_path = 'farazist_dataset'

for d in ['train','test']:
    if not os.path.exists(os.path.join(dataset_path, d)):
        os.makedirs(os.path.join(dataset_path, d))

dir_list = os.listdir(images_path)
dir_list.sort(key=str)

for d in dir_list:
    if os.path.isdir(os.path.join(images_path, d)):

        for f in ['train','test']:
            if not os.path.exists(os.path.join(dataset_path, f, d)):
                os.makedirs(os.path.join(dataset_path, f, d))

        for f in glob.glob(os.path.join(images_path, d, "*.jpg")):
            file_name, file_extension = os.path.splitext(f)
            file_name = file_name.split('/')[-1]

            if random.random() < 0.8:
                shutil.copy(os.path.join(images_path, d, file_name + ".jpg"), os.path.join(dataset_path, 'train', d, file_name + ".jpg"))
                shutil.copy(os.path.join(images_path, d, file_name + ".xml"), os.path.join(dataset_path, 'train', d, file_name + ".xml"))
            else:
                shutil.copy(os.path.join(images_path, d, file_name + ".jpg"), os.path.join(dataset_path, 'test', d, file_name + ".jpg"))
                shutil.copy(os.path.join(images_path, d, file_name + ".xml"), os.path.join(dataset_path, 'test', d, file_name + ".xml"))