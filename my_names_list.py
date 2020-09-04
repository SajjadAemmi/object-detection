import os

images_path = 'images'
dataset_path = 'farazist_dataset'

dir_list = os.listdir(images_path)
dir_list.sort(key=str)

file_labelmap = open(os.path.join(dataset_path, 'labelmap.pbtxt'), 'w')
file_names = open(os.path.join(dataset_path, 'names.txt'), 'w')

for id, d in enumerate(dir_list):
    if os.path.isdir(os.path.join(images_path, d)):
        name = d.split(' ')[-1]
        text = "item {\n  id:" + str(id + 1) + "\n  name: '" + name + "'\n}"
        file_labelmap.write(text + '\n')
        file_names.write(name + '\n')

file_labelmap.close()
file_names.close()