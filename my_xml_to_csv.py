import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

dataset_path = 'farazist_dataset'

def xml_to_csv(path):

    dir_list = os.listdir(path)
    dir_list.sort(key=str)

    xml_list = []
    for d in dir_list:
        if os.path.isdir(os.path.join(path, d)):
            for xml_file in glob.glob(os.path.join(path, d) + '/*.xml'):
                tree = ET.parse(xml_file)
                root = tree.getroot()

                for member in root.findall('object'):
                    value = (os.path.join(d, root.find('filename').text),
                            int(root.find('size')[0].text),
                            int(root.find('size')[1].text),
                            d.split(' ')[-1],
                            int(member[4][0].text),
                            int(member[4][1].text),
                            int(member[4][2].text),
                            int(member[4][3].text)
                            )
                    xml_list.append(value)
    
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == '__main__':
    for folder in ['train','test']:
        image_path = os.path.join(dataset_path, folder)
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv((os.path.join(dataset_path, folder + '_labels.csv')), index=None)
        print('Successfully converted xml to csv.')
