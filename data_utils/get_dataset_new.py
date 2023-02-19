import os.path as osp
import os
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import json
from randaugment import RandAugment
from PIL import Image

from utils.cutout import SLCutoutPIL

# YOUR_PATH
inte_image_path = '/data/sqhy_data/intent_resize'
inte_train_anno_path = './data/intentonomy/intentonomy_train2020.json'
inte_val_anno_path = './data/intentonomy/intentonomy_val2020.json'
inte_test_anno_path = './data/intentonomy/intentonomy_test2020.json'

CLASS_15 = {
    '0': [24, 27],
    '1': [5],
    '2': [13],
    '3': [11],
    '4': [3, 7, 8, 9, 19],
    '5': [4, 6, 20, 21],
    '6': [14],
    '7': [15, 16],
    '8': [2, 22, 23],
    '9': [0],
    '10': [17, 25],
    '11': [10],
    '12': [1],
    '13': [12],
    '14': [18, 26],
}

CLASS_9 = {
    '0': [0, 1],
    '1': [2, 3],
    '2': [4, 5],
    '3': [6],
    '4': [7, 8, 9],
    "5": [10],
    '6': [11],
    '7': [12],
    '8': [13, 14]
}

class InteDataSet(data.Dataset):
    def __init__(self, 
                 image_dir, 
                 anno_path, 
                 input_transform=None, 
                 labels_path=None,
    ):
        self.image_dir = image_dir
        self.anno_path = anno_path
        
        self.input_transform = input_transform
        self.labels_path = labels_path
        
        self.labels = []
        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path).astype(np.float64)
            self.labels = (self.labels > 0).astype(np.float64)
        else:
            print('labels_path not exist, please check the path or run get_label_vector.py first')
    
    def _load_image(self, index):
        image_path = self._get_image_path(index)
        
        return Image.open(image_path).convert("RGB")
    
    def _get_image_path(self, index):
        with open(self.anno_path, 'r') as f:
            annos_dict = json.load(f)
            annos_i = annos_dict['annotations'][index]
            id = annos_i['id']
            if id != index:
                raise ValueError('id not equal to index')
            img_id_i = annos_i['image_id']
            
            imgs = annos_dict['images']
            
            for img in imgs:
                if img['id'] == img_id_i:
                    image_file_name = img['filename']
                    image_file_path = os.path.join(self.image_dir, image_file_name)
                    break
        
        return image_file_path
                    
    def __getitem__(self, index):
        input = self._load_image(index)
        if self.input_transform:
            input = self.input_transform(input)
        label = self.labels[index]
        return input, label
    
    def __len__(self):
        return self.labels.shape[0]


def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                            RandAugment(),
                                               transforms.ToTensor(),
                                               normalize]
    if args.cutout:
        print("Using Cutout!!!")
        train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
        
    train_data_transform = transforms.Compose(train_data_transform_list)
    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    
    if args.dataname == 'intentonomy':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = InteDataSet(
            image_dir=inte_image_path,
            anno_path=inte_train_anno_path,
            input_transform=train_data_transform,
            labels_path='./data/intentonomy/train_label_vectors_intentonomy2020.npy',
        )
        val_dataset = InteDataSet(
            image_dir=inte_image_path,
            anno_path=inte_val_anno_path,
            input_transform=test_data_transform,
            labels_path='./data/intentonomy/val_label_vectors_intentonomy2020.npy',
        )
        test_dataset = InteDataSet(
            image_dir=inte_image_path,
            anno_path=inte_test_anno_path,
            input_transform=test_data_transform,
            labels_path='./data/intentonomy/test_label_vectors_intentonomy2020.npy',
        )

    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    print("len(test_dataset):", len(test_dataset))
    return train_dataset, val_dataset, test_dataset







