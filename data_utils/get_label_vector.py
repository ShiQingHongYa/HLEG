import numpy as np
import json
import os
from tqdm import tqdm

# intentonomy
inte_image_path = '/data/zzy_data/intent_resize/test/low'
inte_train_anno_path = '/data/zzy_data/intent_resize/annotations/intentonomy_train2020.json'
inte_val_anno_path = '/data/zzy_data/intent_resize/annotations/intentonomy_test2020.json'

# val/test data label transfer
def label2vectors(int_label):
    label_vectors = np.zeros((28,))
    for label in int_label:
        label_vectors[label] = 1
    return label_vectors

# train data label transfer
def prob2vectors(prob_vector, TH=0):
    label_vectors = np.zeros_like(prob_vector)
    for i, prob in enumerate(prob_vector):
        if prob > TH:
            label_vectors[i] = 1
    return label_vectors

def get_label_vectors(anno_path, image_path):
    stage = anno_path.split('/')[-1].split('.')[0].split('_')[-1].split('2',1)[0]
    dataset_name = anno_path.split('/')[-1].split('.')[0].split('_')[0]
    print(stage)

    label_vectors = []
    with open(anno_path, 'r') as f:
        label = json.load(f)
        annos = label['annotations'] # [id, image_id, category_id]
        img = label['images'] # [image_id, file_name(path), original id, link]
        matched = False
        for anno in tqdm(annos):
            idx = anno['id']
            image_i_id = anno['image_id']
            
            for img_i in img:
                if img_i['id'] == image_i_id:
                    matched = True
                    # image_file_path: 
                    # no use for label transfer, just make it available for future develpment
                    image_file_name = img_i['filename']
                    image_file_path = os.path.join(image_path, image_file_name)
                    
                    if stage == 'train':
                        category_ids = anno['category_ids_softprob']
                        label_vector_i = prob2vectors(category_ids)
                    else:
                        category_ids = anno['category_ids']
                        label_vector_i = label2vectors(category_ids)
                        
                    label_vectors.append(label_vector_i)
                    break
            
            if not matched:
                print('image_id not match, please check the dataset[path:{}] anno[index:{}]'.format(anno_path, idx))
                continue
            
    label_vectors = np.array(label_vectors)
    np.save(
        '/home/zhangziyi/query2labels/data/intentonomy/{}_label_vectors_{}{}.npy'.format(stage, dataset_name, '2020'),
        label_vectors
    )
    
# anno_path = inte_train_anno_path
anno_path = inte_val_anno_path
image_path = inte_image_path
get_label_vectors(anno_path, image_path)
