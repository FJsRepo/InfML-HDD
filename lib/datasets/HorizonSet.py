import os
import json
import random

SPLIT_FILES = {
    'train': ['train_data_1_train.json', 'train_data_2_train.json', 'train_data_3_train.json',
              'train_data_4_train.json', 'train_data_5_train.json', 'train_data_6_train.json'],

    # 'val': ['val_data_val.json'],

    'test5': ['test_data_5_test.json'],

    'test': ['test_data_1_test.json', 'test_data_2_test.json', 'test_data_3_test.json',
             'test_data_4_test.json', 'test_data_5_test.json', 'test_data_6_test.json']
}

class HorizonNet(object):
    def __init__(self, split='train',  Horizon_num=None, root=None, metric='default'):
        self.split = split
        self.root = root
        self.metric = metric
        if split not in SPLIT_FILES.keys():
            raise Exception('Split `{}` does not exist.'.format(split))
        self.anno_files = [os.path.join(self.root, path) for path in SPLIT_FILES[split]]
        if root is None:
            raise Exception('Please specify the root directory')

        self.img_w, self.img_h = 384, 288
        self.max_points = 0
        self.load_annotations()

        if Horizon_num is not None:
            self.Horizon_num = Horizon_num

    def get_img_heigth(self, path):
        return 288

    def get_img_width(self, path):
        return 384

    def load_annotations(self):
        self.annotations = []
        Horizon_num = 1
        for anno_file in self.anno_files:
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['y_coordinate']
                gt_Horizon = data['x_coordinate']
                Horizon_temp = [(x, y) for (x, y) in zip(gt_Horizon, y_samples) if x >= 0]
                Horizon = []
                Horizon.append(Horizon_temp)
                gt_Horizon_temp = []
                gt_Horizon_temp.append(gt_Horizon)
                self.max_points = len(gt_Horizon)
                self.annotations.append({
                    'path': os.path.join(self.root, data['raw_file']),
                    'org_path': data['raw_file'],
                    'org_Horizon': gt_Horizon_temp,
                    'Horizon': Horizon,#[[(x,y),(x,y),(x,y)...],[],[]...]
                    'aug': False,
                    'y_samples': y_samples
                })

        if self.split == 'train':
            random.shuffle(self.annotations)
        print('total annos', len(self.annotations))
        self.Horizon_num = Horizon_num

    def transform_annotations(self, transform):
        self.annotations = list(map(transform, self.annotations))

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
