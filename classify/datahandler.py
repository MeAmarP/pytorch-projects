import fiftyone as fo
import numpy as np

import torch
from torchvision import datasets, transforms, models

from config import PATH_TO_DATASET


class ConCracksData:
    def __init__(self):
        self.__path_to_dataset = "/home/c3po/Documents/project/learning/datasets/concreate_cracks/"
        self.test_set_size = 0.2
        self.input_shape = 224

    def init_dataset(self):
        # Here we will initialize dataset
        dataset = fo.Dataset.from_dir(self.path_to_dataset,
                                      dataset_type=fo.types.ImageClassificationDirectoryTree,
                                      name='con-cracks')

    def load_split_dataset(self):
        train_transforms = transforms.Compose([transforms.Resize(self.input_shape),
                                               transforms.ToTensor(),
                                               ])
        test_transforms = transforms.Compose([transforms.Resize(self.input_shape),
                                              transforms.ToTensor(),
                                              ])
        train_data = datasets.ImageFolder(self.__path_to_dataset,
                                          transform=train_transforms)
        test_data = datasets.ImageFolder(self.__path_to_dataset,
                                         transform=test_transforms)
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(self.test_set_size * num_train))
        np.random.shuffle(indices)
        from torch.utils.data.sampler import SubsetRandomSampler
        train_idx, test_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        trainloader = torch.utils.data.DataLoader(train_data,
                                                  sampler=train_sampler, batch_size=64)
        testloader = torch.utils.data.DataLoader(test_data,
                                                 sampler=test_sampler, batch_size=64)
        return trainloader, testloader



if __name__ == '__main__':
    print(fo.list_datasets())
