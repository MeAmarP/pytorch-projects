import fiftyone as fo

from config import PATH_TO_DATASET

path_to_dataset = "/home/c3po/Documents/project/learning/datasets/concreate_cracks/"
dataset = fo.Dataset.from_dir(path_to_dataset,
                              dataset_type=fo.types.ImageClassificationDirectoryTree,
                              name='con-cracks')

if __name__ == '__main__':
    print(f"Loading Dataset from {PATH_TO_DATASET}")
    print(dataset)
