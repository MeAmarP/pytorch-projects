import glob
import os
from pathlib import Path


DATASET_DIR = Path.home().joinpath(Path("Documents/project/learning/datasets"))
NAME_OF_DATASET = "concreate_cracks"
PATH_TO_DATASET = Path.joinpath(DATASET_DIR).joinpath(NAME_OF_DATASET)

if __name__ == "__main__":
    print(PATH_TO_DATASET.exists())