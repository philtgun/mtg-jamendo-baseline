import csv
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils import data


class MtgJamendoDataset(data.Dataset):
    SAMPLING_STRATEGY = ['random', 'center']

    def __init__(self, data_dir: str, tsv_file: str, input_length: int, tags_file: Optional[str] = None,
                 sampling_strategy: str = 'center'):
        self.data_dir = Path(data_dir)
        self.input_length = input_length

        if sampling_strategy not in self.SAMPLING_STRATEGY:
            raise ValueError(f'Invalid value of segment_location, should be one of {self.SAMPLING_STRATEGY}')
        self.segment_location = sampling_strategy

        self.paths, tags = self.parse_csv(tsv_file)

        mlb = MultiLabelBinarizer()
        if tags_file is None:
            self.y = mlb.fit_transform(tags)
            self.labels = mlb.classes_
        else:
            self.labels = np.loadtxt(tags_file, dtype=str)
            mlb.fit([self.labels])
            self.y = mlb.transform(tags)

    @staticmethod
    def parse_csv(csv_file: str):
        paths = []
        tags = []

        with open(csv_file) as fp:
            reader = csv.reader(fp, delimiter='\t')
            next(reader, None)  # skip header
            for row in reader:
                paths.append(row[3])
                tags.append(row[5:])

        return paths, tags

    def get_segment_start(self, length):
        if self.segment_location == 'random':
            return int(np.random.random_sample() * (length - self.input_length))

        if self.segment_location == 'center':
            return (length - self.input_length) // 2

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.data_dir / self.paths[index]

        audio = np.load(str(path.with_suffix('.npy')), mmap_mode='r')  # TODO: compare with raw memmap

        start = self.get_segment_start(audio.shape[-1])
        x = audio[:, start:start+self.input_length]

        return x.astype(np.float32), self.y[index].astype(np.float32)

    @staticmethod
    def get_tsv_file(repo_path: str, subset: str, purpose: str, split: int):
        return Path(repo_path) / 'data' / 'splits' / f'split-{split}' / f'{subset}-{purpose}.tsv'

    @staticmethod
    def get_tags_file(repo_path: str, subset: str):
        return Path(repo_path) / 'data' / 'tags' / f'{subset}.txt'
