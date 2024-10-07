# --------------------------------------------------------
# Adapted from the Microsoft project: https://github.com/microsoft/unilm/tree/master/beit3 
# --------------------------------------------------------


import os
import json
import random
import re
import torch
import glob
import csv
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import CenterCrop
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform
from torchsampler import ImbalancedDatasetSampler

import utils
import openslide
import pandas as pd
import numpy as np

import PIL
PIL.Image.MAX_IMAGE_PIXELS = None


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path, split, transform, 
        task=None, k_fold=0,
    ):
        index_files = self.get_index_files(split, k_fold=k_fold, task=task)
        self.data_path = data_path
        items = []
        self.index_files = index_files
        labels = []

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(data_path, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                    labels.append(data["label_morphology"])
                print("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items
        self.loader = default_loader
        self.transform = transform
        self.split = split
        self.labels = labels

    def callback_get_label(dataset, index):
        
        return 
    @staticmethod
    def get_index_files(split):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    def __getitem__(self, index: int):
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = '{' + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  transforms = ["
        for t in self.transform.transforms:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body


def _write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    print("Write %s with %d items !" % (jsonl_file, len(items)))


def df_prep(data, label_dict, ignore, label_col):
    if label_col != 'label':
        data['label'] = data[label_col].copy()

    mask = data['label'].isin(ignore)
    data = data[~mask]
    data.reset_index(drop=True, inplace=True)
    for i in data.index:
        key = data.loc[i, 'label']
        data.at[i, 'label'] = label_dict[key]

    return data


def get_split_from_df(slide_data, all_splits, prop=1.0, seed=1, split_key='train'):
    split = all_splits[split_key].str.rstrip('.svs')
    split = split.dropna().reset_index(drop=True)

    if len(split) > 0:
        mask = slide_data['slide_id'].isin(split.tolist())
        df_slice = slide_data[mask].reset_index(drop=True)
        if split_key == 'train' and prop != 1.0:
            df_slice = df_slice.sample(frac=prop, random_state=seed).reset_index(drop=True)
        if split_key == 'train':
            print(df_slice.head())
        print("Traing Data Size ({%0.2f}): %d" % (prop, df_slice.shape[0]))
    else:
        df_slice = None
    
    return df_slice

class GSClassificationDataset(BaseDataset):
    def __init__(self, data_path, split, transform, task, k_fold, image_dir, seq_parallel=False, cached_randaug=False):
        super().__init__(
            data_path=data_path, split=split, 
            transform=transform, task=task, k_fold=k_fold, 
        )
        self.k_fold = k_fold
        self.image_dir = image_dir
        self.seq_parallel = seq_parallel
        self.cached_randaug = cached_randaug
        self.label_key = 'label_morphology'

    @staticmethod
    def get_index_files(split, k_fold=0, task=None):
        if split in ["train", "val", "test"]:
            return ("{}.index.{}.jsonl".format(split, k_fold), )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        if self.split == "train":
            num_samples = 4
        else:
            num_samples = 4
        if self.split == "train" or self.split == "val":
            sampled_regions = random.choices(item["slide_regions"], k=num_samples)
        if self.split == "test":
            sampled_regions = [item["slide_regions"]]
        imgs = []
        labels = []
        slideindex = []
        for r in sampled_regions:
            try:
                image_path = os.path.join((r.split("-")[0]),r)
                img = self._get_image(image_path)
                imgs.append(img)
                labels.append(torch.tensor(item[self.label_key]))
            except:
                # just have duplicate if file load fails to keep dimensions consistent
                imgs.append(img)
                labels.append(torch.tensor(item[self.label_key]))
        data["image"] = imgs
        data["label"] = torch.tensor(item[self.label_key]) #labels
        data["slide"] = torch.tensor(0)
        return data

    def _get_image(self, image_path: str):
        if self.cached_randaug:
            if self.split == "train":
                cur_epoch = int(os.environ.get('cur_epoch'))
                image_path = os.path.join(self.image_dir, "epoch_{}".format(cur_epoch), image_path)
            else:
                image_path = os.path.join(self.image_dir, "wo_augmentation", image_path)
        else:
            # need to comment out for mhu
            image_path = os.path.join(self.image_dir, image_path)

        image = self.loader(image_path)
        return self.transform(image)

    @staticmethod
    def _make_gs_index(pickle_split_path, k_fold, index_path, ignore, total_folds, split):
        items = [] # List of dict
        index_file = Path(index_path, f"{split}.index.{k_fold}.jsonl")
        if index_file.exists():
            print(f"[INFO] Skip: index file already exists. {index_file}")
            return

        Path(index_path).mkdir(parents=True, exist_ok=True)

        # Process CSV: extract fold and split information
        with open(pickle_split_path, 'rb') as picklefile:
            split_data = pickle.load(picklefile)
            if split == 'val':
                target_folds = [k_fold]
                name = "labels_val.csv"
            elif split == 'test':
                target_folds = [(k_fold + 1) % total_folds]
                name = "labels_test.csv"
            elif split == 'train':
                name = "labels_train.csv"
                target_folds = [i for i in range(total_folds) if i not in [k_fold, (k_fold + 1) % total_folds]]  # anything not in val or test
            counts = []
            for index, row in split_data.iterrows():
                if row["fold"] in target_folds:
                    slide_id = row['id_patient']
                    target_row = split_data.loc[split_data['id_patient'] == slide_id]
                    label = target_row['morphology'].values[0]
                    region_folder = os.path.join("", slide_id)
                    if Path(region_folder).exists():
                        regions = os.listdir(region_folder)
                        if len(regions) > 0:
                            counts.append(len(regions))
                            if split == 'train':
                                slide_data = {"image_path": slide_id,}
                                if target_row['morphology'].values[0] == 'GPC-clusters':
                                    label = 0
                                elif target_row['morphology'].values[0] == 'GPC-pairschains':
                                    label = 1
                                elif target_row['morphology'].values[0] == 'GNR':
                                    label = 2
                                elif target_row['morphology'].values[0] == 'GPR':
                                    label = 3
                                else:
                                    label = 4
                                slide_data.update({
                                        "label_morphology": label
                                })
                                slide_data.update({
                                        "slide_regions": regions
                                })
                                    items.append(slide_data)
                            elif split == "val" or split == "test":
                                slide_data = {"image_path": slide_id,}
                                if target_row['morphology'].values[0] == 'GPC-clusters':
                                    label = 0
                                elif target_row['morphology'].values[0] == 'GPC-pairschains':
                                    label = 1
                                elif target_row['morphology'].values[0] == 'GNR':
                                    label = 2
                                elif target_row['morphology'].values[0] == 'GPR':
                                    label = 3
                                else:
                                    label = 4
                                slide_data.update({
                                        "label_morphology": label
                                })
                                slide_data.update({
                                        "slide_regions": regions
                                })
                                items.append(slide_data)
        _write_data_into_jsonl(items, index_file)

    @classmethod
    def make_dataset_index(cls, pickle_split_path, k_fold, index_path, ignore, total_folds,):

        cls._make_gs_index(pickle_split_path=pickle_split_path, 
             k_fold=k_fold, index_path=index_path, ignore=ignore, 
             total_folds=total_folds, split="train",
        )
        cls._make_gs_index(pickle_split_path=pickle_split_path, 
             k_fold=k_fold, index_path=index_path, ignore=ignore,
             total_folds=total_folds, split="val",
        )
        cls._make_gs_index(pickle_split_path=pickle_split_path, 
             k_fold=k_fold, index_path=index_path, ignore=ignore, 
             total_folds=total_folds, split="test",
        )

task2dataset = {
    "gs_classification": GSClassificationDataset
}


def create_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, seq_parallel=False, seed=None):
    if is_train:
        batch_size =batch_size
        if seq_parallel:
            generator = torch.Generator()  
            generator.manual_seed(seed)
            sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
        else:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
            )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
        batch_size = batch_size
    
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )


def build_transform(is_train, args):

    if is_train:
        t = []
        if args.randaug:
            t += [
                 RandomResizedCropAndInterpolation(args.input_size, scale=(0.5, 1.0), interpolation=args.train_interpolation), 
                 transforms.RandomHorizontalFlip(),
            ]

        t += [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    return t


def create_dataset_by_split(args, split, is_train=True):
    transform = build_transform(is_train=is_train, args=args)
    print(transform)
    dataset_class = task2dataset[args.task]

    opt_kwargs = {}    
    opt_kwargs["k_fold"] = args.k_fold
    opt_kwargs["image_dir"] = args.image_dir
    opt_kwargs["seq_parallel"] = args.seq_parallel
    opt_kwargs["cached_randaug"] = args.cached_randaug

    dataset = dataset_class(
        data_path=args.data_path, split=split, 
        transform=transform, task=args.task, **opt_kwargs, 
    )
    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.0)

    return create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=args.num_workers, pin_mem=args.pin_mem,
        seq_parallel=args.seq_parallel, seed=args.seed,
    )


def create_downstream_dataset(args, is_eval=False):
    if is_eval:
        return create_dataset_by_split(args, split="test", is_train=False)
    else:
        return \
            create_dataset_by_split(args, split="train", is_train=True), \
            create_dataset_by_split(args, split="val", is_train=False),

