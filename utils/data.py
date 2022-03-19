import pandas as pd
from torch.utils import data
from PIL import Image
import os


class Annotation(object):
    """ annotate ISIC 2017

    Attributes:
        df(pd.DataFrame): df.columns=['image_id', 'label']
        categories(list): dermatological types
        class_dict(dict): class name -> index
        label_dict(dict): index -> class name
        class_num(int): the number of classes
        
    Usages:
        count_samples(): get numbers of samples in each class

    """
    def __init__(self, ann_file: str) -> None:
        """
        Args:
            ann_file (str): csv file path
        """
        self.df = pd.read_csv(ann_file, header=0)
        self.df['benign'] = 1 - self.df.select_dtypes(['number']).sum(axis=1)
        self.categories = list(self.df.columns)
        self.categories.pop(0)
        self.class_num = len(self.categories)
        self.class_dict, self.label_dict = self._make_dicts()
        self.df = self._relabel()
        # self.class_nums = self.count_samples()

    def _make_dicts(self):
        """ make class and label dict from categories' names """
        class_dict = {}
        label_dict = {}
        for i, name in enumerate(self.categories):
            class_dict[name] = i
            label_dict[i] = name

        return class_dict, label_dict

    def _relabel(self) -> pd.DataFrame:
        self.df['label'] = self.df.select_dtypes(['number']).idxmax(axis=1)
        self.df['label'] = self.df['label'].apply(lambda x: self.class_dict[x])
        for name in self.categories:
            del self.df[name]
        return self.df

    def count_samples(self) -> list:
        """ count sample_nums """
        value_counts = self.df.iloc[:, 1].value_counts()
        class_nums = [value_counts[i] for i in range(len(value_counts))]
        return class_nums

    def to_names(self, nums):
        """ convert a goup of indices to string names 
        
        Args:
            nums(torch.Tensor): a list of number labels

        Return:
            a list of dermatological names
        
        """
        names = [self.label_dict[int(num)] for num in nums]
        return names


class Data(data.Dataset):
    def __init__(self, annotations: pd.DataFrame, img_dir: str, transform=None, target_transform=None):
        self.img_labels = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.img_labels.image_id[idx] + '.jpg')
        image = Image.open(img_path)
        target = self.img_labels['label'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target