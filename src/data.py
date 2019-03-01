import torch
import torch.utils
import torch.utils.data
import numpy as np
import scipy.ndimage
import scipy.misc
import random
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Silence imresize warning


def _my_open(name):
    if name == "-":
        return sys.stdin
    else:
        return open(name)
    

class AchromaticDataset(torch.utils.data.Dataset):
    """Produces luminance, label images."""
    def __init__(self, list_file, training, size, include_path=False, include_fullres=False):
        self.data = []
        with _my_open(list_file) as f:
            for line in f:
                fields = line.split()
                if not fields:
                    continue
                illuminant = (tuple(map(float, fields[1:4])) if len(fields) == 4 else (1., 1., 1.))
                self.data.append((fields[0], illuminant))
        self.training = training
        self.size = size
        self.include_path = include_path
        self.include_fullres = include_fullres

    def __len__(self):
        return len(self.data)

    def _pad(self, image):
        h, w, _ = image.shape
        if min(h, w) >= self.size:
            return image
        pads = [(0, max(0, self.size - h)), (0, max(0, self.size - w)), (0, 0)]
        return np.pad(image, pads, "constant")

    def _random_crop(self, image):
        h, w, _ = image.shape
        if max(h, w) <= self.size:
            return image
        i = random.randint(0, h - self.size - 1) if h > self.size else 0
        j = random.randint(0, w - self.size - 1) if w > self.size else 0
        return image[i:i + self.size, j:j + self.size, :]

    def _resize(self, image):
        return scipy.misc.imresize(image, (self.size, self.size), "bilinear")
    
    def _random_flip(self, image):
        return (image[:, ::-1, :] if random.random() > 0.5 else image)
        
    def __getitem__(self, index):
        filename, illuminant = self.data[index]
        fullres = scipy.ndimage.imread(filename, mode="RGB")
        image = self._resize(fullres)
        if self.training:
            image = self._random_flip(image)
        image = image.astype(np.float32) / 255.0
        rgb = np.transpose(image, [2, 0, 1])
        t_rgb = torch.tensor(rgb, dtype=torch.float)
        t_illuminant = torch.tensor(illuminant, dtype=torch.float)
        ret = (t_rgb, t_illuminant)
        if self.include_path:
            ret = ret + (filename, )
        if self.include_fullres:
            fr = np.transpose(fullres.astype(np.float32) / 255.0, [2, 0, 1])
            ret = ret + (torch.tensor(fr, dtype=torch.float),)
        return ret


def _test():
    dataset = AchromaticDataset("../data/train.txt", True, 256)
    loader = torch.utils.data.DataLoader(dataset, 10, shuffle=True)
    for rgb, illuminant in loader:
        print(rgb.size(), illuminant.size())


if __name__ == '__main__':
    _test()
