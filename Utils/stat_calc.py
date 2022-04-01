import numpy as np
import torch

from AutoEncoder.ae_dataset import AEDataset
from torch.utils.data import DataLoader

targetSet = AEDataset('D:\BME/7felev/Szakdolgozat/whole_dataset/CH4Classification/all', contrastTest=True)

targetLoader = DataLoader(
    dataset=targetSet,
    batch_size=128,
    shuffle=True
)


def calc_quantile(loader, q):
    quantile, batch_num = 0, 0
    for data in loader:
        quantile += np.quantile(data, q)
        batch_num += 1
    return quantile / batch_num


def calc_mean_std(loader):
    mean_sum, mean_square_sum, batch_num = 0, 0, 0
    for data in loader:
        mean_sum += torch.mean(data, dim=[0, 2, 3])
        mean_square_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        batch_num += 1
    mean = mean_sum / batch_num
    std = (mean_square_sum / batch_num - mean ** 2) ** 0.5

    return mean, std


# q = 0.9
# quantile = calc_quantile(targetLoader, q)
# print('{}th quantile of the dataset: {}'.format(q, quantile))

mean, std = calc_mean_std(targetLoader)
print('Mean of the dataset: {}'.format(mean))
print('Standard deviation of the dataset: {}'.format(std))
