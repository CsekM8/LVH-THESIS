from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch


# def min_max_normalization(tensor, min_value, max_value):
#     min_tensor = tensor.min()
#     tensor = (tensor - min_tensor)
#     max_tensor = tensor.max()
#     tensor = tensor / max_tensor
#     tensor = tensor * (max_value - min_value) + min_value
#     return tensor


class AEDataset(Dataset):

    def extraTransforms(self, inputTensor):
        print('extra transform started')
        noisedTensor = inputTensor + torch.randn_like(inputTensor) * 0.09
        noisedTensor = torch.clamp(noisedTensor, 0., 1.)
        concatedTensor = torch.cat((inputTensor, noisedTensor), 2)
        save_image(concatedTensor, './transform_test/transformTest_{}.png'.format(self.imgCntr))
        self.imgCntr += 1
        print('extra transform finished')
        return noisedTensor

    def __init__(self, sourceFolder, contrastTest=False):
        self.patientImages = []
        self.imgCntr = 0
        self.contrastTest = contrastTest

        for file in os.listdir(sourceFolder):
            if '.png' in file:
                self.patientImages.append(os.path.join(sourceFolder, file))

        self.transforms = transforms.Compose(
            [transforms.Resize([224, 224]), transforms.ToTensor()])

    def __getitem__(self, index):

        normal_img = Image.open(self.patientImages[index])

        normal_img_tensor = self.transforms(normal_img)

        # if not self.contrastTest:
        #     final_tensor = self.extraTransforms(normal_img_tensor)
        #     return final_tensor

        return normal_img_tensor

    def __len__(self):
        return len(self.patientImages)
