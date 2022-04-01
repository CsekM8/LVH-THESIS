import pickle
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

from DicomPreprocess.DataModel.patient import Patient, LaImageCollection

sourceFolder = 'D:/BME/7felev/Szakdolgozat/whole_dataset/csandras_data'
destinationFolder = 'D:/BME/7felev/Szakdolgozat/whole_dataset/filtered_data'

saMean = 0.2037
ch2Mean = 0.2279
ch3Mean = 0.2138
ch4Mean = 0.2217

transforms = transforms.Compose(
            [transforms.Resize([224, 224]), transforms.CenterCrop(168), transforms.ToTensor()])

for file in os.listdir(sourceFolder):
    if ".p" in file:
        tmpPat = pickle.load(open(os.path.join(sourceFolder, file), 'rb'))
        contrastSaImage = None
        if tmpPat.contrastSaImages is not None:
            bestDiffToMean = 0
            for i in range(tmpPat.contrastSaImages.shape[0]):
                for j in range(tmpPat.contrastSaImages.shape[1]):
                    if tmpPat.contrastSaImages[i][j].min() < 254:
                        img = Image.fromarray(tmpPat.contrastSaImages[i][j])
                        tmpMean = torch.mean(transforms(img))
                        if tmpMean <= saMean:
                            diffToMean = np.abs(saMean - tmpMean)
                            if diffToMean < bestDiffToMean:
                                bestDiffToMean = diffToMean
                                contrastSaImage = tmpPat.contrastSaImages[i][j]
        contrastCH2Image = None
        if tmpPat.contrastLaImages.ch2Images is not None:
            bestDiffToMean = 0
            for i in range(tmpPat.contrastLaImages.ch2Images.shape[0]):
                if tmpPat.contrastLaImages.ch2Images[i].min() < 254:
                    img = Image.fromarray(tmpPat.contrastLaImages.ch2Images[i])
                    tmpMean = torch.mean(transforms(img))
                    if tmpMean <= ch2Mean:
                        diffToMean = np.abs(ch2Mean - np.mean(tmpPat.contrastLaImages.ch2Images[i]))
                        if diffToMean < bestDiffToMean:
                            bestDiffToMean = diffToMean
                            contrastCH2Image = tmpPat.contrastLaImages.ch2Images[i]

        contrastCH3Image = None
        if tmpPat.contrastLaImages.ch3Images is not None:
            bestDiffToMean = 1000
            for i in range(tmpPat.contrastLaImages.ch3Images.shape[0]):
                if tmpPat.contrastLaImages.ch3Images[i].min() < 254:
                    img = Image.fromarray(tmpPat.contrastLaImages.ch3Images[i])
                    tmpMean = torch.mean(transforms(img))
                    if tmpMean <= ch3Mean:
                        diffToMean = np.abs(ch3Mean - np.mean(tmpPat.contrastLaImages.ch3Images[i]))
                        if diffToMean < bestDiffToMean:
                            bestDiffToMean = diffToMean
                            contrastCH3Image = tmpPat.contrastLaImages.ch3Images[i]

        contrastCH4Image = None
        if tmpPat.contrastLaImages.ch4Images is not None:
            bestDiffToMean = 1000
            for i in range(tmpPat.contrastLaImages.ch4Images.shape[0]):
                if tmpPat.contrastLaImages.ch4Images[i].min() < 254:
                    img = Image.fromarray(tmpPat.contrastLaImages.ch4Images[i])
                    tmpMean = torch.mean(transforms(img))
                    if tmpMean <= ch4Mean:
                        diffToMean = np.abs(ch4Mean - np.mean(tmpPat.contrastLaImages.ch4Images[i]))
                        if diffToMean < bestDiffToMean:
                            bestDiffToMean = diffToMean
                            contrastCH4Image = tmpPat.contrastLaImages.ch4Images[i]

        if (contrastSaImage is not None or contrastCH2Image is not None or
                contrastCH3Image is not None or contrastCH4Image is not None):
            filteredPatient = Patient(tmpPat.patientID, tmpPat.pathology, tmpPat.normalSaImages,
                                      contrastSaImage, tmpPat.normalLaImages,
                                      LaImageCollection(contrastCH2Image,
                                                        contrastCH3Image,
                                                        contrastCH4Image))
            os.makedirs(destinationFolder, exist_ok=True)
            pickle.dump(filteredPatient, open(os.path.join(destinationFolder, filteredPatient.patientID + '.p'), 'wb'))
