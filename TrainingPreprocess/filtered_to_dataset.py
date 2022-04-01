import os
import pickle
from PIL import Image


class PatientToImageFolder:

    def __init__(self, sourceFolder):
        self.sourceFolder = sourceFolder
        # How many patient with contrast SA for each pathology (used for classification)
        self.contrastSApathologyDict = {}
        # How many patient with contrast LA for each pathology (used for classification)
        self.contrastCH2pathologyDict = {}
        self.contrastCH3pathologyDict = {}
        self.contrastCH4pathologyDict = {}
        # How many patient with SA image (used for autoencoder training)
        self.totalSaImagePatientNum = 0
        self.curSaImagePatientNum = 0
        # How many patient with LA image (used for autoencoder training)
        self.totalCH2ImagePatientNum = 0
        self.curCH2ImagePatientNum = 0
        self.totalCH3ImagePatientNum = 0
        self.curCH3ImagePatientNum = 0
        self.totalCH4ImagePatientNum = 0
        self.curCH4ImagePatientNum = 0

        self.curContrastSaImagePatientNum = {}
        self.curContrastCH2ImagePatientNum = {}
        self.curContrastCH3ImagePatientNum = {}
        self.curContrastCH4ImagePatientNum = {}

        self.collectInfo()

    def collectInfo(self):
        for file in os.listdir(self.sourceFolder):
            if ".p" in file:
                tmpPat = pickle.load(open(os.path.join(self.sourceFolder, file), 'rb'))
                patho = tmpPat.pathology.strip()
                if "U18" in patho or "sport" in patho or "Normal" in patho:
                    continue
                # elif "sport" in patho:
                #     patho = "Sport"
                # elif "Normal" not in patho and "HCM" not in patho:
                #     patho = "Other"
                if tmpPat.normalSaImages is not None:
                    self.totalSaImagePatientNum += 1
                if (tmpPat.contrastSaImages is not None and tmpPat.contrastLaImages.ch2Images is not None and
                        tmpPat.contrastLaImages.ch3Images is not None and tmpPat.contrastLaImages.ch4Images is not None):
                    if patho in self.contrastSApathologyDict:
                        self.contrastSApathologyDict[patho] += 1
                    else:
                        self.contrastSApathologyDict[patho] = 1
                    if patho in self.contrastCH2pathologyDict:
                        self.contrastCH2pathologyDict[patho] += 1
                    else:
                        self.contrastCH2pathologyDict[patho] = 1
                    if patho in self.contrastCH3pathologyDict:
                        self.contrastCH3pathologyDict[patho] += 1
                    else:
                        self.contrastCH3pathologyDict[patho] = 1
                    if patho in self.contrastCH4pathologyDict:
                        self.contrastCH4pathologyDict[patho] += 1
                    else:
                        self.contrastCH4pathologyDict[patho] = 1
                if tmpPat.normalLaImages.ch2Images is not None:
                    self.totalCH2ImagePatientNum += 1
                if tmpPat.normalLaImages.ch3Images is not None:
                    self.totalCH3ImagePatientNum += 1
                if tmpPat.normalLaImages.ch4Images is not None:
                    self.totalCH4ImagePatientNum += 1

        for key in self.contrastSApathologyDict:
            self.curContrastSaImagePatientNum[key] = 0
        for key in self.contrastCH2pathologyDict:
            self.curContrastCH2ImagePatientNum[key] = 0
        for key in self.contrastCH3pathologyDict:
            self.curContrastCH3ImagePatientNum[key] = 0
        for key in self.contrastCH4pathologyDict:
            self.curContrastCH4ImagePatientNum[key] = 0

    def convertImage(self, image_2d):
        # if image_2d.min() > 254:
        #     return None
        # Converting image from numpy array to PIL.
        pil_img = Image.fromarray(image_2d)
        if pil_img.getbbox() is None:
            return None
        return pil_img

    def createAutoEncoderImageFolderStructure(self, folderName):
        autoFolder = os.path.join(os.path.dirname(self.sourceFolder), folderName)
        autoTrainingFolder = os.path.join(autoFolder, "training")
        autoTestFolder = os.path.join(autoFolder, "test")

        os.makedirs(autoTrainingFolder)
        os.makedirs(autoTestFolder)

        return autoFolder, autoTrainingFolder, autoTestFolder

    def createClassificationImageFolderStructure(self, folderName):
        classFolder = os.path.join(os.path.dirname(self.sourceFolder), folderName)
        classTrainingFolder = os.path.join(classFolder, "training")
        classValidationFolder = os.path.join(classFolder, "validation")
        classTestFolder = os.path.join(classFolder, "test")
        classAllFolder = os.path.join(classFolder, 'all')

        os.makedirs(classTrainingFolder)
        os.makedirs(classValidationFolder)
        os.makedirs(classTestFolder)
        os.makedirs(classAllFolder)

        return classFolder, classTrainingFolder, classValidationFolder, classTestFolder, classAllFolder

    def saveImageForClassification(self, image, patientId, patho, testFolder, validationFolder, trainingFolder,
                                   axis, imPatho, curPatientNum, allFolder, pathologyDict):
        pil_img = self.convertImage(image[:, :])
        if pil_img is not None:
            if (curPatientNum[patho] <= pathologyDict[patho] * 0.075 or
                    (pathologyDict[patho] * 0.85 <= curPatientNum[patho] <= pathologyDict[patho] * 0.925)):
                imFolder = os.path.join(testFolder, imPatho)
                os.makedirs(imFolder, exist_ok=True)
                patientFolder = os.path.join(self.patientSeperatedTestFolder, imPatho + '_' + patientId)
                os.makedirs(patientFolder, exist_ok=True)
            elif ((pathologyDict[patho] * 0.075 <= curPatientNum[patho] <= pathologyDict[patho] * 0.15) or
                  curPatientNum[patho] >= int(pathologyDict[patho] * 0.925)):
                imFolder = os.path.join(validationFolder, imPatho)
                os.makedirs(imFolder, exist_ok=True)
                patientFolder = os.path.join(self.patientSeperatedValidationFolder, imPatho + '_' + patientId)
                os.makedirs(patientFolder, exist_ok=True)
            else:
                imFolder = os.path.join(trainingFolder, imPatho)
                os.makedirs(imFolder, exist_ok=True)
                patientFolder = os.path.join(self.patientSeperatedTrainingFolder, imPatho + '_' + patientId)
                os.makedirs(patientFolder, exist_ok=True)
            axisFolder = os.path.join(patientFolder, axis)
            os.makedirs(axisFolder, exist_ok=True)
            pil_img.save(os.path.join(imFolder, "{}.png".format(patientId)))
            # pil_img.save(os.path.join(allFolder, "{}.png".format(patientId)))
            pil_img.save(os.path.join(axisFolder, "{}.png".format(patientId)))
            file = open(os.path.join(patientFolder, "pathology.txt"), "w")
            file.write("{}\n".format(patho))
            file.close()

    def saveImageForAutoEncoder(self, images, patientId, testFolder, trainingFolder,
                                curPatientNum, totalPatientNum, sliceIdx, frameIdx):
        if sliceIdx is not None:
            pil_img = self.convertImage(images[sliceIdx, frameIdx, :, :])
        else:
            pil_img = self.convertImage(images[frameIdx, :, :])
        if pil_img is not None:
            if (curPatientNum <= totalPatientNum * 0.1
                    or curPatientNum >= int(totalPatientNum * 0.9)):
                if sliceIdx is not None:
                    pil_img.save(os.path.join(testFolder, "{}_{}_{}.png".format(patientId, sliceIdx, frameIdx)))
                else:
                    pil_img.save(os.path.join(testFolder, "{}_{}.png".format(patientId, frameIdx)))
            else:
                if sliceIdx is not None:
                    pil_img.save(os.path.join(trainingFolder, "{}_{}_{}.png".format(patientId, sliceIdx, frameIdx)))
                else:
                    pil_img.save(os.path.join(trainingFolder, "{}_{}.png".format(patientId, frameIdx)))

    def createImageFolderDatasets(self):
        subfol = "only_abnormal"
        # autoSaFolder, autoSaTrainingFolder, autoSaTestFolder = self.createAutoEncoderImageFolderStructure(
        #     "SaAutoEncoder")

        (contrastSaFolder, contrastSaTrainingFolder,
         contrastSaValidationFolder, contrastSaTestFolder,
         contrastSaAllFolder) = self.createClassificationImageFolderStructure(
            "{}/SaClassification".format(subfol))

        # autoCH2Folder, autoCH2TrainingFolder, autoCH2TestFolder = self.createAutoEncoderImageFolderStructure(
        #     "CH2AutoEncoder")

        (contrastCH2Folder, contrastCH2TrainingFolder,
         contrastCH2ValidationFolder, contrastCH2TestFolder,
         contrastCH2AllFolder) = self.createClassificationImageFolderStructure(
            "{}/CH2Classification".format(subfol))

        # autoCH3Folder, autoCH3TrainingFolder, autoCH3TestFolder = self.createAutoEncoderImageFolderStructure(
        #     "CH3AutoEncoder")

        (contrastCH3Folder, contrastCH3TrainingFolder,
         contrastCH3ValidationFolder, contrastCH3TestFolder,
         contrastCH3AllFolder) = self.createClassificationImageFolderStructure(
            "{}/CH3Classification".format(subfol))

        # autoCH4Folder, autoCH4TrainingFolder, autoCH4TestFolder = self.createAutoEncoderImageFolderStructure(
        #     "CH4AutoEncoder")

        (contrastCH4Folder, contrastCH4TrainingFolder,
         contrastCH4ValidationFolder, contrastCH4TestFolder,
         contrastCH4AllFolder) = self.createClassificationImageFolderStructure(
            "{}/CH4Classification".format(subfol))

        self.patientSeperatedFolder = os.path.join(os.path.dirname(self.sourceFolder), '{}/patients'.format(subfol))
        os.makedirs(self.patientSeperatedFolder)
        self.patientSeperatedTrainingFolder = os.path.join(self.patientSeperatedFolder, 'training')
        self.patientSeperatedValidationFolder = os.path.join(self.patientSeperatedFolder, 'validation')
        self.patientSeperatedTestFolder = os.path.join(self.patientSeperatedFolder, 'test')
        os.makedirs(self.patientSeperatedTrainingFolder)
        os.makedirs(self.patientSeperatedValidationFolder)
        os.makedirs(self.patientSeperatedTestFolder)

        for file in os.listdir(self.sourceFolder):
            if ".p" in file:
                tmpPat = pickle.load(open(os.path.join(self.sourceFolder, file), 'rb'))
                patho = tmpPat.pathology.strip()
                if "U18" in patho or "sport" in patho or "Normal" in patho:
                    continue
                # elif "sport" in patho:
                #     patho = "Sport"
                # elif "Normal" not in patho and "HCM" not in patho:
                #     patho = "Other"

                imPatho = patho
                # if "sport" in patho:
                #     imPatho = "Sport"
                # if "Normal" not in patho:
                #     imPatho = "Hypertrophic"

                classificationReady = False
                if (tmpPat.contrastSaImages is not None and tmpPat.contrastLaImages.ch2Images is not None and
                        tmpPat.contrastLaImages.ch3Images is not None and tmpPat.contrastLaImages.ch4Images is not None):
                    classificationReady = True

                # if tmpPat.normalSaImages is not None:
                #     for i in range(tmpPat.normalSaImages.shape[0]):
                #         for j in range(tmpPat.normalSaImages.shape[1]):
                #             self.saveImageForAutoEncoder(tmpPat.normalSaImages, tmpPat.patientID, autoSaTestFolder,
                #                                          autoSaTrainingFolder, self.curSaImagePatientNum,
                #                                          self.totalSaImagePatientNum, i, j)
                #     self.curSaImagePatientNum += 1

                if classificationReady:
                    self.saveImageForClassification(tmpPat.contrastSaImages, tmpPat.patientID, patho,
                                                    contrastSaTestFolder, contrastSaValidationFolder,
                                                    contrastSaTrainingFolder, 'SA', imPatho,
                                                    self.curContrastSaImagePatientNum, contrastSaAllFolder,
                                                    self.contrastSApathologyDict)
                    self.curContrastSaImagePatientNum[patho] += 1

                # if tmpPat.normalLaImages.ch2Images is not None:
                #     for i in range(tmpPat.normalLaImages.ch2Images.shape[0]):
                #         self.saveImageForAutoEncoder(tmpPat.normalLaImages.ch2Images, tmpPat.patientID,
                #                                      autoCH2TestFolder,
                #                                      autoCH2TrainingFolder, self.curCH2ImagePatientNum,
                #                                      self.totalCH2ImagePatientNum, None, i)
                #     self.curCH2ImagePatientNum += 1

                if classificationReady:
                    self.saveImageForClassification(tmpPat.contrastLaImages.ch2Images, tmpPat.patientID, patho,
                                                    contrastCH2TestFolder, contrastCH2ValidationFolder,
                                                    contrastCH2TrainingFolder, 'CH2', imPatho,
                                                    self.curContrastCH2ImagePatientNum, contrastCH2AllFolder,
                                                    self.contrastCH2pathologyDict)
                    self.curContrastCH2ImagePatientNum[patho] += 1

                # if tmpPat.normalLaImages.ch3Images is not None:
                #     for i in range(tmpPat.normalLaImages.ch3Images.shape[0]):
                #         self.saveImageForAutoEncoder(tmpPat.normalLaImages.ch3Images, tmpPat.patientID,
                #                                      autoCH3TestFolder,
                #                                      autoCH3TrainingFolder, self.curCH3ImagePatientNum,
                #                                      self.totalCH3ImagePatientNum, None, i)
                #     self.curCH3ImagePatientNum += 1

                if classificationReady:
                    self.saveImageForClassification(tmpPat.contrastLaImages.ch3Images, tmpPat.patientID, patho,
                                                    contrastCH3TestFolder, contrastCH3ValidationFolder,
                                                    contrastCH3TrainingFolder, 'CH3', imPatho,
                                                    self.curContrastCH3ImagePatientNum, contrastCH3AllFolder,
                                                    self.contrastCH3pathologyDict)
                    self.curContrastCH3ImagePatientNum[patho] += 1

                # if tmpPat.normalLaImages.ch4Images is not None:
                #     for i in range(tmpPat.normalLaImages.ch4Images.shape[0]):
                #         self.saveImageForAutoEncoder(tmpPat.normalLaImages.ch4Images, tmpPat.patientID,
                #                                      autoCH4TestFolder,
                #                                      autoCH4TrainingFolder, self.curCH4ImagePatientNum,
                #                                      self.totalCH4ImagePatientNum, None, i)
                #     self.curCH4ImagePatientNum += 1

                if classificationReady:
                    self.saveImageForClassification(tmpPat.contrastLaImages.ch4Images, tmpPat.patientID, patho,
                                                    contrastCH4TestFolder, contrastCH4ValidationFolder,
                                                    contrastCH4TrainingFolder, 'CH4', imPatho,
                                                    self.curContrastCH4ImagePatientNum, contrastCH4AllFolder,
                                                    self.contrastCH4pathologyDict)
                    self.curContrastCH4ImagePatientNum[patho] += 1

        self.createLabelFileFromPathoDict(contrastSaFolder, self.contrastSApathologyDict)
        self.createLabelFileFromPathoDict(contrastCH2Folder, self.contrastCH2pathologyDict)
        self.createLabelFileFromPathoDict(contrastCH3Folder, self.contrastCH3pathologyDict)
        self.createLabelFileFromPathoDict(contrastCH4Folder, self.contrastCH4pathologyDict)

    def createLabelFileFromPathoDict(self, destination, pathoDict):
        file = open(os.path.join(destination, "pathologies.txt"), "w")
        for key in pathoDict:
            file.write("{}\n".format(key))
        file.close()


if __name__ == "__main__":
    sourceFolder = 'D:/BME/7felev/Szakdolgozat/whole_dataset/filtered_data'
    imageFolderArranger = PatientToImageFolder(sourceFolder)
    imageFolderArranger.createImageFolderDatasets()
