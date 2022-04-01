import os
from DicomPreprocess.dicom_reader_sa import DCMreaderVMSa
from DicomPreprocess.dicom_reader_la import DCMreaderVMLa
from DicomPreprocess.DataModel.patient import Patient
from DicomPreprocess.DataModel.patient import LaImageCollection
import numpy as np
import pickle


# Data collection mainly based on dcm files, and meta.txt (for pathology).

class DataCollector:

    def __init__(self, rootFolder="none"):
        self.patients = []
        self.requiredSaNormalImageSliceCount = 5
        self.requiredSaNormalImagePerSliceCount = 3
        self.requiredSaContrastImageSliceCount = 5
        self.requiredSaContrastImagePerSliceCount = 3
        self.requiredLaNormalImagePerChamberView = 15
        self.requiredLaContrastImagePerChamberView = 15

        if rootFolder != "none":
            patientIDs = os.listdir(rootFolder)

            for patient in patientIDs:
                dcmReaderSaNormal = None
                dcmReaderSaContrast = None
                dcmReaderLaNormal = None
                dcmReaderLaContrast = None
                normalSaImages = None
                contrastSaImages = None
                normalLaImages = dict(
                    CH2Images=None,
                    CH4Images=None,
                    CH3Images=None
                )
                contrastLaImages = dict(
                    CH2Images=None,
                    CH4Images=None,
                    CH3Images=None
                )
                pathology = 'X'
                print('Processing dcm file for patient: ' + patient)
                for root, dirs, files in os.walk(os.path.join(rootFolder, patient)):
                    if 'meta.txt' in files:
                        with open(os.path.join(root, 'meta.txt'), 'rt') as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                content = line.strip().split(':')
                            pathology = content[1]
                    if 'sa' in dirs:
                        joinedSaPath = os.path.join(root, 'sa')
                        saListDir = os.listdir(joinedSaPath)
                        #print('sa')
                        if saListDir:
                            if 'images' in saListDir:
                                dcmReaderSaNormal = DCMreaderVMSa(os.path.join(joinedSaPath, 'images'))
                            else:
                                dcmReaderSaNormal = DCMreaderVMSa(joinedSaPath)
                    if 'sale' in dirs:
                        joinedSaLePath = os.path.join(root, 'sale')
                        #print('sale')
                        if os.listdir(joinedSaLePath):
                            dcmReaderSaContrast = DCMreaderVMSa(joinedSaLePath)
                    if 'la' in dirs:
                        joinedLaPath = os.path.join(root, 'la')
                        #print('la')
                        if os.listdir(joinedLaPath):
                            dcmReaderLaNormal = DCMreaderVMLa(joinedLaPath)
                    if 'lale' in dirs:
                        joinedLaLePath = os.path.join(root, 'lale')
                        #print('lale')
                        if os.listdir(joinedLaLePath):
                            dcmReaderLaContrast = DCMreaderVMLa(joinedLaLePath)

                if dcmReaderSaNormal is not None and not dcmReaderSaNormal.isBroken():
                    normalSaImages = self.createSaImageSubMatrice(dcmReaderSaNormal, self.requiredSaNormalImageSliceCount,
                                              self.requiredSaNormalImagePerSliceCount)

                if dcmReaderSaContrast is not None and not dcmReaderSaContrast.isBroken() and pathology != 'X':
                    contrastSaImages = self.createSaImageSubMatrice(dcmReaderSaContrast, self.requiredSaContrastImageSliceCount,
                                              self.requiredSaContrastImagePerSliceCount)

                if dcmReaderLaNormal is not None and not dcmReaderLaNormal.isBroken():
                    self.fillLaImageMatriceDict(dcmReaderLaNormal, self.requiredLaNormalImagePerChamberView,
                                                normalLaImages)

                if dcmReaderLaContrast is not None and not dcmReaderLaContrast.isBroken() and pathology != 'X':
                    self.fillLaImageMatriceDict(dcmReaderLaContrast, self.requiredLaContrastImagePerChamberView,
                                                contrastLaImages)

                self.patients.append(Patient(patient, pathology,
                                             normalSaImages, contrastSaImages,
                                             LaImageCollection(normalLaImages['CH2Images'],
                                                               normalLaImages['CH3Images'],
                                                               normalLaImages['CH4Images']),
                                             LaImageCollection(contrastLaImages['CH2Images'],
                                                               contrastLaImages['CH3Images'],
                                                               contrastLaImages['CH4Images'])))
        print('Processing of dcm files done. Check hypertrophy.log for possible errors.')

    def createSaImageSubMatrice(self, saReader, requiredSaSliceCount, requiredSaImagePerSliceCount):
        if saReader.getSliceNum() < requiredSaSliceCount:
            requiredSaSliceCount = saReader.getSliceNum()
        if saReader.getFrameNum() < requiredSaImagePerSliceCount:
            requiredSaImagePerSliceCount = saReader.getFrameNum()
        imageSliceStep = int(np.ceil(saReader.getSliceNum() / requiredSaSliceCount / 2))
        halfPointOfSlices = int(saReader.getSliceNum() / 2)
        sliceCountFromOneHalf = int(requiredSaSliceCount / 2)
        startSliceIdx = halfPointOfSlices - imageSliceStep * sliceCountFromOneHalf
        h, w = saReader.get_image(0, 0).shape
        saImages = np.empty((requiredSaSliceCount, requiredSaImagePerSliceCount, h, w), dtype=np.uint8)
        for i in range(requiredSaSliceCount):
            for k in range(imageSliceStep):
                curSlice = saReader.get_imagesOfSlice(startSliceIdx + i * imageSliceStep + k)
                firstFrameInSlice = self.convertImage(curSlice[0, :, :])
                if (firstFrameInSlice is not None
                        or startSliceIdx + i * imageSliceStep + k + 1 >= saReader.getSliceNum()
                        or k == imageSliceStep - 1):
                    imageStep = int(curSlice.shape[0] / requiredSaImagePerSliceCount)
                    for j in range(requiredSaImagePerSliceCount):
                        tmpConvertedImg = self.convertImage(curSlice[j * imageStep, :, :])
                        if tmpConvertedImg is not None:
                            saImages[i][j] = tmpConvertedImg
                        else:
                            saImages[i][j].fill(255)
                    break
        return saImages

    def fillLaImageMatriceDict(self, laReader, requiredImagePerChamberView, fillableLaDict):
        if laReader.ch2_frames_matrice is not None:
            fillableLaDict['CH2Images'] = self.createLaImageSubMatrice(laReader.ch2_frames_matrice,
                                                                             requiredImagePerChamberView)
        if laReader.ch3_frames_matrice is not None:
            fillableLaDict['CH3Images'] = self.createLaImageSubMatrice(laReader.ch3_frames_matrice,
                                                                             requiredImagePerChamberView)
        if laReader.ch4_frames_matrice is not None:
            fillableLaDict['CH4Images'] = self.createLaImageSubMatrice(laReader.ch4_frames_matrice,
                                                                             requiredImagePerChamberView)

    def createLaImageSubMatrice(self, ch_frame_matrice, requiredImagePerChamberView):
        if ch_frame_matrice.shape[0] < requiredImagePerChamberView:
            requiredImagePerChamberView = ch_frame_matrice.shape[0]
        imageStep = int(ch_frame_matrice.shape[0] / requiredImagePerChamberView)
        h, w = ch_frame_matrice[0].shape
        chImages = np.empty((requiredImagePerChamberView, h, w), dtype=np.uint8)
        for i in range(requiredImagePerChamberView):
            for j in range(imageStep):
                tmpConvertedImg = self.convertImage(ch_frame_matrice[i * imageStep + j, :, :])
                if tmpConvertedImg is not None:
                    chImages[i] = tmpConvertedImg
                    break
                elif j == imageStep - 1 or i * imageStep + j + 1 >= ch_frame_matrice.shape[0]:
                    chImages[i].fill(255)
                    break
        return chImages


    def convertImage(self, image_2d):

        # Convert to float to avoid overflow or underflow losses.
        image_2d_f = image_2d.astype(float)

        # image is not blank
        if image_2d_f.min() < 0.99 and image_2d_f.max() > 0.01:

            # Rescaling grey scale between 0-255
            image_2d_scaled = (np.maximum(image_2d_f, 0) / image_2d_f.max()) * 255.0

            # Convert to uint
            image_2d_scaled = np.uint8(image_2d_scaled)

            return image_2d_scaled

        else:
            return None

    def serializePatients(self, destinationFolder):
        print('Pickling data')
        os.makedirs(destinationFolder, exist_ok=True)
        for patient in self.patients:
            pickle.dump(patient, open(os.path.join(destinationFolder, patient.patientID + '.p'), 'wb'))
        print('Serialization done. Files can be found at: ' + destinationFolder)

    def deserializePatients(self, sourceFolder):
        for file in os.listdir(sourceFolder):
            if '.p' in file:
                self.patients.append(pickle.load(open(os.path.join(sourceFolder, file), 'rb')))
