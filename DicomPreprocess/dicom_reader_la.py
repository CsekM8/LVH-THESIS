from Utils.utils import get_logger
import pydicom as dicom
import numpy as np
from numpy.linalg import norm
import os
from pydicom.pixel_data_handlers.util import apply_modality_lut

logger = get_logger(__name__)

class DCMreaderVMLa:

    def __init__(self, folder_name):
        self.broken = False
        self.ch2_frames = []
        self.ch3_frames = []
        self.ch4_frames = []
        self.ch2_file_paths = []
        self.ch3_file_paths = []
        self.ch4_file_paths = []
        self.ch2_frames_matrice = None
        self.ch3_frames_matrice = None
        self.ch4_frames_matrice = None

        dcm_files = sorted(os.listdir(folder_name))

        for idx, file in enumerate(dcm_files):
            if file.find('.dcm') != -1:
                try:
                    temp_ds = dicom.dcmread(os.path.join(folder_name, file))
                    self.classifyLaFrame(temp_ds, os.path.join(folder_name, file))
                except Exception as ex:
                    print('Couldnt read file: {}'.format(os.path.join(folder_name, file)))
                    print('Failed due to: ')
                    print(ex)
                    self.broken = True
                    return

        if len(self.ch2_frames) == 0 and len(self.ch3_frames) == 0 and len(self.ch4_frames) == 0:
            self.broken = True
            logger.warning("There are no frames. This folder should be deleted. Path: {}".format(folder_name))
        else:
            self.loadMatrices()

    def classifyLaFrame(self, ds, file_path):
        orientationDirCosines = ds.data_element('ImageOrientationPatient')
        orientNPArray = np.cross(orientationDirCosines[0:3], orientationDirCosines[3:6])
        ch2_direction = np.array([0.7692, 0.6184, 0.0081])
        ch3_direction = np.array([0.7335, 0.1403, 0.6574])
        ch4_direction = np.array([0.0144, -0.5744, 0.7982])
        windowedFrame = apply_modality_lut(ds.pixel_array, ds)
        cosOfAngle_ch2 = np.dot(orientNPArray, ch2_direction) / norm(ch2_direction) / norm(orientNPArray)
        cosOfAngle_ch3 = np.dot(orientNPArray, ch3_direction) / norm(ch3_direction) / norm(orientNPArray)
        cosOfAngle_ch4 = np.dot(orientNPArray, ch4_direction) / norm(ch4_direction) / norm(orientNPArray)
        cosofAngles = [abs(cosOfAngle_ch2), abs(cosOfAngle_ch3), abs(cosOfAngle_ch4)]
        minIdx = np.argmax(cosofAngles)
        if minIdx == 0:
            self.ch2_frames.append(windowedFrame)
            self.ch2_file_paths.append(file_path)
            return
        if minIdx == 1:
            self.ch3_frames.append(windowedFrame)
            self.ch3_file_paths.append(file_path)
            return
        if minIdx == 2:
            self.ch4_frames.append(windowedFrame)
            self.ch4_file_paths.append(file_path)
            return

    def loadMatrices(self):
        if len(self.ch2_frames) > 0:
            size_h, size_w = self.ch2_frames[0].shape
            self.ch2_frames_matrice = np.ones((len(self.ch2_frames), size_h, size_w))
            for i in range(len(self.ch2_frames)):
                if self.ch2_frames[i].shape == self.ch2_frames[0].shape:
                    self.ch2_frames_matrice[i] = self.ch2_frames[i]
                else:
                    logger.error('Wrong shape at {}'.format(self.ch2_file_paths[i]))
        if len(self.ch3_frames) > 0:
            size_h, size_w = self.ch3_frames[0].shape
            self.ch3_frames_matrice = np.ones((len(self.ch3_frames), size_h, size_w))
            for i in range(len(self.ch3_frames)):
                if self.ch3_frames[i].shape == self.ch3_frames[0].shape:
                    self.ch3_frames_matrice[i] = self.ch3_frames[i]
                else:
                    logger.error('Wrong shape at {}'.format(self.ch3_file_paths[i]))
        if len(self.ch4_frames) > 0:
            size_h, size_w = self.ch4_frames[0].shape
            self.ch4_frames_matrice = np.ones((len(self.ch4_frames), size_h, size_w))
            for i in range(len(self.ch4_frames)):
                if self.ch4_frames[i].shape == self.ch4_frames[0].shape:
                    self.ch4_frames_matrice[i] = self.ch4_frames[i]
                else:
                    logger.error('Wrong shape at {}'.format(self.ch4_file_paths[i]))

    def isBroken(self):
        return self.broken
