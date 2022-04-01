from Utils.utils import get_logger
import pydicom as dicom
import numpy as np
import os
from pydicom.pixel_data_handlers.util import apply_modality_lut

logger = get_logger(__name__)


class DCMreaderVMSa:

    def __init__(self, folder_name):
        '''
        Reads in the dcm files in a folder which corresponds to a patient.
        It follows carefully the physical slice locations and the frames in a hearth cycle.
        It does not matter if the location is getting higher or lower.
        '''
        self.num_slices = 0
        self.num_frames = 0
        self.broken = False
        images = []
        slice_locations = []
        file_paths = []

        dcm_files = sorted(os.listdir(folder_name))
        dcm_files = [d for d in dcm_files if len(d.split('.')[-2]) < 4]
        if len(dcm_files) == 0:  # sometimes the order number is missing at the end
            dcm_files = sorted(os.listdir(folder_name))

        for file in dcm_files:
            if file.find('.dcm') != -1:
                try:
                    temp_ds = dicom.dcmread(os.path.join(folder_name, file))
                    windowed = apply_modality_lut(temp_ds.pixel_array, temp_ds)
                    images.append(windowed)
                    slice_locations.append(temp_ds.SliceLocation)
                    file_paths.append(os.path.join(folder_name, file))
                except:
                    print('Couldnt parse file: {}'.format(file))
                    self.broken = True
                    return

        current_sl = -1
        frames = 0
        increasing = False
        indices = []
        for idx, slice_loc in enumerate(slice_locations):
            if abs(slice_loc - current_sl) > 0.01:  # this means a new slice is started
                self.num_slices += 1
                self.num_frames = max(self.num_frames, frames)
                frames = 0
                indices.append(idx)

                if (slice_loc - current_sl) > 0.01:
                    increasing = True
                else:
                    increasing = False

                current_sl = slice_loc
            frames += 1
        self.num_frames = max(self.num_frames, frames)

        if self.num_slices != 0 and self.num_frames != 0:
            self.load_matrices(images, indices, increasing, slice_locations, file_paths)
        else:
            self.broken = True
            logger.warning("There are no frames. This folder should be deleted. Path: {}".format(folder_name))

    def load_matrices(self, images, indices, increasing, slice_locations, file_paths):
        size_h, size_w = images[0].shape
        self.dcm_images = np.ones((self.num_slices, self.num_frames, size_h, size_w))
        self.dcm_slicelocations = np.ones((self.num_slices, self.num_frames, 1))
        self.dcm_file_paths = np.zeros((self.num_slices, self.num_frames), dtype=object)

        for i in range(len(indices) - 1):

            for idx in range(indices[i], indices[i + 1]):
                slice_idx = (i if increasing else (len(indices) - 1 - i))
                frame_idx = idx - indices[i]
                if images[idx].shape == self.dcm_images[slice_idx, frame_idx, :, :].shape:
                    self.dcm_images[slice_idx, frame_idx, :, :] = images[idx]
                    self.dcm_slicelocations[slice_idx, frame_idx, 0] = slice_locations[idx]
                    self.dcm_file_paths[slice_idx, frame_idx] = file_paths[idx]
                else:
                    logger.error('Wrong shape at {}'.format(file_paths[idx]))

        for idx in range(indices[-1], len(images)):
            slice_idx = (len(indices) - 1 if increasing else 0)
            frame_idx = idx - indices[-1]
            if self.dcm_images.shape[1] == frame_idx:
                logger.info(file_paths[idx])
            if images[idx].shape == self.dcm_images[slice_idx, frame_idx, :, :].shape:
                self.dcm_images[slice_idx, frame_idx, :, :] = images[idx]
                self.dcm_slicelocations[slice_idx, frame_idx, 0] = slice_locations[idx]
                self.dcm_file_paths[slice_idx, frame_idx] = file_paths[idx]
            else:
                logger.error('Wrong shape at {}'.format(file_paths[idx]))

    def get_image(self, slice, frame):
        return self.dcm_images[slice, frame, :, :]

    def get_imagesOfSlice(self, slice):
        return self.dcm_images[slice, :, :, :]

    def get_slicelocation(self, slice, frame):
        return self.dcm_slicelocations[slice, frame, 0]

    def get_dcm_path(self, slice, frame):
        return self.dcm_file_paths[slice, frame]

    def getSliceNum(self):
        return self.num_slices

    def getFrameNum(self):
        return self.num_frames

    def isBroken(self):
        return self.broken