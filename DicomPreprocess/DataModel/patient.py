class LaImageCollection:
    def __init__(self, ch2Images, ch3Images, ch4Images):
        self.ch2Images = ch2Images
        self.ch3Images = ch3Images
        self.ch4Images = ch4Images

class Patient:
    def __init__(self, patientID, pathology, normalSaImages, contrastSaImages, normalLaImages, contrastLaImages):
        self.patientID = patientID
        self.pathology = pathology
        self.normalSaImages = normalSaImages
        self.contrastSaImages = contrastSaImages
        self.normalLaImages = normalLaImages
        self.contrastLaImages = contrastLaImages
