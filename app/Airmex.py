import dash
import os
import numpy as np
import cv2
from utils.preprocess_image import preprocess

class Airmex(dash.Dash):

    def __init__(self):
        super(Airmex, self).__init__(__name__)
        self.library_path = r'data/mendeley/kneeKL299/test'
        self.dicoms = {}


    def init_dicom_library(self):
        self.library = {}
        for label_folder in os.listdir(self.library_path):
            label_folder_path = os.path.join(self.library_path, label_folder)
            self.library[label_folder] = []
            for filename in os.listdir(label_folder_path):
                self.library[label_folder].append(filename)

    def load_dicoms(self, n=10):
        for label_folder, filenames in self.library.items():
            count = 0
            self.dicoms[label_folder] = []
            for filename in filenames:
                dicom = cv2.imread(os.path.join(self.library_path, label_folder, filename),0).astype(float)
                dicom = preprocess(dicom)
                self.dicoms[label_folder].append(dicom)
                count +=1
                if count == n:
                    break

    def select_xai(self):
        pass

    def select_dicom(self):
        pass

    def select_baselines(self):
        pass

    def select_projection(self):
        pass

    def run_xai(self):
        pass


if __name__ == '__main__':
    a = Airmex()
    a.init_dicom_library()
    a.load_dicoms()
    print('done')
    a.run_server()





