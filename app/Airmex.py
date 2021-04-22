import dash
import os
import numpy as np
import cv2
from utils.preprocess_image import preprocess, normalize_minmax
from utils.model import rebuild_kneenet
from captum.attr import DeepLift, GuidedGradCam
import torch.nn as nn
import torch
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

class Airmex(dash.Dash):

    def __init__(self, xai=DeepLift):
        super(Airmex, self).__init__(__name__)
        # self.library_path = r'data/mendeley/kneeKL299/test'
        self.library_path = r'../data/mendeley/kneeKL299/test'
        self.images = {}
        self.init_image_library()
        self.load_images()
        self.model = rebuild_kneenet()
        self.select_image()
        self.select_xai(xai)
        self.select_baseline()
        self.attr = self.run_xai()
        self.run_projection()
        self.set_layout()

    def init_image_library(self):
        self.library = {}
        for label_folder in os.listdir(self.library_path):
            label_folder_path = os.path.join(self.library_path, label_folder)
            self.library[label_folder] = []
            for filename in os.listdir(label_folder_path):
                self.library[label_folder].append(filename)

    def load_images(self, n=10):
        for label_folder, filenames in self.library.items():
            count = 0
            self.images[label_folder] = []
            for filename in filenames:
                image = cv2.imread(os.path.join(self.library_path, label_folder, filename), 0)
                image = preprocess(image)
                image = torch.from_numpy(image).float().unsqueeze(0)
                self.images[label_folder].append(image)
                count += 1
                if count == n:
                    break

    def select_xai(self, xai):
        self.xai = xai

    def select_image(self, kl='3', idx=3):
        # self.image = self.images[kl][idx]
        self.image = self.images[kl][idx]

    def select_baseline(self, kl='1', idx=3):
        self.baseline_image = self.images[kl][idx]

    def select_projection(self):
        pass

    def run_xai(self, **kwargs):
        prediction_logits = self.model(self.image)[0]
        softmax = nn.Softmax(dim=0)
        prediction_probas = softmax(prediction_logits)
        explain_label = int(np.argmax(prediction_probas))
        if self.xai == DeepLift:
            attr_model = self.xai(self.model, multiply_by_inputs=True)
        if self.xai == GuidedGradCam:
            attr_model = self.xai(self.model, layer=self.model.features.denseblock4.denselayer32.conv2)
        attr = attr_model.attribute(self.image, target=explain_label, **kwargs).detach().numpy()
        return np.rollaxis(attr, 1, 4)

    def run_projection(self):
        image = self.image.detach().numpy()
        image = np.rollaxis(image, 1, 4)
        self.projection = normalize_minmax(normalize_minmax(self.attr) + normalize_minmax(image))

    def set_layout(self):
        fig = px.imshow(self.projection[0])
        self.layout = html.Div(children=[
            html.H1(children='A   I   R   M   E   X'),

            html.Div(children='''
                AIRMEX: AI for Radiology Must be EXplainable
            '''),

            dcc.Graph(
                id='Explanation',
                figure=fig
            )
        ])

        fig.update_layout(dragmode="drawrect")
        config = {
            "modeBarButtonsToAdd": [
                "drawline",
                "drawopenpath",
                "drawclosedpath",
                "drawcircle",
                "drawrect",
                "eraseshape",
            ]
        }



if __name__ == '__main__':
    a = Airmex(GuidedGradCam)
    a.init_image_library()
    a.load_images()
    print('done')
    a.run_server()





