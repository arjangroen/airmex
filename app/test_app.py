from utils.xai import explain, project_redgreen, normalize
from utils.model import rebuild_kneenet
from captum.attr import DeepLift
import numpy as np
import dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from utils.load_images import load_images
import torch

n_images_per_label = 10
images, labels = load_images(n_images_per_score=n_images_per_label, return_labels=True)
current_image, current_label = images[[-1]], labels[-1]
#current_image = np.rollaxis(current_image.detach().numpy(), 0, 3)

baseline_images = torch.cat([images[[10]], images[[20]], images[[25]]], axis=0)
#baseline_images = np.rollaxis(baseline_images.detach().numpy(), 0,3 )


attr = explain(current_image, DeepLift, baseline_images)
viz_image = np.rollaxis(current_image.detach().numpy(), 1,4)
projection = normalize(normalize(attr) + normalize(viz_image))

fig = px.imshow(projection[0])
app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
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

app.layout = html.Div(
    [html.H3("Drag and draw annotations"), dcc.Graph(figure=fig, config=config), ]
)
app.run_server(debug=True)
