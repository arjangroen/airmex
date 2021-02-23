from utils.xai import Airmex
from captum.attr import DeepLift
import numpy as np
import dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html



target = 4
img_num = 3


a = Airmex(attr_model=DeepLift)
attr = a.explain(target=target)[img_num]
img = np.rollaxis(a.images[img_num].detach().numpy(), 0, 3)
projection = a.project_deeplift(attr, img)


app = dash.Dash(__name__)
img = projection
fig = px.imshow(img)
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

app = dash.Dash(__name__)
app.layout = html.Div(
    [html.H3("Drag and draw annotations"), dcc.Graph(figure=fig, config=config), ]
)
app.run_server(debug=True)
