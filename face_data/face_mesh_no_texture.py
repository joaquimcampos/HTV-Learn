import numpy as np
import copy
import plotly.graph_objects as go
from plots.base_plot import BasePlot


def read_face():

    obj_file = 'face_data/obj_free_male_head.obj'

    V = [] # vertices
    S = [] # simplices
    with open(obj_file, "r") as file1:
        for line in file1.readlines():
            f_list = [i.strip() for i in line.split(" ")]
            if len(f_list) == 0:
                continue
            if f_list[0] == 'v':
                assert len(f_list) == 4
                V += [[float(i) for i in f_list[1::]]]
            elif f_list[0] == 'f':
                S += [[int(i)-1 for i in f_list[1::]]]

    vert = np.array(V) # (nvert, 3)
    simplices_idx = np.array(S)

    return vert, simplices_idx


vert, simplices_idx = read_face()
x, y = vert[:, 0], vert[:, 1]
z = vert[:, 2]

# zero facecolors
rgb_colors = BasePlot.get_zero_facecolor(simplices_idx.shape[0])

fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, opacity=1, facecolor=rgb_colors,
                            i=simplices_idx[:, 0], j=simplices_idx[:, 1], k=simplices_idx[:, 2])])

ax_dict = dict(linecolor='#000000', linewidth=4, showgrid=True,
            showticklabels=False, gridcolor='#000000', gridwidth=0.3,
            title=dict(font=dict(size=20))
)
fig_dict = dict(
    scene_aspectmode='data',
    scene = dict(
        xaxis=copy.deepcopy(ax_dict),
        yaxis=copy.deepcopy(ax_dict),
        zaxis=copy.deepcopy(ax_dict),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0)
        )
    ),
    font=dict(size=16),
)

fig.update_traces(lighting=dict(ambient=0.1, diffuse=1, specular=0.1),
                lightposition=dict(x=x.mean(), y=y.mean(), z=4),
                selector=dict(type='mesh3d'))

# up view
fig_dict['scene']['camera']['up'] = dict(x=0, y=1, z=0)
fig_dict['scene']['camera']['eye'] = dict(x=0, y=0, z=4)
# fig_dict['scene']['zaxis']['visible'] = False

fig.update_layout(**fig_dict)
BasePlot.export_fig(fig, f'face', 'tmp/')
