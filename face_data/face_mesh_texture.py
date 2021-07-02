import numpy as np
import cv2
import copy
import plotly.graph_objects as go
from plots.base_plot import BasePlot


def read_face():

    obj_file = 'face_data/face_obj/Pasha_guard_head.obj'
    img_file = 'face_data/face_obj/Pasha_guard_head_0.png'

    V = [] # vertices
    T = [] # texture locations
    S = [] # simplices
    ST = [] # simplices texture idx
    with open(obj_file, "r") as file1:
        for line in file1.readlines():
            f_list = [i.strip() for i in line.split(" ")]
            if len(f_list) == 0:
                continue
            if f_list[0] == 'v':
                assert len(f_list) == 4
                V += [[float(i) for i in f_list[1::]]]
            elif f_list[0] == 'vt':
                assert len(f_list) == 3
                T += [[float(i) for i in f_list[1::]]]
            elif f_list[0] == 'f':
                S += [[int(i.split('/')[0])-1 for i in f_list[1:]]]
                ST += [[int(i.split('/')[1])-1 for i in f_list[1:]]]

    vert = np.array(V) # (nvert, 3)
    text = np.array(T) # (ntext, 2)
    simplices_idx = np.array(S) # (nface, 3)
    simplices_text_idx = np.array(ST) # (nface, 3)

    # Using cv2.imread() method
    img = cv2.normalize(cv2.imread(img_file), None, alpha=0, beta=1,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    face_text = text[simplices_text_idx] # (nface, 3, 2)
    face_text[:, :, 0] = (face_text[:, :, 0]) * img.shape[0]
    face_text[:, :, 1] = (1-face_text[:, :, 1]) * img.shape[1]
    face_text = np.round(face_text).astype(np.int)

    face_colors = img[face_text[:, :, 1], face_text[:, :, 0]] # (nface, 3, 3)
    # switch r,g,b
    face_colors = np.concatenate((face_colors[:, :, 2:3], face_colors[:, :, 1:2],
                                    face_colors[:, :, 0:1]), axis=2)


    return vert, simplices_idx, face_colors


vert, simplices_idx, face_colors = read_face()
x, y = vert[:, 0], vert[:, 1]
z = vert[:, 2]

color_list = np.mean(face_colors, axis=1) # average across face triangles
rgb_colors = BasePlot.map_frac_float_to_rgb(color_list)

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

fig.update_traces(lighting=dict(ambient=0.4, diffuse=1, specular=0.1),
                lightposition=dict(x=x.mean(), y=4, z=z.mean()),
                selector=dict(type='mesh3d'))

# side view
fig_dict['scene']['camera']['eye'] = dict(x=0, y=-2.5, z=0)

fig.update_layout(**fig_dict)
BasePlot.export_fig(fig, f'face', 'tmp/')
