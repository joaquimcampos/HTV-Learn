import numpy as np
import cv2
import copy
from scipy.spatial import Delaunay
import plotly.figure_factory as ff
import matplotlib.cm as cm
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

    vert = np.array(V)
    text = np.array(T)
    simplices_idx = np.array(S)
    simplices_text_idx = np.array(ST)

    # Using cv2.imread() method
    img = cv2.normalize(cv2.imread(img_file), None, alpha=0, beta=1,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # fix axis to be facing z
    vert = np.concatenate((vert[:, 0:1], vert[:, 2:3], -vert[:, 1:2]), axis=1)

    # sort vertices by highest z values and textures accordingly
    simplices_idx_flat = simplices_idx.flatten()
    simplices_text_idx_flat = simplices_text_idx.flatten()

    _, unique_idx = np.unique(simplices_idx_flat, return_index=True)
    vert = vert[simplices_idx_flat[unique_idx]]
    vert_text = text[simplices_text_idx_flat[unique_idx]]

    vert_text[:, 0] = vert_text[:, 0] * img.shape[0]
    vert_text[:, 1] = (1.-vert_text[:, 1]) * img.shape[1]
    vert_text = vert_text.astype(np.int)

    vert_colors = img[vert_text[:, 1], vert_text[:, 0]]
    # switch r,g,b
    vert_colors = np.concatenate((vert_colors[:, 2:3], vert_colors[:, 1:2],
                                    vert_colors[:, 0:1]), axis=1)
    vert = np.concatenate((vert, vert_colors), axis=1)

    # sort vertices by z coordinates in descending direction
    sort_vert = vert[vert[:, 2].argsort()][::-1]

    # get unique_idx of first occurences (largest height)
    _, unique_dx = np.unique(sort_vert[:, 0:2], return_index=True, axis=0)
    unique_sort_vert = sort_vert[unique_dx]

    # eliminate vertices whose height is below cutoff
    eps = 0.55
    min_height, max_height = unique_sort_vert[:, 2].min(), unique_sort_vert[:, 2].max()
    cutoff_val = min_height + (max_height - min_height) * eps
    cutoff_mask = np.where(unique_sort_vert[:, 2] > cutoff_val)[0]
    cleaned_vert = unique_sort_vert[cutoff_mask]

    vert_colors = cleaned_vert[:, 3:6]

    return cleaned_vert[:, 0:3], vert_colors


vert, vert_colors = read_face()
x, y = vert[:, 0], vert[:, 1]
z = vert[:, 2]

points2D = np.vstack([x,y]).T
tri = Delaunay(points2D)
simplices = tri.simplices

color_list = np.mean(vert_colors[simplices], axis=1) # average across face triangles
rgb_colors = BasePlot.map_frac_float_to_rgb(color_list)

fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, opacity=1, facecolor=rgb_colors,
                            i=simplices[:, 0], j=simplices[:, 1], k=simplices[:, 2])])

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
                lightposition=dict(x=x.mean(), y=y.mean(), z=4),
                selector=dict(type='mesh3d'))

# up view
fig_dict['scene']['camera']['up'] = dict(x=0, y=1, z=0)
fig_dict['scene']['camera']['eye'] = dict(x=0, y=0, z=4)
# fig_dict['scene']['zaxis']['visible'] = False

fig.update_layout(**fig_dict)
BasePlot.export_fig(fig, f'face', 'tmp/')
