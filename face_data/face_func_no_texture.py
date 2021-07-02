import numpy as np
import copy
from scipy.spatial import Delaunay
import plotly.graph_objects as go
from plots.base_plot import BasePlot

def read_face():

    obj_file = 'face_data/obj_free_male_head.obj'

    V = []
    with open(obj_file, "r") as file1:
        for line in file1.readlines():
            f_list = [i for i in line.split(" ") if i.strip()]
            if len(f_list) == 0:
                continue
            if f_list[0] != 'v':
                continue
            V += [float(i) for i in f_list[1::]]

    # vertices
    vert = np.array(V).reshape(-1, 3)
    # sort vertices by z coordinates in descending direction
    sort_vert = vert[vert[:, 2].argsort()][::-1]

    # get unique_idx of first occurences (largest height)
    _, unique_dx = np.unique(sort_vert[:, 0:2], return_index=True, axis=0)
    unique_sort_vert = sort_vert[unique_dx]

    # eliminate vertices whose height is below cutoff
    eps = 0.6
    min_height, max_height = unique_sort_vert[:, 2].min(), unique_sort_vert[:, 2].max()
    cutoff_val = min_height + (max_height - min_height) * eps
    cutoff_mask = np.where(unique_sort_vert[:, 2] > cutoff_val)[0]
    cleaned_vert = unique_sort_vert[cutoff_mask]
    cleaned_vert[:, 2] = cleaned_vert[:, 2] - cutoff_val # shift z.min() to z = 0
    cleaned_vert[:, 0] = cleaned_vert[:, 0] - cleaned_vert[:, 0].mean() # shift x around 0
    cleaned_vert[:, 1] = cleaned_vert[:, 1] - cleaned_vert[:, 1].mean() # shift y around 0

    return cleaned_vert


vert = read_face()
x, y = vert[:, 0], vert[:, 1]
z = vert[:, 2]

points2D = np.vstack([x,y]).T
tri = Delaunay(points2D)
simplices = tri.simplices

# zero facecolors
rgb_colors = BasePlot.get_zero_facecolor(simplices.shape[0])

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

fig.update_traces(lighting=dict(ambient=0.1, diffuse=1, specular=0.1),
                lightposition=dict(x=x.mean(), y=y.mean(), z=4),
                selector=dict(type='mesh3d'))

# up view
fig_dict['scene']['camera']['up'] = dict(x=0, y=1, z=0)
fig_dict['scene']['camera']['eye'] = dict(x=0, y=0, z=4)
# fig_dict['scene']['zaxis']['visible'] = False

fig.update_layout(**fig_dict)
BasePlot.export_fig(fig, f'face', 'tmp/')
