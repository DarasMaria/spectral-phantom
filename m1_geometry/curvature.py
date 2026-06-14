import numpy as np
from scipy.sparse import lil_matrix, diags
from m1_geometry.mesh import make_ellipsoid, vertex_areas
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def laplace_beltrami(mesh):
    V = mesh.vertices
    F = mesh.faces
    N = len(V)
    L = lil_matrix((N, N))

    for face in F:
        i, j, k = face
        for (a, b, c) in [(i,j,k), (j,k,i), (k,i,j)]:
            v1 = V[a] - V[c]
            v2 = V[b] - V[c]
            cos_a = np.dot(v1, v2)
            sin_a = np.linalg.norm(np.cross(v1, v2))
            cot_a = cos_a / (sin_a + 1e-12)
            L[a, b] += cot_a / 2
            L[b, a] += cot_a / 2

    row_sums = np.array(L.sum(axis=1)).flatten()
    L = L - diags(row_sums)
    return L.tocsr()

def compute_mean_curvature(mesh, L):
    V = mesh.vertices
    A = vertex_areas(mesh)
    LV = L.dot(V)
    H = 0.5 * np.linalg.norm(LV, axis=1) / (A + 1e-12)
    return H

def plot_curvature(mesh, H, save_path='data/m1_curvature_map.png'):
    V = mesh.vertices
    F = mesh.faces
    face_H = H[F].mean(axis=1)

    norm = Normalize(vmin=np.percentile(face_H, 2),
                     vmax=np.percentile(face_H, 98))
    cmap = plt.cm.plasma
    face_colors = cmap(norm(face_H))

    fig = plt.figure(figsize=(10, 7), facecolor='#0d0d0d')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0d0d0d')

    triangles = V[F]
    poly = Poly3DCollection(triangles, antialiased=False, shade=False)
    poly.set_facecolor(face_colors)
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    ax.set_xlim(V[:,0].min(), V[:,0].max())
    ax.set_ylim(V[:,1].min(), V[:,1].max())
    ax.set_zlim(V[:,2].min(), V[:,2].max())

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Mean curvature H', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ax.set_title('Module 1 — Mean curvature on tissue phantom ellipsoid',
                 color='white', pad=20)
    ax.set_axis_off()
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0d0d0d')
    plt.show()
    print(f"Saved to {save_path}")

if __name__ == '__main__':
    mesh = make_ellipsoid()
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    L = laplace_beltrami(mesh)
    H = compute_mean_curvature(mesh, L)
    print(f"Mean curvature — min: {H.min():.3f}, max: {H.max():.3f}, mean: {H.mean():.3f}")
    plot_curvature(mesh, H)