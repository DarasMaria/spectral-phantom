import numpy as np
import trimesh

def make_ellipsoid(subdivisions=4, a=1.0, b=0.7, c=0.5):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh.vertices[:, 0] *= a
    mesh.vertices[:, 1] *= b
    mesh.vertices[:, 2] *= c
    return mesh

def vertex_areas(mesh):
    V = mesh.vertices
    F = mesh.faces
    areas = np.zeros(len(V))
    for face in F:
        i, j, k = face
        v0 = V[j] - V[i]
        v1 = V[k] - V[i]
        face_area = 0.5 * np.linalg.norm(np.cross(v0, v1))
        areas[i] += face_area / 3
        areas[j] += face_area / 3
        areas[k] += face_area / 3
    return areas