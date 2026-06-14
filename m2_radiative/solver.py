import numpy as np
import trimesh
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def voxelise(mesh, pitch=0.08):
    vox = trimesh.voxel.creation.voxelize(mesh, pitch=pitch)
    # fill the interior — voxelize only gives the shell
    filled = vox.fill()
    grid = filled.matrix.astype(float)
    return grid, pitch

def solve_fluence(mesh, pitch=0.08):
    mu_a   = 0.01
    D_val  = 1.0 / (3.0 * (mu_a + 1.0))

    grid, dx = voxelise(mesh, pitch=pitch)
    nx, ny, nz = grid.shape
    N = nx * ny * nz
    print(f"Voxel grid: {nx} x {ny} x {nz} = {N} voxels")

    def idx(i, j, k):
        return i*(ny*nz) + j*nz + k

    # Gaussian source at tissue centre
    cx, cy, cz = nx//2, ny//2, nz//2
    xi = np.arange(nx)[:,None,None]
    yi = np.arange(ny)[None,:,None]
    zi = np.arange(nz)[None,None,:]
    S = np.exp(-((xi-cx)**2+(yi-cy)**2+(zi-cz)**2)/8.0) * grid

    print("Building matrix...")
    A = lil_matrix((N, N))
    b = np.zeros(N)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n = idx(i,j,k)
                if grid[i,j,k] < 0.5:
                    A[n,n] = 1.0
                    continue
                # diffusion + absorption
                A[n,n] += mu_a
                for di,dj,dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    ni,nj,nk = i+di,j+dj,k+dk
                    if 0<=ni<nx and 0<=nj<ny and 0<=nk<nz:
                        A[n,n]         += D_val/dx**2
                        A[n,idx(ni,nj,nk)] -= D_val/dx**2 * grid[ni,nj,nk]
                    else:
                        A[n,n] += D_val/dx**2
                b[n] = S[i,j,k]

    print("Solving...")
    phi = spsolve(A.tocsr(), b)
    phi = np.clip(phi.reshape(grid.shape), 0, None) * grid
    return phi, grid, dx

if __name__ == '__main__':
    from m1_geometry.mesh import make_ellipsoid
    from m2_radiative.fluence import plot_fluence, fluence_stats
    mesh = make_ellipsoid(subdivisions=3, a=1.0, b=0.8, c=0.9)
    phi, grid, dx = solve_fluence(mesh, pitch=0.08)
    fluence_stats(phi, grid)
    plot_fluence(phi, grid)