import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_fluence(phi, grid, save_path='data/m2_fluence_map.png'):
    # find the voxel with maximum fluence and slice through it
    max_idx = np.unravel_index(np.argmax(phi), phi.shape)
    ix, iy_max, iz = max_idx
    iy = phi.shape[1] // 2  # slice through geometric centre in y
    print(f"Max fluence at voxel ({ix}, {iy}, {iz}), value={phi[ix,iy,iz]:.6f}")

    slice_xy  = phi[:, :, iz]
    slice_xz  = phi[:, iy, :]
    tissue_xy = grid[:, :, iz]
    tissue_xz = grid[:, iy, :]

    fig = plt.figure(figsize=(14, 5), facecolor='#0d0d0d')
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    for ax_idx, (sl, tis, title) in enumerate([
        (slice_xy, tissue_xy, f'XY slice at z={iz}'),
        (slice_xz, tissue_xz, f'XZ slice at y={iy} (centre)')
    ]):
        ax = fig.add_subplot(gs[ax_idx], facecolor='#0d0d0d')
        masked = np.where(tis > 0, sl, np.nan)
        im = ax.imshow(masked.T, origin='lower', cmap='inferno',
                       interpolation='bilinear')
        ax.contour(tis.T, levels=[0.5], colors='white',
                   linewidths=0.5, alpha=0.4)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Photon fluence Φ', color='white', fontsize=9)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        ax.set_title(f'Module 2 — {title}', color='white', pad=10)
        ax.set_xlabel('x (voxels)', color='white')
        ax.set_ylabel('z (voxels)', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    plt.suptitle('Photon fluence in tissue phantom — diffusion approximation',
                 color='white', y=1.02, fontsize=11)
    plt.savefig(save_path, dpi=150, facecolor='#0d0d0d', bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")

def fluence_stats(phi, grid):
    interior = phi[grid > 0]
    print(f"Fluence inside tissue:")
    print(f"  min:  {interior.min():.6f}")
    print(f"  max:  {interior.max():.6f}")
    print(f"  mean: {interior.mean():.6f}")
    print(f"  Attenuation ratio (max/mean): {interior.max()/interior.mean():.2f}")