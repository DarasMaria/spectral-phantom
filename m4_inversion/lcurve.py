import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_all(residuals, norms, lambdas, corner, lam_opt,
            alphas_obs, flux_obs, alphas_dense, F_rec,
            save_path='data/m4_inversion.png'):

    fig = plt.figure(figsize=(16, 10), facecolor='#0d0d0d')
    gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.4)

    # --- L-curve ---
    ax1 = fig.add_subplot(gs[0, 0], facecolor='#0d0d0d')
    ax1.loglog(residuals, norms, color='#7b68ee', linewidth=1.5)
    ax1.scatter(residuals[corner], norms[corner],
                color='#ff6b6b', s=100, zorder=5, label=f'λ={lam_opt:.1e}')
    ax1.set_title('L-curve — regularisation parameter selection',
                color='white', pad=8)
    ax1.set_xlabel('Residual ||Kα − F_obs||', color='white')
    ax1.set_ylabel('Solution norm ||α||',      color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=9)
    for sp in ax1.spines.values(): sp.set_edgecolor('#444')

    # --- Legendre coefficients ---
    ax2 = fig.add_subplot(gs[0, 1], facecolor='#0d0d0d')
    from numpy.polynomial.legendre import legval
    x_dense = np.cos(alphas_dense)
    deg = 8
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, deg+1))
    # recompute a_hat for plotting individual terms
    from m4_inversion.invert import build_polynomial_operator, tikhonov_solve
    K_plot = build_polynomial_operator(alphas_obs, deg=deg)
    flux_scale = flux_obs.max()
    a_hat = tikhonov_solve(K_plot, flux_obs/flux_scale, lam_opt)
    for n in range(deg+1):
        coeffs = np.zeros(deg+1); coeffs[n] = 1.0
        contrib = a_hat[n] * legval(x_dense, coeffs) * flux_scale
        ax2.plot(np.degrees(alphas_dense), contrib,
                color=colors[n], linewidth=1, alpha=0.8, label=f'P_{n}')
    ax2.set_title('Legendre basis contributions', color='white', pad=8)
    ax2.set_xlabel('Phase angle α (°)', color='white')
    ax2.set_ylabel('Flux contribution', color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=7,
            ncol=3, loc='upper right')
    for sp in ax2.spines.values(): sp.set_edgecolor('#444')

    # --- fit quality ---
    ax3 = fig.add_subplot(gs[1, :], facecolor='#0d0d0d')
    ax3.scatter(np.degrees(alphas_obs), flux_obs,
                color='white', s=20, alpha=0.8, label='Observed flux', zorder=2)
    ax3.plot(np.degrees(alphas_dense), F_rec,
            color='#4fc3f7', linewidth=2.5,
            label='Tikhonov-regularised Legendre fit', zorder=3)
    ax3.set_title('Module 4 — Tikhonov inversion: observed vs recovered phase curve\n'
                '(same ill-posed inverse problem as diffuse optical tomography)',
                color='white', pad=8)
    ax3.set_xlabel('Phase angle α (degrees)', color='white')
    ax3.set_ylabel('Reflected flux F(α)',      color='white')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=9)
    for sp in ax3.spines.values(): sp.set_edgecolor('#444')

    plt.suptitle('Geometric-Spectral Light Transport — Module 4',
                color='white', fontsize=12, y=1.01)
    plt.savefig(save_path, dpi=150, facecolor='#0d0d0d', bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")