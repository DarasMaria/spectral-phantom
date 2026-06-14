import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from m3_orbital.kepler import propagate_orbit, energy

def phase_angle(x, y):
    """
    Phase angle alpha: angle between star-planet vector
    and planet-observer vector.
    Observer is fixed at (0, -inf) i.e. along -y axis.
    Star is at origin.
    """
    # star -> planet vector
    r_vec = np.array([x, y])
    r     = np.linalg.norm(r_vec)

    # planet -> observer direction (observer far along -y)
    obs   = np.array([0.0, -1.0])

    # planet -> star direction
    to_star = -r_vec / r

    cos_alpha = np.dot(to_star, obs)
    alpha = np.arccos(np.clip(cos_alpha, -1, 1))
    return alpha

def lambertian_flux(alpha, R=0.1, r=1.0, A_g=0.3):
    """
    Disk-integrated flux from a Lambertian sphere.
    F(alpha) = A_g * (R/r)^2 * (sin(alpha) + (pi-alpha)*cos(alpha)) / pi
    This is the same surface integral as the tissue fluence exit integral.
    """
    return A_g * (R/r)**2 * (np.sin(alpha) + (np.pi - alpha)*np.cos(alpha)) / np.pi

def compute_lightcurve(a=1.5, e=0.3, R=0.1, A_g=0.4, noise_level=0.02, n_steps=1000):
    t, x, y, vx, vy = propagate_orbit(a=a, e=e, n_steps=n_steps)

    # verify energy conservation
    E = energy(x, y, vx, vy)
    print(f"Orbital energy conservation: min={E.min():.6f}, max={E.max():.6f}, "
        f"variation={100*(E.max()-E.min())/abs(E.mean()):.4f}%")

    r      = np.sqrt(x**2 + y**2)
    alphas = np.array([phase_angle(x[i], y[i]) for i in range(len(t))])
    flux   = lambertian_flux(alphas, R=R, r=r, A_g=A_g)

    # add Gaussian noise
    np.random.seed(42)
    noise     = noise_level * flux.max() * np.random.randn(len(flux))
    flux_obs  = flux + noise

    return t, x, y, alphas, flux, flux_obs

def plot_lightcurve(t, x, y, alphas, flux, flux_obs,
                    save_path='data/m3_lightcurve.png'):

    fig = plt.figure(figsize=(16, 10), facecolor='#0d0d0d')
    gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.4)

    # --- panel 1: orbit ---
    ax1 = fig.add_subplot(gs[0, 0], facecolor='#0d0d0d')
    ax1.plot(x, y, color='#7b68ee', linewidth=1.5, label='Orbit')
    ax1.scatter([0], [0], color='#FFD700', s=200, zorder=5, label='Star')
    ax1.scatter([x[0]], [y[0]], color='#4fc3f7', s=60, zorder=5, label='Periapsis')
    # observer direction
    ax1.annotate('Observer', xy=(0, -2.2), xytext=(0, -1.8),
                color='white', fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='white', lw=0.8))
    ax1.set_xlim(-2.5, 2.5); ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.set_title('Keplerian orbit (e=0.3)', color='white', pad=8)
    ax1.set_xlabel('x (AU)', color='white'); ax1.set_ylabel('y (AU)', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=8)
    for sp in ax1.spines.values(): sp.set_edgecolor('#444')

    # --- panel 2: phase angle vs time ---
    ax2 = fig.add_subplot(gs[0, 1], facecolor='#0d0d0d')
    ax2.plot(t, np.degrees(alphas), color='#ff9f7f', linewidth=1.2)
    ax2.set_title('Phase angle over one orbit', color='white', pad=8)
    ax2.set_xlabel('Time (orbital units)', color='white')
    ax2.set_ylabel('Phase angle α (degrees)', color='white')
    ax2.tick_params(colors='white')
    for sp in ax2.spines.values(): sp.set_edgecolor('#444')

    # --- panel 3: true + observed light curve ---
    ax3 = fig.add_subplot(gs[1, :], facecolor='#0d0d0d')
    ax3.plot(t, flux,     color='#4fc3f7', linewidth=1.5,
            label='True flux F(α)', zorder=3)
    ax3.scatter(t[::10], flux_obs[::10], color='white', s=4, alpha=0.6,
                label='Observed (+ noise)', zorder=2)
    ax3.set_title('Module 3 — Reflected light curve from Keplerian orbit\n'
                '(same BRDF surface integral as Module 2 tissue fluence exit)',
                color='white', pad=8)
    ax3.set_xlabel('Time (orbital units)', color='white')
    ax3.set_ylabel('Reflected flux F(α)', color='white')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=9)
    for sp in ax3.spines.values(): sp.set_edgecolor('#444')

    plt.suptitle('Geometric-Spectral Light Transport — Module 3',
                color='white', fontsize=12, y=1.01)
    plt.savefig(save_path, dpi=150, facecolor='#0d0d0d', bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")

if __name__ == '__main__':
    t, x, y, alphas, flux, flux_obs = compute_lightcurve()
    plot_lightcurve(t, x, y, alphas, flux, flux_obs)