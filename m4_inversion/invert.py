import numpy as np
from m3_orbital.lightcurve import compute_lightcurve

def build_polynomial_operator(alphas, deg=8):
    """
    Forward operator K using Legendre polynomial basis.
    Each column is P_n(cos(alpha)) for n=0..deg.
    This is the standard basis for planetary phase curve inversion.
    """
    from numpy.polynomial.legendre import legval
    n_obs  = len(alphas)
    n_coeffs = deg + 1
    K = np.zeros((n_obs, n_coeffs))
    x = np.cos(alphas)
    for n in range(n_coeffs):
        coeffs = np.zeros(n_coeffs)
        coeffs[n] = 1.0
        K[:, n] = legval(x, coeffs)
    return K

def tikhonov_solve(K, F_obs, lam):
    A = K.T @ K + lam * np.eye(K.shape[1])
    b = K.T @ F_obs
    return np.linalg.solve(A, b)

def l_curve(K, F_obs, lambdas):
    residuals, norms = [], []
    for lam in lambdas:
        a = tikhonov_solve(K, F_obs, lam)
        residuals.append(np.linalg.norm(K @ a - F_obs))
        norms.append(np.linalg.norm(a))
    return np.array(residuals), np.array(norms)

def recover_phase_curve(a_hat, deg=8):
    from numpy.polynomial.legendre import legval
    alphas_dense = np.linspace(0, np.pi, 500)
    x = np.cos(alphas_dense)
    F_rec = np.zeros(len(alphas_dense))
    for n in range(deg+1):
        coeffs = np.zeros(deg+1)
        coeffs[n] = 1.0
        F_rec += a_hat[n] * legval(x, coeffs)
    return alphas_dense, F_rec

if __name__ == '__main__':
    from m4_inversion.lcurve import plot_all

    print("Computing light curve...")
    t, x, y, alphas, flux, flux_obs = compute_lightcurve(n_steps=500)
    r_vals = np.sqrt(x**2 + y**2)

    idx        = np.linspace(0, len(t)-1, 80, dtype=int)
    alphas_sub = alphas[idx]
    flux_sub   = flux_obs[idx]

    # normalise
    flux_scale = flux_sub.max()
    flux_norm  = flux_sub / flux_scale

    print("Building Legendre forward operator...")
    K = build_polynomial_operator(alphas_sub, deg=8)

    print("Running L-curve...")
    lambdas = np.logspace(-12, 0, 60)
    residuals, norms = l_curve(K, flux_norm, lambdas)

    # find corner: closest point to origin on normalised log-log plot
    log_r = np.log10(residuals + 1e-20)
    log_n = np.log10(norms     + 1e-20)
    lr = (log_r - log_r.min()) / (log_r.max() - log_r.min() + 1e-20)
    ln = (log_n - log_n.min()) / (log_n.max() - log_n.min() + 1e-20)
    corner  = max(0, np.argmin(np.sqrt(lr**2 + ln**2)) - 5)
    lam_opt = lambdas[corner]
    print(f"Optimal lambda: {lam_opt:.2e}")

    a_hat = tikhonov_solve(K, flux_norm, lam_opt)
    alphas_dense, F_rec = recover_phase_curve(a_hat, deg=8)
    F_rec_scaled = F_rec * flux_scale

    print(f"Fit residual: {np.linalg.norm(K @ a_hat - flux_norm):.4f}")
    plot_all(residuals, norms, lambdas, corner, lam_opt,
            alphas_sub, flux_sub, alphas_dense, F_rec_scaled)