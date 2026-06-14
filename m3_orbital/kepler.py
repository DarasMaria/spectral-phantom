import numpy as np

def rk4_step(f, t, y, dt, **kwargs):
    k1 = np.array(f(t,       y,           **kwargs))
    k2 = np.array(f(t+dt/2,  y+dt/2*k1,  **kwargs))
    k3 = np.array(f(t+dt/2,  y+dt/2*k2,  **kwargs))
    k4 = np.array(f(t+dt,    y+dt*k3,    **kwargs))
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def kepler_rhs(t, state, mu=1.0):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    return [vx, vy, ax, ay]

def propagate_orbit(a=1.0, e=0.3, mu=1.0, n_steps=1000):
    """
    Propagate a Keplerian orbit for one full period.
    a = semi-major axis, e = eccentricity, mu = GM
    Returns arrays: t, x, y, vx, vy
    """
    # orbital period
    T = 2 * np.pi * np.sqrt(a**3 / mu)
    dt = T / n_steps

    # initial conditions: periapsis on +x axis
    r0  = a * (1 - e)
    v0  = np.sqrt(mu * (1+e) / (a*(1-e)))
    state = np.array([r0, 0.0, 0.0, v0])

    times  = [0.0]
    states = [state.copy()]

    for i in range(n_steps):
        state = rk4_step(kepler_rhs, times[-1], state, dt, mu=mu)
        times.append(times[-1] + dt)
        states.append(state.copy())

    states = np.array(states)
    return np.array(times), states[:,0], states[:,1], states[:,2], states[:,3]

def energy(x, y, vx, vy, mu=1.0):
    r = np.sqrt(x**2 + y**2)
    return 0.5*(vx**2+vy**2) - mu/r