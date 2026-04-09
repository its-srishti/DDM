import numpy as np
from scipy.optimize import minimize

def simulate_ddm(v, a, z, t0, n_trials, dt=0.001, max_time=5.0):
    """
    Simulate n_trials from the Drift-Diffusion Model.
    
    Parameters:
        v  : drift rate (signal strength, can be negative)
        a  : boundary separation (total distance between boundaries)
        z  : starting point as proportion of a (0 to 1, 0.5 = unbiased)
        t0 : non-decision time in seconds
        n_trials : number of trials to simulate
        dt : time step in seconds (smaller = more accurate, slower)
        max_time : maximum trial duration before timeout
    
    Returns:
        numpy array of shape (n_trials, 2)
        column 0 = reaction time in seconds
        column 1 = choice (1 = upper boundary, 0 = lower boundary)
    """
    results = []
    n_steps = int(max_time / dt)
    
    for _ in range(n_trials):
        # Starting position: z is proportion, so multiply by a
        x = z * a
        
        for step in range(n_steps):
            # Euler-Maruyama step
            # Deterministic part: v * dt
            # Stochastic part: noise drawn from N(0,1) scaled by sqrt(dt)
            dx = v * dt + np.sqrt(dt) * np.random.normal(0, 1)
            x += dx
            
            # Check upper boundary
            if x >= a:
                rt = t0 + (step + 1) * dt
                results.append([rt, 1])
                break
            
            # Check lower boundary
            if x <= 0:
                rt = t0 + (step + 1) * dt
                results.append([rt, 0])
                break
        
        else:
            # Trial timed out without hitting a boundary
            # This should be rare with reasonable parameters
            results.append([max_time, np.random.randint(0, 2)])
    
    return np.array(results)

def ddm_log_likelihood(data, v, a, z, t0, K=20):
    """
    Compute log-likelihood of data under DDM parameters.
    Uses the series solution (Feller 1968, Bogacz et al. 2006).
    
    Parameters:
        data : array of shape (n_trials, 2), columns = [rt, choice]
        v, a, z, t0 : DDM parameters
        K : number of series terms (7 is sufficient for convergence)
    
    Returns:
        scalar log-likelihood (negative infinity if parameters invalid)
        K=20 ensures convergence for fast reaction times.
    """
    # Parameter bounds — return impossible value if violated
    if a <= 0 or z <= 0 or z >= 1 or t0 <= 0:
        return -1e10
    
    # Check t0 against data
    min_rt = np.min(data[:, 0])
    if t0 >= min_rt:
        return -1e10
    
    log_lik = 0.0
    k_vals = np.arange(1, K + 1)
    
    for rt, choice in data:
        t = rt - t0
        if t <= 0:
            log_lik += -1e10
            continue

        # Common decay term
        decay = np.exp(-k_vals**2 * np.pi**2 * t / (2 * a**2))
        
        # Shared drift-time correction component
        drift_correction = -0.5 * v**2 * t
        
        if choice == 1:
            # UPPER BOUNDARY: Exponent uses z
            # arg = (v * a * z) - 0.5 * v**2 * t
            arg = (v * a * z) + drift_correction
            eigen = k_vals * np.sin(k_vals * np.pi * z)
        else:
            # LOWER BOUNDARY: Exponent does NOT use (1-z)
            # This fixes the z-inversion (0.3 -> 0.7) bug.
            # arg = (v * a * 0) - 0.5 * v**2 * t
            arg = drift_correction
            eigen = k_vals * np.sin(k_vals * np.pi * (1 - z))
        
        # Apply Clipping to exponent argument to prevent Overflow
        change_of_measure = np.exp(np.clip(arg, -500, 500))
        
        series = np.sum(eigen * decay)
        density = (np.pi / a**2) * change_of_measure * series
        
        if density <= 1e-10 or not np.isfinite(density):
            log_lik += -1e10
        else:
            log_lik += np.log(density)
            
    return log_lik

def fit_ddm(data, n_starts=5):
    """
    Fit DDM to data via Maximum Likelihood Estimation.
    Uses Nelder-Mead optimisation with multiple starting points.
    
    Parameters:
        data     : array of shape (n_trials, 2)
        n_starts : number of random starting points (more = more reliable)
    
    Returns:
        dict with estimated parameters and fit quality
    """
    min_rt = np.min(data[:, 0])
    
    def neg_log_lik(params):
        v, a, z, t0 = params
        return -ddm_log_likelihood(data, v, a, z, t0)
    
    best_result = None
    best_nll = np.inf
    
    for start in range(n_starts):
        # Random starting point within reasonable bounds
        # Different start each iteration to avoid local optima
         if start == 0:
            x0 = [0.5, 1.0, 0.5, min_rt * 0.8]
         else:
            x0 = [
                np.random.uniform(-2.0, 2.0),
                np.random.uniform(0.5, 2.5),
                np.random.uniform(0.1, 0.9),
                np.random.uniform(0.01, min_rt * 0.9)
            ]
        
         result = minimize(
            neg_log_lik,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 3000,
                'xatol': 1e-3,
                'fatol': 1e-3
            }
        )
        
         if result.fun < best_nll:
            best_nll = result.fun
            best_result = result
    
    v_est, a_est, z_est, t0_est = best_result.x
    
    return {
        'v': v_est,
        'a': a_est,
        'z': z_est,
        't0': t0_est,
        'log_likelihood': -best_nll,
        'converged': best_result.success
    }
    
