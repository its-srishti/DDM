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

def ddm_log_likelihood(data, v, a, z, t0, K=7):
    """
    Compute log-likelihood of data under DDM parameters.
    Uses the series solution (Feller 1968, Bogacz et al. 2006).
    
    Parameters:
        data : array of shape (n_trials, 2), columns = [rt, choice]
        v, a, z, t0 : DDM parameters
        K : number of series terms (7 is sufficient for convergence)
    
    Returns:
        scalar log-likelihood (negative infinity if parameters invalid)
    """
    # Parameter bounds — return impossible value if violated
    if a <= 0 or z <= 0 or z >= 1 or t0 <= 0:
        return -1e10
    
    # Check t0 is less than all observed RTs
    if t0 >= np.min(data[:, 0]):
        return -1e10
    
    log_lik = 0.0
    k_vals = np.arange(1, K + 1)  # k = 1, 2, 3, ..., K
    
    for rt, choice in data:
        # Decision time = RT minus non-decision time
        t = rt - t0
        
        if t <= 0:
            log_lik += -1e10
            continue
        
        # Series terms — shared between upper and lower
        # exp(−k²π²t / 2a²) decays fast, so series converges quickly
        decay = np.exp(-k_vals**2 * np.pi**2 * t / (2 * a**2))
        
        if choice == 1:
            # Upper boundary response
            # Change of measure term: exp(v*a*z - v²t/2)
            change_of_measure = np.exp(v * a * z - 0.5 * v**2 * t)
            
            # Eigenfunction terms: k * sin(k*pi*z)
            eigen = k_vals * np.sin(k_vals * np.pi * z)
            
            # Full series
            series = np.sum(eigen * decay)
            density = (np.pi / a**2) * change_of_measure * series
        
        else:
            # Lower boundary response
            # Mirror image: replace z with (1-z)
            change_of_measure = np.exp(-v * a * (1 - z) - 0.5 * v**2 * t)
            eigen = k_vals * np.sin(k_vals * np.pi * (1 - z))
            series = np.sum(eigen * decay)
            density = (np.pi / a**2) * change_of_measure * series
        
        # Guard against numerical issues
        if density <= 0 or not np.isfinite(density):
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
            # First start: sensible default
            x0 = [0.5, 1.0, 0.5, min_rt * 0.5]
        else:
            # Subsequent starts: random
            x0 = [
                np.random.uniform(-1.5, 1.5),   # v
                np.random.uniform(0.5, 2.0),     # a
                np.random.uniform(0.2, 0.8),     # z
                np.random.uniform(0.05, min_rt * 0.8)  # t0
            ]
        
        result = minimize(
            neg_log_lik,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-4,
                'fatol': 1e-4
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
    
