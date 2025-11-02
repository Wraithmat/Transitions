import numpy as np

def sliding_windows(trajectory, window_size):
    """Computes sliding window means and covariances for a 1D trajectory.
    
    Inputs:
        trajectory: 1D numpy array of shape (T,) representing the trajectory.
        window_size: Integer, size of the sliding window.
    """

    assert window_size > 0, "Window size must be positive."
    assert len(trajectory) >= window_size, "Trajectory length must be at least as large as window size."
    assert len(trajectory.shape) == 1, "Trajectory must be a 1D array."

    sliding_means_x = np.convolve(trajectory, np.ones(window_size)/window_size, mode='valid')

    covariances = []
    for i in range(len(trajectory) - window_size + 1):
        window = trajectory[i:i+window_size]
        cov = np.var(window, ddof=1)
        # cov = np.cov(window, ddof=1)  # For 1D data, variance is the covariance
        covariances.append(cov)
    covariances = np.array(covariances)

    return sliding_means_x, covariances

def E_step(obs, mus, sigmas, pis, L):
    """Expectation step of the EM algorithm for a Gaussian mixture model in the mean-variance space.

    Inputs:
        obs: array of shape (N, 2) where each row is (mean, variance) observation.
        mus: array of shape (K,) representing the means of the K Gaussian components.
        sigmas: array of shape (K,) representing the standard deviations of the K Gaussian components.
        pis: array of shape (K,) representing the mixture weights of the K Gaussian components.
        L: Integer, length of the sliding window used to compute the observations.
    """
    covariance = np.zeros((len(mus), 2, 2))
    for k in range(len(mus)):
        covariance[k,0,0]=sigmas[k]**2/L
        covariance[k,1,1]=sigmas[k]**4*2/(L-1)
    mean = np.zeros((len(mus), 2))
    for k in range(len(mus)):
        mean[k,0]=mus[k]
        mean[k,1]=sigmas[k]**2

    probs = np.zeros((obs.shape[0], len(mus)))
    for k in range(len(mus)):
        diff = obs - mean[k]
        inv_cov = np.linalg.inv(covariance[k])
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        denom = np.sqrt((2 * np.pi) ** 2 * (covariance[k,0,0] * covariance[k,1,1]))
        probs[:, k] = pis[k] * np.exp(exponent) / denom
    responsibilities = probs / np.sum(probs, axis=1, keepdims=True)
    return responsibilities

def M_step(obs, responsibilities, L):
    """Maximization step of the EM algorithm for a Gaussian mixture model in the mean-variance space.
    Inputs:
        obs: array of shape (N, 2) where each row is (mean, variance) observation.
        responsibilities: array of shape (N, K) representing the responsibilities from the E-step.
        L: Integer, length of the sliding window used to compute the observations.
    """
    N_k = np.sum(responsibilities, axis=0)
    pis = N_k / obs.shape[0]

    mus = np.zeros(len(N_k))
    sigmas = np.zeros(len(N_k))

    for k in range(len(N_k)):
        weighted_sum_x = np.sum(responsibilities[:, k] * obs[:, 0])
        mus[k] = weighted_sum_x / N_k[k]

        diff = obs[:, 0] - mus[k]
        weighted_var = np.sum(responsibilities[:, k] * diff ** 2) 
        weighted_sigmas = np.sum(responsibilities[:, k] * obs[:, 1]**2)
        weighted_sigma = np.sum(responsibilities[:, k] * obs[:, 1])
        a = -3*N_k[k]
        b = L*weighted_var-(L-1)*weighted_sigma
        c = (L-1)*weighted_sigmas

        discriminant = b**2 - 4*a*c
        sigmas[k] = np.sqrt((-b - np.sqrt(discriminant)) / (2*a))        

    return mus, sigmas, pis