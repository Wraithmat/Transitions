import numpy as np

def dist_on_sphere(x,y):
    """Computes the great-circle distance between points on a sphere.
    Notice: points should be NORMALIZED to lie on the sphere BEFOREHAND.

    Args:
        x: array of shape (..., D) representing points on the D-dimensional sphere.
        y: array of shape (D,) representing a point on the D-dimensional sphere.
    Returns:
        The great-circle distance between x and y, of shape (...,).
    """
    inner_prod = np.clip(np.sum(x*y, axis=-1), -1.0, 1.0)
    return np.arccos(inner_prod)

def frechet_mean_sphere(points, num_steps=1000, step_size=0.1, initial_mean=None):
    """Computes the Fréchet mean on the sphere using gradient descent.
    Notice: the function is not thought to work with ANGLES directly, points should be converted to Cartesian coordinates beforehand (e.g. if you have 1 angle, you should transform it into vectors in R^2).

    Args:
        points: array of shape (N, D) representing N points on the D-dimensional sphere.
        num_steps: number of gradient descent steps.
        step_size: step size for gradient descent, if an array is passed, it is used as step size schedule.
        rng: random number generator (not used here but kept for consistency).

    Returns:
        The Fréchet mean point on the sphere of shape (D,).
    """
    N, D = points.shape
    assert D > 1, "Points must be on a sphere in an embedding dimension at least 2."

    points = points / np.linalg.norm(points, axis=1, keepdims=True)

    # Initialize with the mean of the points
    if initial_mean is None:
        mean_estimate = np.mean(points, axis=0)
        mean_estimate = mean_estimate / np.linalg.norm(mean_estimate)
    else:
        mean_estimate = initial_mean / np.linalg.norm(initial_mean)
    
    if isinstance(step_size, float) or isinstance(step_size, int):
        step_size = np.full(num_steps, step_size)

    for step in range(num_steps):
        dists = dist_on_sphere(points, mean_estimate)

        # Compute gradient of the mean squared distance
        mask = dists > 1e-10
        proj_diffs = points - np.dot(mean_estimate, points.T)[:, None] * mean_estimate[None, :]
        coeffs = -2 * dists[mask] / np.sin(dists[mask])
        
        grad = np.sum(proj_diffs[mask] * coeffs[:, None], axis=0)

        # Gradient descent step
        mean_estimate -= step_size[step] * grad
        # Project back onto the sphere
        mean_estimate /= np.linalg.norm(mean_estimate)

    dists = dist_on_sphere(points, mean_estimate)
    variance = np.mean(dists**2)

    return mean_estimate, variance

def logarithmic_map_sphere(base_point, points):
    """Computes the logarithmic map on the sphere.

    Args:
        base_point: array of shape (D,) representing the base point on the sphere.
        points: array of shape (N,D) representing the points to map.

    Returns:
        The tangent vector at base_point pointing towards point, of shape (D,).
    """
    base_point = base_point / np.linalg.norm(base_point)
    points = points / np.linalg.norm(points, axis=-1, keepdims=True)

    inner_prod = np.clip(np.sum(base_point*points, axis=-1), -1.0, 1.0)
    theta = np.arccos(inner_prod)

    tangent_vector = np.zeros_like(base_point)

    for i in range(points.shape[0]):
        if theta[i] > 1e-10:
            tangent_vector += (points[i] - inner_prod[i] * base_point) * (theta[i] / np.sin(theta[i]))
    return tangent_vector

























"""import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm.auto import tqdm

def dist_on_sphere(x,y):
    inner_prod = jnp.clip(jnp.sum(x*y/jnp.linalg.norm(y, keepdims=True), axis=-1), -1.0, 1.0)
    return jnp.arccos(inner_prod)

def frechet_mean_sphere(points, num_steps=1000, step_size=0.1, rng=None):
    "Computes the Fréchet mean on the sphere using gradient descent.

    Args:
        points: array of shape (N, D) representing N points on the D-dimensional sphere.
        num_steps: number of gradient descent steps.
        step_size: step size for gradient descent.
        rng: JAX random key (not used here but kept for consistency).

    Returns:
        The Fréchet mean point on the sphere of shape (D,).
    "
    N, D = points.shape
    assert D > 1, "Points must be on a sphere of dimension at least 2."
    
    points = points / jnp.linalg.norm(points, axis=1, keepdims=True)

    # Initialize the mean estimate randomly on the sphere
    if rng is None:
        rng = jax.random.PRNGKey(0)
    mean_estimate = jnp.mean(points, axis=0)
    mean_estimate = mean_estimate / jnp.linalg.norm(mean_estimate)

    def loss_fn(mean):
        dists = dist_on_sphere(points, mean)
        return jnp.mean(dists**2)
    
    params_training = jnp.empty(shape=(num_steps + 1, D))
    var_training = jnp.empty(shape=(num_steps ,))

    params_training = params_training.at[0].set(mean_estimate)

    optimizer = optax.adam(step_size)
    state = train_state.TrainState.create(apply_fn=None, params=mean_estimate, tx=optimizer)

    for i in tqdm(range(num_steps), desc="Computing Fréchet mean"):
        variance, grads = jax.value_and_grad(loss_fn)(state.params)

        if jnp.isnan(variance) or jnp.isinf(grads).any() or jnp.isnan(grads).any():
            return params_training[:i+1], var_training[:i+1]
            

        state = state.apply_gradients(grads=grads)
        # Project back onto the sphere
        state = state.replace(params=state.params / jnp.linalg.norm(state.params))

        params_training = params_training.at[i + 1].set(state.params)
        var_training = var_training.at[i].set(variance)

    return params_training, var_training

def logarithmic_map_sphere(base_point, points):
    "Computes the logarithmic map on the sphere.

    Args:
        base_point: array of shape (D,) representing the base point on the sphere.
        points: array of shape (N,D) representing the points to map.

    Returns:
        The tangent vector at base_point pointing towards point, of shape (D,).
    "
    base_point = base_point / jnp.linalg.norm(base_point)
    points = points / jnp.linalg.norm(points, axis=-1, keepdims=True)

    inner_prod = jnp.clip(jnp.sum(base_point*points, axis=-1), -1.0, 1.0)
    theta = jnp.arccos(inner_prod)

    tangent_vector = jnp.zeros_like(base_point)

    mask = theta > 1e-10
    tangent_vector = jnp.where(mask[:, None], (points - inner_prod[:, None] * base_point) * (theta[:, None] / jnp.sin(theta[:, None])), jnp.zeros_like(base_point))
    return tangent_vector
"""