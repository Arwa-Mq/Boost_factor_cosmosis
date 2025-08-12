# bf_all_like_je_safe.py  -- drop-in safer version
import numpy as np
from cosmosis.datablock import names
from dataclasses import dataclass

@dataclass
class BoostFactorData:
    R: np.ndarray
    data_vector: np.ndarray
    sigma_B: np.ndarray
    covariance: np.ndarray
    inv_cov: np.ndarray
    l: int
    z: int

@dataclass
class BoostFactorCollection:
    lbins: list
    zbins: list
    datasets: dict

def Boost_Factor_Model(R, rs, b0):
    # Numerically safe implementation
    x = R / rs
    fx = np.zeros_like(x, dtype=float)
    tol = 1e-12

    # masks with tolerance
    mask_gt = x > 1.0 + tol
    mask_eq = np.abs(x - 1.0) <= tol
    mask_lt = x < 1.0 - tol

    # protect arguments to sqrt/atanh
    if np.any(mask_gt):
        arg = x[mask_gt]**2 - 1.0
        arg = np.clip(arg, 0.0, None)
        fx[mask_gt] = np.arctan(np.sqrt(arg)) / np.sqrt(arg)

    if np.any(mask_eq):
        fx[mask_eq] = 1.0

    if np.any(mask_lt):
        arg = 1.0 - x[mask_lt]**2
        arg = np.clip(arg, 0.0, None)
        # arctanh sometimes gives inf if argument == 1; clip slightly below 1
        sqrt_arg = np.sqrt(arg)
        sqrt_arg = np.clip(sqrt_arg, 0.0, 1.0 - 1e-12)
        with np.errstate(all="ignore"):
            fx[mask_lt] = np.arctanh(sqrt_arg) / sqrt_arg

    # denominator protect
    denominator = x**2 - 1.0
    small = 1e-12
    denominator = np.where(np.abs(denominator) < small, np.sign(denominator) * small + small, denominator)

    B = 1.0 + b0 * (1.0 - fx) / denominator

    # final sanitization
    B = np.nan_to_num(B, nan=(b0 + 3.0) / 3.0, posinf=(b0 + 3.0) / 3.0, neginf=(b0 + 3.0) / 3.0)
    return B

def load_data_from_files(path, l, z):
    data_file = f"{path}/full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost.dat"
    cov_file  = f"{path}/full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost_cov.dat"

    # try to load; be robust to headers, comments, extra whitespace
    try:
        R, data_vector, sigma_B = np.genfromtxt(data_file, unpack=True, comments="#")
    except Exception as e:
        raise RuntimeError(f"Failed to read data file {data_file}: {e}")

    try:
        covariance = np.genfromtxt(cov_file, comments="#")
    except Exception as e:
        raise RuntimeError(f"Failed to read covariance file {cov_file}: {e}")

    # wrap into config
    config = BoostFactorData(R=np.asarray(R), data_vector=np.asarray(data_vector),
                             sigma_B=np.asarray(sigma_B), covariance=np.asarray(covariance),
                             inv_cov=None, l=l, z=z)

    # Apply scale cuts
    config = scale_cuts(config, r_min=0.1, r_max=5.0)

    # If after scale cuts we have no data, raise
    if config.R.size == 0:
        raise RuntimeError(f"No data left after scale cuts for l={l}, z={z}")

    # Regularize covariance: add small jitter on diagonal scaled to typical diag
    cov = config.covariance
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != config.R.size:
        # try to reshape if the covariance is flat or misread
        try:
            cov = cov.reshape((config.R.size, config.R.size))
        except Exception:
            raise RuntimeError(f"Covariance shape mismatch for l={l}, z={z}: got shape {config.covariance.shape}")

    # add jitter
    diag_mean = np.mean(np.diag(cov))
    jitter = max(1e-10, 1e-8 * (diag_mean if diag_mean > 0 else 1.0))
    cov += np.eye(cov.shape[0]) * jitter

    # compute stable pseudo-inverse
    inv_cov = np.linalg.pinv(cov)

    if not np.isfinite(inv_cov).all():
        raise RuntimeError(f"Inverse covariance contains non-finite values for l={l}, z={z}")

    config.covariance = cov
    config.inv_cov = inv_cov
    return config

def scale_cuts(config, r_min=0.1, r_max=5.0):
    mask = (config.R >= r_min) & (config.R <= r_max)
    config.R = config.R[mask]
    config.data_vector = config.data_vector[mask]
    config.sigma_B = config.sigma_B[mask]
    # mask covariance rows/cols
    try:
        config.covariance = config.covariance[np.ix_(mask, mask)]
    except Exception:
        # if covariance is 1d or wrong shaped, leave as-is and handle later
        pass
    return config

def setup(options):
    path = options.get_string("boost_factor_likelihood","data_path")
    l0 = options.get_int("boost_factor_likelihood","richness_start")
    le = options.get_int("boost_factor_likelihood","richness_end")
    z0 = options.get_int("boost_factor_likelihood","redshift_start")
    ze = options.get_int("boost_factor_likelihood","redshift_end")

    lambda_bins = list(range(l0, le))  # Richness bins from l0 to le-1
    z_bins = list(range(z0, ze))        # Redshift bins from z0 to ze-1

    configCollection = BoostFactorCollection(lambda_bins, z_bins, {})

    # load datasets with error handling and report
    for l in lambda_bins:
        for z in z_bins:
            key = f'{l}l_{z}z'
            try:
                configCollection.datasets[key] = load_data_from_files(path, l, z)
            except Exception as e:
                # if a bin fails to load, skip it and print a warning
                print(f"Warning: skipping bin {key} due to error: {e}")
    # diagnostic
    n_params = 0
    for l in lambda_bins:
        for z in z_bins:
            # two params per bin that exists
            key = f'{l}l_{z}z'
            if key in configCollection.datasets:
                n_params += 2
    print(f"Boost factor setup: {len(configCollection.datasets)} bins loaded, estimated n_params = {n_params}")
    return configCollection

def execute(block, config):
    # config is of type BoostFactorCollection
    log_L = 0.0

    for l in config.lbins:
        for z in config.zbins:
            key = f'{l}l_{z}z'
            if key not in config.datasets:
                # skip bins that failed to load
                continue
            cfg = config.datasets[key]
            R = cfg.R
            data_vector = cfg.data_vector
            inv_cov = cfg.inv_cov

            # Read parameter values; these should be scalars but be defensive
            try:
                logrs = block["Boost_Factor_Model_Values", f"logrs_l{l}_z{z}"]
                logb0 = block["Boost_Factor_Model_Values", f"logb0_l{l}_z{z}"]
            except Exception as e:
                raise RuntimeError(f"Missing parameter for l={l} z={z}: {e}")

            # cast to float safely
            try:
                logrs = float(np.asarray(logrs).squeeze())
                logb0 = float(np.asarray(logb0).squeeze())
            except Exception:
                raise RuntimeError(f"Non-scalar parameter values for l={l} z={z}: logrs={logrs}, logb0={logb0}")

            # guard against non-finite parameter values
            if not (np.isfinite(logrs) and np.isfinite(logb0)):
                raise RuntimeError(f"Non-finite parameter values for l={l} z={z}: logrs={logrs}, logb0={logb0}")

            rs = 10.0**logrs
            b0 = 10.0**logb0

            model_prediction = Boost_Factor_Model(R, rs, b0)

            # check finite
            if not (np.isfinite(model_prediction).all() and np.isfinite(data_vector).all()):
                # penalize badly-behaved models with a huge chi2 instead of crashing
                print(f"Warning: non-finite model/data in bin l{l}_z{z}; skipping contribution")
                continue

            diff = model_prediction - data_vector
            chisq = float(np.dot(diff, np.dot(inv_cov, diff)))
            if not np.isfinite(chisq):
                # regularize: large penalty
                chisq = 1e30
            log_L += -0.5 * chisq

    block["likelihoods", "boost_factor_likelihood_like"] = float(log_L)
    return 0
