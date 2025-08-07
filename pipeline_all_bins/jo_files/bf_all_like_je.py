#Arwa's like
# we will use it for each bin individually 
# not tested yet
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
    datasets: dict[str, BoostFactorData]

def Boost_Factor_Model(R, rs, b0):
    x = R / rs
    fx = np.zeros_like(x)
    fx[x > 1] = np.arctan(np.sqrt(x[x > 1]**2 - 1)) / np.sqrt(x[x > 1]**2 - 1)
    fx[x == 1] = 1
    fx[x < 1] = np.arctanh(np.sqrt(1 - x[x < 1]**2)) / np.sqrt(1 - x[x < 1]**2)
    #fix the warning error
    denominator = x**2 - 1
    denominator[denominator == 0] = 1e-10  # or some small value
    B = 1 + b0 * (1 - fx) / denominator
    B[np.isnan(B)] = (b0 + 3) / 3
    return B

def load_data_from_files(path, l, z):
    config = BoostFactorData(None, None, None, None, None, l, z)
    data_file = f"{path}/full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost.dat"
    cov_file  = f"{path}/full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost_cov.dat"
    
    # load the data
    config.R, config.data_vector, config.sigma_B = np.genfromtxt(data_file, unpack=True)
    config.covariance = np.genfromtxt(cov_file)

    # Apply scale cuts
    # r_max <5 makes the same as R[:8]
    config = scale_cuts(config, r_min=0.1, r_max=5.0)

    # Invert covariance matrix 
    # np.linalg.pinv is more stable than np.linalg.inv
    config.inv_cov = np.linalg.pinv(config.covariance)

    return config

def scale_cuts(config, r_min=0.1, r_max=5.0):
    mask = (config.R >= r_min) & (config.R <= r_max)
    config.R = config.R[mask]
    config.data_vector = config.data_vector[mask]
    config.sigma_B = config.sigma_B[mask]
    config.covariance = config.covariance[np.ix_(mask, mask)]
    return config

def setup(options):
    path = options.get_string("boost_factor_likelihood","data_path")
    l0 = options.get_int("boost_factor_likelihood","richness_start")
    le = options.get_int("boost_factor_likelihood","richness_end")
    z0 = options.get_int("boost_factor_likelihood","redshift_start")
    ze = options.get_int("boost_factor_likelihood","redshift_end")

    lambda_bins = range(l0, le)  # Richness bins from l0 to le
    z_bins = range(z0, ze)        # Redshift bins from z0 to ze
    
    configCollection = BoostFactorCollection(lambda_bins, z_bins, {})
    for l in lambda_bins:
        for z in z_bins:
            configCollection.datasets[f'{l}l_{z}z'] = load_data_from_files(path, l, z)
    return configCollection


def execute(block, config):
    # config is of type BoostFactorCollection

    # Initialize log-likelihood
    log_L = 0

    # Loop over all bins
    for l in config.lbins:
        for z in config.zbins:
            cfg = config.datasets[f'{l}l_{z}z']
            R = cfg.R
            data_vector = cfg.data_vector
            inv_cov = cfg.inv_cov

            # Read parameter values from the block
            logrs = block["Boost_Factor_Model_Values", f"logrs_l{l}_z{z}"]
            logb0 = block["Boost_Factor_Model_Values", f"logb0_l{l}_z{z}"]

            # Convert log-parameters to linear scale
            rs = 10**logrs
            b0 = 10**logb0

            # Compute model prediction at current parameter values
            model_prediction = Boost_Factor_Model(R, rs, b0)

            diff = model_prediction - data_vector
            # Chi-squared using covariance
            chisq = np.dot(diff, np.dot(inv_cov, diff))
            log_L += -0.5 * chisq

    # Store likelihood in datablock
    block["likelihoods", "boost_factor_likelihood_like"] = log_L

    return 0
