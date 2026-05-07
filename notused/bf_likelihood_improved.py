# Improved Boost Factor Likelihood Module for CosmoSIS
# Supports multi-bin automation, flexible data handling, and better code structure
# Author: Arwa (improved version)

import numpy as np
import os
from pathlib import Path

# CosmoSIS import - only needed when running within CosmoSIS
try:
    from cosmosis.datablock import names
    HAS_COSMOSIS = True
except ImportError:
    HAS_COSMOSIS = False


# =============================================================================
# BOOST FACTOR MODEL
# =============================================================================

def boost_factor_model(R, rs, b0):
    """
    Compute the boost factor B(R) for a given scale radius and amplitude.

    The boost factor accounts for contamination of source galaxies by
    cluster member galaxies in weak lensing analyses.

    Parameters
    ----------
    R : array_like
        Radial distances (in Mpc/h or consistent units with rs)
    rs : float
        Scale radius parameter
    b0 : float
        Amplitude parameter

    Returns
    -------
    B : ndarray
        Boost factor values at each radius
    """
    x = np.atleast_1d(R / rs).astype(float)
    B = np.zeros_like(x, dtype=float)

    # Define tolerance for x ~ 1 (removable singularity)
    tol = 1e-6

    # Masks for different regimes
    mask_near1 = np.abs(x - 1) < tol
    mask_gt1 = (x > 1) & ~mask_near1
    mask_lt1 = (x < 1) & ~mask_near1

    # x > 1: arctan regime
    if np.any(mask_gt1):
        x_gt1 = x[mask_gt1]
        sqrt_term = np.sqrt(x_gt1**2 - 1)
        fx = np.arctan(sqrt_term) / sqrt_term
        B[mask_gt1] = 1 + b0 * (1 - fx) / (x_gt1**2 - 1)

    # x < 1: arctanh regime
    if np.any(mask_lt1):
        x_lt1 = x[mask_lt1]
        sqrt_term = np.sqrt(1 - x_lt1**2)
        fx = np.arctanh(sqrt_term) / sqrt_term
        B[mask_lt1] = 1 + b0 * (1 - fx) / (x_lt1**2 - 1)

    # x ~ 1: use the analytic limit (b0 + 3) / 3
    # This comes from L'Hopital's rule or series expansion around x=1
    if np.any(mask_near1):
        B[mask_near1] = (b0 + 3) / 3

    # Handle any remaining NaN/inf values
    B = np.where(np.isnan(B) | np.isinf(B), (b0 + 3) / 3, B)

    return B


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

class BoostFactorData:
    """
    Container for boost factor data from a single bin.
    """
    def __init__(self, R, data_vector, covariance, richness_bin, redshift_bin):
        self.R = R
        self.data_vector = data_vector
        self.covariance = covariance
        self.inv_cov = np.linalg.inv(covariance)
        self.richness_bin = richness_bin
        self.redshift_bin = redshift_bin
        self.n_points = len(R)

    @property
    def bin_label(self):
        return f"l{self.richness_bin}_z{self.redshift_bin}"


def load_y1_data(data_path, richness_bin, redshift_bin, n_points=8):
    """
    Load DES Year 1 boost factor data.

    Parameters
    ----------
    data_path : str
        Path to the directory containing the data files
    richness_bin : int
        Richness bin index (0-3)
    redshift_bin : int
        Redshift bin index (0-2)
    n_points : int
        Number of radial points to use (default: 8)

    Returns
    -------
    BoostFactorData
        Data container with loaded data
    """
    data_file = os.path.join(
        data_path,
        f"full-unblind-v2-mcal-zmix_y1clust_l{richness_bin}_z{redshift_bin}_zpdf_boost.dat"
    )
    cov_file = os.path.join(
        data_path,
        f"full-unblind-v2-mcal-zmix_y1clust_l{richness_bin}_z{redshift_bin}_zpdf_boost_cov.dat"
    )

    R, data_vector, sigma_B = np.genfromtxt(data_file, unpack=True)
    covariance = np.genfromtxt(cov_file)

    # Truncate to n_points
    R = R[:n_points]
    data_vector = data_vector[:n_points]
    covariance = covariance[:n_points, :n_points]

    return BoostFactorData(R, data_vector, covariance, richness_bin, redshift_bin)


def load_y3_data(data_path, richness_bin, redshift_bin, n_radial_bins=13,
                 use_diagonal_cov=False, regularization=1e-10):
    """
    Load DES Year 3 f_cl posterior data.

    The Y3 data files contain posterior samples for f_cl at each radial bin.
    We compute the mean and covariance from these samples.

    Parameters
    ----------
    data_path : str
        Path to the directory containing the data files
    richness_bin : int
        Richness bin index (0-3)
    redshift_bin : int
        Redshift bin index (0-3)
    n_radial_bins : int
        Number of radial bins to use (default: 13)
    use_diagonal_cov : bool
        If True, use only diagonal elements (independent errors).
        Default False uses full covariance with regularization.
    regularization : float
        Small value added to diagonal for numerical stability.

    Returns
    -------
    BoostFactorData
        Data container with loaded data
    """
    data_file = os.path.join(
        data_path,
        f"fcl_z{redshift_bin}_l{richness_bin}.txt"
    )

    # Load posterior samples (rows = samples, columns = radial bins)
    # Skip header row (contains column names like r0, r1, ...)
    samples = np.loadtxt(data_file, delimiter='\t', skiprows=1)

    # Use specified number of radial bins
    samples = samples[:, :n_radial_bins]

    # Compute mean and covariance from posterior samples
    data_vector = np.mean(samples, axis=0)
    covariance = np.cov(samples, rowvar=False)

    if use_diagonal_cov:
        # Use only diagonal (independent errors)
        covariance = np.diag(np.diag(covariance))
    else:
        # Regularize covariance matrix for numerical stability
        # Add small value to diagonal to ensure positive definiteness
        covariance = covariance + regularization * np.eye(n_radial_bins)

    # Define radial bins (log-spaced, adjust as needed)
    R = np.logspace(-1, 2, n_radial_bins)

    return BoostFactorData(R, data_vector, covariance, richness_bin, redshift_bin)


def discover_bins(data_path, data_type='y3'):
    """
    Automatically discover available bins in the data directory.

    Parameters
    ----------
    data_path : str
        Path to the data directory
    data_type : str
        'y1' or 'y3' data format

    Returns
    -------
    list of tuple
        List of (richness_bin, redshift_bin) tuples
    """
    bins = []
    path = Path(data_path)

    if data_type == 'y3':
        pattern = 'fcl_z*_l*.txt'
        for f in path.glob(pattern):
            # Parse filename like fcl_z0_l1.txt
            name = f.stem  # fcl_z0_l1
            parts = name.split('_')
            z = int(parts[1][1:])  # z0 -> 0
            l = int(parts[2][1:])  # l1 -> 1
            bins.append((l, z))
    elif data_type == 'y1':
        pattern = '*_l*_z*_zpdf_boost.dat'
        for f in path.glob(pattern):
            name = f.stem
            # Parse richness and redshift from filename
            parts = name.split('_')
            for i, p in enumerate(parts):
                if p.startswith('l') and p[1:].isdigit():
                    l = int(p[1:])
                if p.startswith('z') and p[1:].isdigit():
                    z = int(p[1:])
            bins.append((l, z))

    return sorted(bins)


# =============================================================================
# COSMOSIS INTERFACE
# =============================================================================

def setup(options):
    """
    CosmoSIS setup function - loads data and prepares configuration.

    Options (from .ini file):
    - data_path: Path to data directory
    - data_type: 'y1' or 'y3'
    - richness_bins: Comma-separated list or 'all'
    - redshift_bins: Comma-separated list or 'all'
    - n_radial_points: Number of radial points to use
    """
    # Get configuration from options
    section = "boost_factor_likelihood"

    data_path = options.get_string(section, "data_path")
    data_type = options.get_string(section, "data_type", default="y3")
    n_radial_points = options.get_int(section, "n_radial_points", default=13 if data_type == 'y3' else 8)

    # Determine which bins to use
    richness_bins_str = options.get_string(section, "richness_bins", default="all")
    redshift_bins_str = options.get_string(section, "redshift_bins", default="all")

    if richness_bins_str == "all" or redshift_bins_str == "all":
        available_bins = discover_bins(data_path, data_type)
        if richness_bins_str == "all":
            richness_bins = sorted(set(b[0] for b in available_bins))
        else:
            richness_bins = [int(x.strip()) for x in richness_bins_str.split(',')]
        if redshift_bins_str == "all":
            redshift_bins = sorted(set(b[1] for b in available_bins))
        else:
            redshift_bins = [int(x.strip()) for x in redshift_bins_str.split(',')]
    else:
        richness_bins = [int(x.strip()) for x in richness_bins_str.split(',')]
        redshift_bins = [int(x.strip()) for x in redshift_bins_str.split(',')]

    # Load data for all requested bins
    loader = load_y3_data if data_type == 'y3' else load_y1_data

    bin_data = {}
    for l in richness_bins:
        for z in redshift_bins:
            try:
                data = loader(data_path, l, z, n_radial_points)
                bin_data[(l, z)] = data
                print(f"Loaded data for bin l={l}, z={z} ({data.n_points} points)")
            except Exception as e:
                print(f"Warning: Could not load bin l={l}, z={z}: {e}")

    # Check if data is f_cl (Y3) or B (Y1)
    data_is_fcl = (data_type == 'y3')

    config = {
        'bin_data': bin_data,
        'data_type': data_type,
        'data_is_fcl': data_is_fcl,
        'bins': list(bin_data.keys())
    }

    print(f"Successfully loaded {len(bin_data)} bins")
    print(f"Data format: {'f_cl (contamination fraction)' if data_is_fcl else 'B (boost factor)'}")
    return config


def execute(block, config):
    """
    CosmoSIS execute function - computes likelihood for all bins.
    """
    bin_data = config['bin_data']
    data_is_fcl = config.get('data_is_fcl', True)

    total_log_L = 0.0

    for (l, z), data in bin_data.items():
        # Read parameters for this bin from datablock
        # Parameters are named like logrs_l0_z0, logb0_l0_z0
        param_suffix = f"l{l}_z{z}"

        try:
            logrs = block["boost_factor_params", f"logrs_{param_suffix}"]
            logb0 = block["boost_factor_params", f"logb0_{param_suffix}"]
        except:
            # Fallback to old naming convention
            logrs = block["Boost_Factor_Model_Values", f"logrs_{l}{z}"]
            logb0 = block["Boost_Factor_Model_Values", f"logb0_{l}{z}"]

        rs = 10**logrs
        b0 = 10**logb0

        # Compute model prediction (boost factor B)
        B_model = boost_factor_model(data.R, rs, b0)

        # Convert to f_cl if data is in contamination fraction units
        if data_is_fcl:
            model = boost_to_fcl(B_model)
        else:
            model = B_model

        # Compute chi-squared
        diff = model - data.data_vector
        chisq = np.dot(diff, np.dot(data.inv_cov, diff))

        log_L = -0.5 * chisq
        total_log_L += log_L

        # Store individual bin likelihoods for diagnostics
        block["boost_factor_diagnostics", f"chisq_{param_suffix}"] = chisq
        block["boost_factor_diagnostics", f"logL_{param_suffix}"] = log_L

    # Store total likelihood
    block["likelihoods", "boost_factor_likelihood_like"] = total_log_L

    return 0


# =============================================================================
# STANDALONE TESTING UTILITIES
# =============================================================================

def boost_to_fcl(B):
    """
    Convert boost factor B to contamination fraction f_cl.

    The relationship is: B = 1 / (1 - f_cl)
    Therefore: f_cl = (B - 1) / B = 1 - 1/B

    Parameters
    ----------
    B : array_like
        Boost factor values (B >= 1)

    Returns
    -------
    f_cl : ndarray
        Contamination fraction values (0 <= f_cl < 1)
    """
    return 1.0 - 1.0 / np.maximum(B, 1.0)


def fcl_to_boost(f_cl):
    """
    Convert contamination fraction f_cl to boost factor B.

    The relationship is: B = 1 / (1 - f_cl)

    Parameters
    ----------
    f_cl : array_like
        Contamination fraction values (0 <= f_cl < 1)

    Returns
    -------
    B : ndarray
        Boost factor values (B >= 1)
    """
    return 1.0 / (1.0 - np.clip(f_cl, 0, 0.999))


def compute_likelihood_standalone(R, data_vector, covariance, rs, b0, data_is_fcl=True):
    """
    Compute the log-likelihood for given parameters (for testing outside CosmoSIS).

    Parameters
    ----------
    R : array_like
        Radial distances
    data_vector : array_like
        Observed values (f_cl or B depending on data_is_fcl)
    covariance : array_like
        Covariance matrix
    rs : float
        Scale radius
    b0 : float
        Amplitude
    data_is_fcl : bool
        If True, data is f_cl (contamination fraction) and model B is converted.
        If False, data is B (boost factor) and compared directly.

    Returns
    -------
    log_L : float
        Log-likelihood value
    chisq : float
        Chi-squared value
    model : ndarray
        Model predictions (in same units as data)
    """
    inv_cov = np.linalg.inv(covariance)

    # Compute boost factor model
    B_model = boost_factor_model(R, rs, b0)

    # Convert to f_cl if data is in f_cl units
    if data_is_fcl:
        model = boost_to_fcl(B_model)
    else:
        model = B_model

    diff = model - data_vector
    chisq = np.dot(diff, np.dot(inv_cov, diff))
    log_L = -0.5 * chisq

    return log_L, chisq, model


def generate_values_ini(bins, output_file, prior_range=(-1.0, 0.0, 1.0)):
    """
    Generate a CosmoSIS values .ini file for the given bins.

    Parameters
    ----------
    bins : list of tuple
        List of (richness_bin, redshift_bin) tuples
    output_file : str
        Output file path
    prior_range : tuple
        (min, start, max) for log parameters
    """
    lines = ["[boost_factor_params]"]
    lines.append("# Auto-generated parameter file for boost factor model")
    lines.append("# Format: logrs_lX_zY = min start max")
    lines.append("")

    for l, z in sorted(bins):
        suffix = f"l{l}_z{z}"
        lines.append(f"logrs_{suffix} = {prior_range[0]} {prior_range[1]} {prior_range[2]}")
        lines.append(f"logb0_{suffix} = {prior_range[0]} {prior_range[1]} {prior_range[2]}")

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Generated values file: {output_file}")


if __name__ == "__main__":
    # Quick test when run directly
    print("Boost Factor Likelihood Module")
    print("=" * 40)

    # Test the model function
    R_test = np.logspace(-1, 2, 50)
    B_test = boost_factor_model(R_test, rs=1.0, b0=0.3)
    print(f"Model test: B(R=1, rs=1, b0=0.3) = {boost_factor_model(np.array([1.0]), 1.0, 0.3)[0]:.4f}")
    print(f"Expected at x=1: (b0+3)/3 = {(0.3+3)/3:.4f}")
