# Improved Boost Factor Module - DES Y1

This folder contains the improved boost factor likelihood code for DES Year 1 data with multi-bin automation.

**Reference**: McClintock et al. (2019), arXiv:1805.00039

## Files

| File | Description |
|------|-------------|
| `bf_likelihood_improved.py` | Main likelihood module for CosmoSIS |
| `bf_pipeline_improved.ini` | CosmoSIS pipeline configuration |
| `bf_values_all_bins.ini` | Parameter priors for all 12 bins |
| `test_bf_likelihood.ipynb` | Jupyter notebook to test the module |

## Usage

### Testing locally (without CosmoSIS):
```python
from bf_likelihood_improved import (
    boost_factor_model,
    load_y1_data,
    discover_y1_bins,
    compute_likelihood_standalone
)

# Load data
data = load_y1_data('path/to/y1/profiles', l=0, z=0, n_points=8)

# Compute likelihood
log_L, chi2, model = compute_likelihood_standalone(
    data.R, data.data_vector, data.covariance,
    rs=1.0, b0=0.3
)
```

### Running with CosmoSIS:
```bash
cosmosis bf_pipeline_improved.ini
```

## Data Files (Y1)

Expected file format:
- `full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost.dat` - R, B, sigma_B
- `full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost_cov.dat` - Covariance matrix
