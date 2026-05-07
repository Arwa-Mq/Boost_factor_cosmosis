# Boost Factor Likelihood for CosmoSIS

Fitting boost factor measurements from DES galaxy cluster weak lensing data.

## Use the Improved Module

**The updated code with multi-bin automation is in the `improved_module/` folder.**

```
improved_module/
├── bf_likelihood_improved.py   # Main likelihood module
├── bf_pipeline_improved.ini    # CosmoSIS pipeline config
├── bf_values_all_bins.ini      # Parameter priors (12 bins)
├── test_bf_likelihood.ipynb    # Test notebook
└── README.md                   # Documentation
```

## Quick Start

### 1. Copy to your working directory
```bash
cp improved_module/bf_likelihood_improved.py /path/to/your/cosmosis/modules/
cp improved_module/bf_pipeline_improved.ini /path/to/your/run/
cp improved_module/bf_values_all_bins.ini /path/to/your/run/
```

### 2. Update the data path in `bf_pipeline_improved.ini`
```ini
[boost_factor_likelihood]
data_path = /your/path/to/y1/profiles
```

### 3. Run with CosmoSIS
```bash
cosmosis bf_pipeline_improved.ini
```

## Model

The boost factor model is:

```
B(R) = 1 + b0 * (1 - f(x)) / (x^2 - 1)
```

where `x = R / rs`, and parameters are:
- `rs`: scale radius
- `b0`: amplitude

