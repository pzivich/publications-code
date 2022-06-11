# Usage

For ease of use across the system, the inverse probability weighting estimator for bridged treatment comparisons is
packaged into a local library called `chimera`. To install an editable version of `local`, use the following command

```commandline
cd BridgeComparisonIntro/Python/Chimera
python -m pip install -e .
```

The `-e` flag means that local changes in the `chimera` files. This way, as the library is updated, `chimera` does
not need to be re-installed with each update. A constant version is possible by removing the `-e` flag.

To use the bridging estimator, use the following command to import
```python
from chimera import SurvivalFusionIPW
```

# Dependencies and versions

The analysis was completed with Python 3.6.5 and using
NumPy:       1.19.5
SciPy:       1.5.4
Pandas:      1.1.5
Statsmodels: 0.12.2
Matplotlib:  3.3.4
Chimera:     0.0.3
