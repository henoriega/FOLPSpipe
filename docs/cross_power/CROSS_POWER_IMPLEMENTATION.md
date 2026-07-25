# Cross-Power Implementation

This document summarizes the Stage 2 equal-time two-tracer redshift-space
power-spectrum implementation in `folps/folps.py`.

## API

The existing auto-spectrum API is preserved. Cross spectra are enabled by
passing tracer-B parameters and optional pair-level nuisance parameters:

```python
multipoles = RSDMultipolesPowerSpectrumCalculator(model="EFT")

pkmu_ab = multipoles.get_rsd_pkmu(
    k, mu, pars_a, table, table_now,
    IR_resummation=True,
    damping=None,
    pars_b=pars_b,
    cross_nuisance=cross_nuisance_ab,
)

pells_ab = multipoles.get_rsd_pkell(
    kobs, qpar, qper, pars_a, table, table_now,
    bias_scheme="folps",
    damping=None,
    nmu=8,
    ells=(0, 2, 4),
    IR_resummation=True,
    pars_b=pars_b,
    cross_nuisance=cross_nuisance_ab,
)
```

`pars_b=None` remains auto mode. `bias_scheme_b` may be supplied to
`get_rsd_pkell`; otherwise tracer B uses the same bias scheme as tracer A.

The implementation uses small internal `NamedTuple` containers:

```text
bias = (b1, b2, bs2, b3nl)
nuisance = (alpha0, alpha2, alpha4, ctilde,
            alphashot0, alphashot2, PshotP, X_FoG_p)
```

`cross_nuisance` can be either the 8-value nuisance tuple above or a full
12-value Folps power-spectrum parameter array, in which case the last 8 values
are used by explicit convention. If `pars_b` is supplied and `cross_nuisance`
is omitted, the call raises `ValueError`: pair-level EFT and stochastic
parameters must be supplied explicitly for a cross-spectrum. They are not
computed from either tracer's auto-spectrum nuisance values.

## Implemented Equations

The public auto path and cross path both call one shared pair-level contraction,
`RSDMultipolesPowerSpectrumCalculator._get_eft_pkmu_pair`.

The IR-resummed linear term uses the cross Kaiser polynomial:

```text
P_K,AB = (b1_a + f(k) mu^2) (b1_b + f(k) mu^2) P_L.
```

The density-density loop is polarized as:

```text
b1_a b1_b P_dd
+ (b1_a b2_b + b2_a b1_b) P_b1b2
+ (b1_a bs2_b + bs2_a b1_b) P_b1bs2
+ b2_a b2_b P_b2b2
+ (b2_a bs2_b + bs2_a b2_b) P_b2bs2
+ bs2_a bs2_b P_bs2bs2
+ (b1_a b3nl_b + b3nl_a b1_b) sigma23pkl.
```

The density-velocity sector is:

```text
f0 mu^2 [P_delta_a_theta + P_delta_b_theta],

P_delta_x_theta =
    b1_x P_delta_theta
  + b2_x P_b2theta
  + bs2_x P_s2theta
  + b3nl_x F(k) sigma23pkl.
```

The velocity-velocity sector is common:

```text
f0^2 mu^4 P_theta_theta.
```

The D term is:

```text
D_AB =
  b1_a b1_b f0^2 D2
+ 0.5 (b1_a + b1_b) f0^3 D3
+ f0^4 D4.
```

The G term replaces the auto Kaiser polynomial by the cross Kaiser polynomial
using the existing FolpsD convention:

```text
G_AB = -(k mu f0)^2 sigma2w
       [b1_a b1_b P_L
        + (b1_a + b1_b) f0 mu^2 P_delta_theta^L
        + f0^2 mu^4 P_theta_theta^L].
```

For `A_full=False`, the A term is:

```text
A_AB =
  b1_a b1_b f0 mu^2 I1udd_1
+ 0.5 (b1_a + b1_b) f0^2 (mu^2 I2uud_1 + mu^4 I2uud_2)
+ f0^3 (mu^4 I3uuu_2 + mu^6 I3uuu_3).
```

For `A_full=True`, the following polarized terms are added:

```text
Delta A_b2 =
  0.25 (b2_a b1_b + b1_a b2_b) f0 mu^2 I1udd_1_b2
+ 0.25 (b2_a + b2_b) f0^2
   (mu^2 I2uud_1_b2 + mu^4 I2uud_2_b2),

Delta A_bs2 =
  0.25 (bs2_a b1_b + b1_a bs2_b) f0 mu^2 I1udd_1_bs2
+ 0.25 (bs2_a + bs2_b) f0^2
   (mu^2 I2uud_1_bs2 + mu^4 I2uud_2_bs2).
```

The EFT and stochastic terms use the existing polynomial basis and
normalization:

```text
P_ct = (alpha0_ab + alpha2_ab mu^2 + alpha4_ab mu^4) k^2 P_L
P_ct,NLO = ctilde_ab (k mu f0)^4 sigma2w^2 P_K,AB
P_stoch = PshotP_ab [alphashot0_ab + alphashot2_ab (k mu)^2].
```

## Table Reuse

No new FFTLog, IR, no-wiggle, or wiggle tables are introduced. The
implementation reuses:

- the existing loop table and no-wiggle table;
- the existing `A_full=True` six-row extension;
- the existing `sigma2w`, `sigma2_NW`, and `delta_sigma2_NW` scalar tail;
- the existing interpolation, AP remapping, IR resummation, and multipole
  quadrature.

## Matrix Conclusion

The Stage 1B endpoint audit concluded that no new analytic FFTLog matrices are
needed for the equal-time cross A term. Endpoint exchange swaps both matrix
arguments and the FFTLog coefficient vectors, so the existing scalar rows can be
polarized with the coefficients listed above. This is used for both
`A_full=False` and `A_full=True`.

## Limitations

This implementation is power-spectrum only and assumes one common effective
redshift, common matter and velocity fields, no velocity bias, plane-parallel
geometry, and even multipoles.

Cross spectra with `pars_b` are currently restricted to `model="EFT"` and
`damping=None`. Auto spectra keep the existing TNS/FolpsD damping behavior. No
arithmetic-average, geometric-average, `sqrt(D_AA D_BB)`, or independent
cross-FoG damping prescription is implemented.

The implementation does not add bispectrum cross-correlations, wide-angle
terms, relativistic terms, odd multipoles, DESI data loading, or window
convolution.

## Tests

Relevant commands:

```bash
/opt/anaconda3/envs/aaenv/bin/python test_cross_power_spectrum.py
/opt/anaconda3/envs/aaenv/bin/python test_folps_numpy.py
/opt/anaconda3/envs/aaenv/bin/python test_folps_jax.py
/opt/anaconda3/envs/aaenv/bin/python test_compare_folps_numpy_vs_jax.py --skip-run
```

`test_cross_power_spectrum.py` enforces:

- auto-limit recovery for `A_full=False` and `A_full=True`;
- IR off and IR on auto-limit checks;
- pkmu and multipole checks;
- exchange symmetry for asymmetric tracer biases;
- explicit rejection of missing cross-spectrum pair nuisance parameters;
- linear Kaiser recovery when loops, EFT, stochastic, and IR terms are disabled;
- one-row synthetic-table checks for density loops, density-velocity terms,
  velocity-velocity terms, `A`, full-`A`, `D`, and `G` responses;
- AP-remapped multipole consistency with direct pkmu quadrature;
- current auto-spectrum regression against existing backend artifacts;
- NumPy/JAX cross agreement with `rtol=5e-3`, `atol=5e-2`.

## Tutorial notebook

A public, self-contained NumPy tutorial for the equal-time two-tracer
power-spectrum API is available at
[`notebooks/example_cross_power_numpy.ipynb`](../../notebooks/example_cross_power_numpy.ipynb).

## Example

```python
pars_a = [
    1.65, -0.46, -4.0 / 7.0 * (1.65 - 1.0), 32.0 / 315.0 * (1.65 - 1.0),
    0.7, -1.3, 0.2, 0.0, 0.015, -0.45, 4800.0, 0.0,
]
pars_b = [
    1.10, 0.23, -4.0 / 7.0 * (1.10 - 1.0), 32.0 / 315.0 * (1.10 - 1.0),
    0.2, -0.4, 0.1, 0.0, 0.0, 0.0, 3600.0, 0.0,
]
cross_nuisance_ab = [0.4, -0.9, 0.15, 0.0, 0.01, -0.2, 3600.0, 0.0]

multipoles = RSDMultipolesPowerSpectrumCalculator(model="EFT")
p0_ab, p2_ab, p4_ab = multipoles.get_rsd_pkell(
    kobs, 1.0, 1.0, pars_a, table, table_now,
    damping=None,
    IR_resummation=True,
    pars_b=pars_b,
    cross_nuisance=cross_nuisance_ab,
)
```
