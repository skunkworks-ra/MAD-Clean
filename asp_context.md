# ASP and WAsp Context

## What ASP is

Asp-Clean (Bhatnagar & Cornwell 2004) is a scale-sensitive deconvolution algorithm. Instead of delta functions (Hogbom) or fixed scale sets (MS-Clean), it models the sky as a sum of Gaussians called **Aspen**, each with parameters {amplitude, location, scale (sigma)}. The scale is solved for optimally at each iteration -- the algorithm is adaptive, not fixed-scale.

Each Aspen P(p) = (1/sigma*sqrt(2pi)) * exp(-r^2 / 2*sigma^2).

The residual is: IR = ID - B*IM, where IM is the sum of all accepted Aspen convolved to the model image.

## WAsp (Wide-band Asp-Clean, Hsieh, Bhatnagar & Rau 2026)

WAsp replaces MS-Clean inside the MS-MFS (Multi-Scale Multi-Frequency Synthesis) framework. MS-MFS handles the wideband Taylor-term spectral modeling; WAsp handles the minor cycle scale modeling.

### Key improvements over original Asp-Clean

1. **Scale initialisation**: Initial scales are {0, W, 2W, 4W, 8W} where W = HWHM of PSF main lobe. A `largestscale` parameter overrides 8W when needed (e.g. partially-sampled large scales or extreme sidelobe cases).

2. **Single-Aspen update**: Original Asp-Clean maintains a permanent set and re-optimises all Aspen each iteration (expensive). WAsp uses only the latest Aspen to update model and residual (Eqs. 6-7). This approximates a diagonal covariance -- more Aspen needed but 3-20x faster.

3. **Fused deconvolution**: Automatically switches to zero-scale (Hogbom-like) iterations when residuals are dominated by compact emission. Triggered by: (a) peak residual below fusedthreshold, (b) Aspen amplitude near zero for 3 consecutive iterations, or (c) 5 of 10 consecutive iterations pick zero scale. NH = 51 (or 510 if residual RMS hasn't dropped 50%).

4. **Wideband**: WAsp optimises P(p) on TT0 (first Taylor term) only. The optimised Aspen is then used to update all Nt Taylor terms via the Hessian solve (Eq. 9). Ns is always 2 (zero scale + optimal scale), so the Hessian is small and recomputed each cycle.

### Normalisation for initial amplitude guess (Appendix A.1)

Convolve residual with each initial scale, find global peak F, then:
  F0 = F * sqrt(d / 2pi),  where d = sqrt(1/W^2 + 1/sigma_opt^2)

This prevents 0-scale collapse at the start and keeps the amplitude guess in a range where the optimiser converges.

### Optimisation (Appendix A.2-A.3)

Objective: chi^2 = ||IR - a*(B*P(p))||^2, minimised over {a, sigma}. Gradient expressions for both amplitude and scale are available (Eqs. A5-A6). Final implementation uses ALGLIB (not GSL or LBFGS++) for stability and 2x speed over GSL. Parameter scaling via `minlbfgssetcscale` needed because amplitude and sigma can differ by orders of magnitude.

## Performance (from paper)

- G55 supernova remnant (wideband): WAsp 5 major cycles / 33 min vs MS-MFS 7 cycles / 56 min.
- Jet simulation: WAsp 9 major cycles at gain=0.6 vs MS-MFS 15 cycles at gain=0.1 (or 35 at gain=0.4).
- Narrow-band: ~20x faster than original Asp-Clean at similar imaging fidelity.
- No tight masks needed in any test case.

## Relevance to MAD-Clean / SBI-Asp

The MDN-Asp work in this repo replaces the Aspen parameter optimisation step (Step 2b in the Asp-Clean loop) with a trained MDN that predicts the posterior over {amplitude, sigma} given a dirty cutout. The MDN output is a Gaussian mixture over 6D Aspen parameters (amplitude, x, y, sigma, plus two shape params). This substitutes the ALGLIB optimiser with a learned approximate posterior -- same outer loop, different inner solver.

PatchFlow is the next step: replace the 6D Gaussian Aspen representation entirely with a pixel patch, allowing the model to represent non-Gaussian morphology (shells, filaments) that a single Gaussian Aspen cannot.
