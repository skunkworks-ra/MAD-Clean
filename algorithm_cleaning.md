# MAD-CLEAN flow tile-sweep — algorithm as currently implemented

Source of truth: `scripts/run_imaging_3c391.py` (`--solver flow`) +
`mad_clean/minor_cycle.py::minor_cycle_flow`. This file documents what the code
does today, line-for-line, so we can argue about it. It is descriptive, not
aspirational.

## Setup (once)

```
tclean(niter=0, calcpsf=True, calcres=True)      # major cycle "zero"
psf      = .psf image
residual = .residual image                        # = dirty
model    = .model image (zeros)
mask     = circle, radius mask_radius_px
dirty_peak = max(residual within mask)
```

## Major cycle loop   `for major in 0 .. max_major-1:`

```
residual = read .residual                         # fresh from CASA each time
res_peak = max(residual within mask)

# threshold RECOMPUTED every major cycle on the full residual
sigma     = MAD noise of residual within mask (sigma-clipped)
threshold = 3 * sigma
# (overridable: --global_threshold fixes sigma=thr/3 and threshold=thr)

if res_peak < threshold:  STOP (converged)

masked_residual = residual where mask else 0
```

### Minor cycle = TILE SWEEP (`minor_cycle_flow`)

```
model_update = zeros
total_passes = 0

for r0 in tile_origins(H, 128, stride=64):        # 50% overlap
  for c0 in tile_origins(W, 128, stride=64):
    sub = residual[r0:r0+128, c0:c0+128]          # VIEW, mutated in place
    if tile has no mask pixels: skip

    repeat up to inner_max (=3) times:            # passes per tile PER major cycle
      if total_passes >= max_components: break
      pr,pc = peak location in sub (within mask)
      peak  = sub[pr,pc]
      if peak < threshold:  break                 # tile done for this cycle

      # flow forward
      draws = flow.sample(sub, psf_cut, n_samples=8, n_steps=50)
      med   = median over draws
      mad   = 1.4826 * MAD over draws

      # GATE — both RELATIVE to this tile's own window peak wpk
      wpk   = max(med)
      keep  = (med >= conf_k*mad) AND (med >= speckle_frac*wpk)
      win   = med where keep else 0
      if win empty: break

      # COMMIT (feathered, gain-scaled)
      commit = loop_gain * win * hann_feather
      commit = commit where tile-mask else 0
      model_update[r0:r1, c0:c1] += commit

      # LOCAL residual clean — NO convolution
      sub[commit>0] *= (1 - loop_gain * feather)
      sub[pr,pc]    *= (1 - loop_gain)            # force peak to drain

      total_passes += 1
```

### Back in the major cycle

```
model = read .model
model += model_update                             # accumulate into persistent CASA model
write .model

tclean(niter=0, calcpsf=False, calcres=True, restart=True)   # exact re-image
new_residual = read .residual
new_peak = max(new_residual within mask)

if new_peak > res_peak * (1 + divergence_tol):  STOP (diverged)
```

## Three points where the ambiguity / current failure lives

1. **Tile sweep visits a fixed grid, not peak order.** Per tile, at most
   `inner_max=3` flow passes per major cycle, then move on regardless of whether
   the tile is clean.

2. **The gate is entirely relative to the tile's own peak `wpk`** — there is no
   absolute noise floor on what gets committed. In a faint/empty tile `wpk` is
   small, so `speckle_frac*wpk` is a tiny floor and the flow's low-level texture
   commits. **This is the speckle carpet.**

3. **The minor-cycle residual update is a local multiplicative scrub, not a PSF
   subtraction.** The only exact subtraction is the `tclean` re-image at the
   bottom of the major loop.

## Known symptoms (3C391 / G55, 2026-06-29)

- Model reconstructs the connected shell well (extended structure is captured).
- Fine speckle carpet dusts the whole masked field (point 2).
- Convergence stalls *above* the noise floor: light peel (`inner_max=3`) goes
  0.121 -> 0.020 over 25 major cycles, monotone but asymptoting, never reaching
  3-sigma. Residual retains large-scale (super-beam) structure while the model
  accumulates small-scale speckle.
