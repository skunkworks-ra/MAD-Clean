# Stage 3 — Uncertainty Image as Posterior Pushforward

**Status: designed and APPROVED (2026-06-04), not yet implemented.**
Pick this up and implement; the design decision is settled.

## One-line goal

Replace the current flux-only + pixel-wise-`max` uncertainty heuristic in
`scripts/run_imaging.py::build_uncertainty_image` with the statistically correct
**posterior pushforward**: sample the full 6-D MDN posterior per committed
component, render each draw, and combine per-pixel variances by **quadrature
sum** (variances of independent components add).

## Approved decision (do not relitigate)

- **Quadrature-sum pushforward**, not pixel-wise `max`. Where sources overlap the
  new map reads higher — that is correct. This changes the numbers in existing G55
  runs; that is expected and intended.
- **Local-stamp rendering** for tractability (see Performance).
- Sample the **full 6-D** $\theta$ (position, flux, shape), not flux alone — so
  positional and morphological uncertainty propagate into per-pixel flux
  uncertainty.

## Math

For committed component $i$ with posterior $q_{\psi,i}(\theta\mid r,c)$, draw
$\theta_i^{(s)}\sim q_{\psi,i}$, $s=1,\dots,S$. The image posterior is
$I^{(s)}(p)=\sum_i g_{\theta_i^{(s)}}(p)$. Components are independent given the
residual, so

$$
\operatorname{Var}[I(p)] = \sum_i \operatorname{Var}_{\theta\sim q_{\psi,i}}[g_\theta(p)],
\qquad
\sigma_I(p) = \sqrt{\sum_i \operatorname{Var}_i[g(p)]}.
$$

So we accumulate **per-component per-pixel variance** into a global variance map
and take the sqrt at the end. No need to build $S$ full images.

## Algorithm (per committed component `i`)

1. Draw `S` samples from the mixture posterior `commit.params`:
   - sample component index `k ~ Categorical(softmax(logits))`;
   - sample `θ̃ = μ_k + s_k · ε`, `ε ~ N(0, I)` in the 7-emitted space;
   - decode to 6-D (`_decode_7d_to_6d` / `decode_pa`); unstandardise `log_flux`
     via `unstandardise_log_flux`; clip flux to the training range as today.
2. For each draw, render the elliptical Gaussian **into a local bounding box**
   around `(cx, cy)` of half-width `≈ ceil(5 * sig_maj_max)` (clamped to image
   bounds), not the full frame.
3. Maintain an **online (Welford) per-pixel mean/M2** over the `S` stamps within
   that bbox → per-pixel variance for component `i`.
4. **Add** that variance into the global variance accumulator at the bbox slice.
5. After all components: `uncertainty = sqrt(global_variance)`; write as the
   `.uncertainty` CASA image exactly as now.

## Performance (why local stamps are required)

`render_aspen` (`mad_clean/minor_cycle.py:56`) builds a full `H×W` meshgrid every
call. A true pushforward is `S × n_components` renders (e.g. 200 × 16,872 on G55).
Full-frame that is ~3.4M renders of a 1280² grid — a non-starter. Rendering into a
`(~10σ)²` bbox makes each render tiny and the whole map tractable. Keep the
existing full-frame `render_aspen` unchanged for the **model** image.

## Files / functions to touch

- `scripts/run_imaging.py::build_uncertainty_image` — rewrite the loop body to the
  algorithm above. Keep the signature `(all_commits, shape, n_samples, device)`.
- Add a **local-stamp renderer**. Either a new helper in
  `mad_clean/minor_cycle.py` (e.g. `render_aspen_bbox(cx, cy, flux, sig_maj,
  sig_min, pa, shape) -> (stamp, (r0, r1, c0, c1))`) reused by the pushforward, or
  a private helper inside `run_imaging.py`. Prefer the shared helper so the model
  image and the uncertainty map use identical geometry.
- No change to `render_aspen` (full-frame) — model image path stays as is.

## Output semantics / caveats to bake into the docstring

- The map is a **marginal** pushforward: per-component posteriors are conditional
  on the residual at commit time, treated independently across major cycles. State
  this; it is not a joint image posterior.
- It is a **Layer-2** object (see `draft_paper.md` §4.9). It does **not** capture
  model misspecification (Layer 3): OOD morphology yields confident-but-wrong
  posteriors and therefore deceptively clean uncertainty.

## Acceptance / verification

- On a small synthetic field (a couple of components), confirm the new map equals
  $\sqrt{\sum_i \operatorname{Var}_i}$ and that two overlapping components give a
  higher central value than either alone (the quadrature-add behaviour the old
  `max` suppressed).
- Sanity: with `S` large, a single isolated bright point's `σ_I` peak should track
  the radiometric flux error implied by its posterior flux std.
- Re-run G55 (`results/g55_mdn_asp_v2`-style command) and confirm the map is
  finite, non-negative, and peaks on the brightest/most-uncertain components.

## Related

- Design rationale and three-layer framing: `draft_paper.md` §4.9.
- Audit context: `audit_training.md` (finding 2.1 noise blind-spot also limits
  absolute calibration of this map on real data).
