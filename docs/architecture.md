# Architecture

How the models are built, and why. The code is the reference; each section names
the file it describes.

## 1. The graph — `src/utils/graph.py`

The Saudi domain is a 25 × 31 grid at 0.75°, of which **251 cells are valid land**.
Each cell is a node. Edges join the 8 surrounding neighbours, so a wind at 45° is
representable in one hop rather than as two orthogonal steps.

Node positions are stored in a geographic frame, `(+east, +north)`. This matters:
latitude *decreases* with row index in ERA5 and CAMS, so a naive row-wise gradient
is the southward derivative. That sign error is silent — the model still trains and
the dust simply drifts the wrong way — so the convention is pinned by
`tests/test_dust_physics.py`.

## 2. Transport as message passing — `src/layers/flux.py`

### `PhysicsFluxLayer`

For each directed edge `j → i` the message is an MLP over both nodes' features plus
two physical terms computed from the **unnormalised** wind and concentration:

```
w_ji      = u_j · r_ji                    source-cell wind projected on the edge
adv_ji    = w_ji · (C_j − C_i)            upwind advective flux
m_ji      = MLP([x_i, x_j, w_ji, adv_ji])
h_i       = Σ_j m_ji                      net inflow (aggr = "add")
```

`r_ji = pos_i − pos_j`. The directional prior is built into the message rather
than learned: the network is not asked to discover that dust moves downwind, only
how much of it survives the trip.

### `ConservativeFluxLayer`

`PhysicsFluxLayer` computes the two directions of a face independently, so it is
free to create dust at one end and lose it at the other. The conservative variant
is a finite-volume scheme instead:

```
u_n  = ((u_i + u_j) / 2) · n̂_ji          midpoint normal velocity
c_up = C_j if u_n > 0 else C_i            upwind concentration
F_ji = softplus(s) · tanh(gate(x_i + x_j)) · u_n · c_up
```

Swapping `i` and `j` flips `n̂`, leaves `x_i + x_j` unchanged and selects the same
upwind cell, so `F_ij = −F_ji` exactly and the transport term sums to zero over the
graph. `tests/test_conservation.py` checks this to floating-point precision. The
gate is bounded with `tanh` because the unbounded first version grew exponentially
with hop count (1.8 × 10¹⁶ larger than `PhysicsFluxLayer` at 12 hops).

## 3. Models

### `StationDustModel` — the model behind the headline results

`experiments/station_dust/train_station_dust.py`. Two stacked `PhysicsFluxLayer`
hops with a residual connection, dropout 0.3, and a linear head giving one logit
per visibility threshold at every node. Input features are standardised with
buffers fitted on the training years only.

Two hops is a 152 km receptive field: enough at 3 h, far short of the ~500 km dust
travels in 24 h at p90 wind. That is what the next model fixes.

### `DustTransportGNN` — the successor, under evaluation

`src/models/dust_transport.py`, enabled with `--transport`.

```
h⁽⁰⁾ = encode(x)                                          all nodes
for k in 1..K:
    h⁽ᵏ⁾ = keep · h⁽ᵏ⁻¹⁾ + relu(transport(h⁽ᵏ⁻¹⁾)) + source(x_emission)
logits = head(h⁽ᴷ⁾)                                       interior nodes only
```

- **Every node emits, and emission is re-injected each hop.** Dust is lifted
  continuously along a shamal's path; a single injection would model a puff.
- **One shared transport operator**, so parameters do not grow with the horizon.
- **Hops are derived from the horizon**, `hops_for_horizon(h) = ⌈5.8 m/s · h / 76 km⌉`,
  clamped to [2, 12]:

  | Horizon | Dust travels (p90 wind) | Hops | Reach |
  |---|---:|---:|---:|
  | 3 h | 63 km | 2 | 152 km |
  | 24 h | 501 km | 7 | 532 km |
  | 48 h | 1,002 km | 12 (capped) | 912 km |

- **Removal**: `keep` is a learned global survival fraction, or with
  `--deposition` a per-node `exp(−(v_dry + λ·P)·Δt / H)` from precipitation `P` and
  mixing depth `H`.
- **Readout at interior cells only.** Boundary cells have a truncated stencil and
  no inflow, so they are poor targets but valuable sources: 42% of observed storms
  had no local emission.
- **Optional CAMS auxiliary head** (`--aux-cams`): a sibling head regressing CAMS
  PM10 at all 251 nodes, to add dense supervision alongside ~11–26 station cells.

## 4. Target and loss — `src/data/station_target.py`

The target is **observed** dust, from METAR airport visibility, as three nested
events:

| Event | Visibility | Daily base rate (interior, test 2021–2024) |
|---|---|---:|
| `dusty` | < 5 km | 8.09% |
| `dust_storm` | < 1 km | 1.26% |
| `severe` | < 0.4 km | 0.23% |

- One model per horizon (3 h, 24 h, 48 h), each fed ERA5 meteorology **at its own
  valid time**. ERA5 at `t + h` stands in for a perfect wind forecast — the same
  idealisation AI-DUST's evaluation makes — so scores are an upper bound on what an
  operational NWP-driven version would achieve.
- The forward pass covers the whole graph; the loss is masked to cells that contain
  a station.
- Class-weighted BCE with `pos_weight = (1 − p) / p`, capped at 200. `--ordinal`
  replaces the three sigmoids with a SORD head (`src/models/ordinal.py`) whose
  probabilities are nested by construction.
- `--dust-gate` drops fog and rain hours (by METAR weather code) from the labels
  while keeping them as inputs.

## 5. The original Gulf PM10 model — `src/models/pi_gnn.py`

The project's first phase regressed CAMS PM10 on a 7 × 12 grid over the Arabian
Gulf. `PIDustModel` stacks the same `PhysicsFluxLayer` and predicts a residual in
log space, `Ĉ_{t+1} = C_t + Δ`, with a zero-initialised head so training starts
from persistence. Training balances two terms with homoscedastic uncertainty
weighting (Kendall et al., 2018):

```
L_data    = MSE(Ĉ_{t+1}, C_{t+1})
L_physics = ‖(Ĉ_{t+1} − C_t) + (u · ∇C_t) Δt‖²
L         = Σ_k  L_k / (2σ_k²) + log σ_k
```

with `dy ≈ 83.5 km`, `dx ≈ 74.9 km` at 26°N and the row-axis derivative negated for
the descending latitude. Mass conservation is reported as a diagnostic, not
penalised: the domain is open. Measured on the Saudi domain, adding physics as a
loss trades accuracy for physical fidelity (see the README) — the reason the later
models encode physics in the architecture and features instead.
