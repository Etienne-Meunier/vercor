# Veros Linear Solver Cache Design

## Problem

Veros 1.6.2 memoizes its linear solver by the identity of the `VerosState`
passed to `veros.core.external.solvers.get_linear_solver()`. VerCOR copy-owns
host component payloads before every component step, and its Veros adapter also
copies native state before mutation. Each coupling step therefore reaches Veros
with a new state identity even though the grid and solver geometry are
unchanged.

The real four-degree setup reproduces the defect: Veros' solver cache grows
from one entry after setup to two and then three entries after two copied-state
steps, with `Computing ILU preconditioner...` emitted for both steps. Besides
recomputing the ILU preconditioner, the identity cache retains every copied
state.

## Scope and cache lifetime

The cached solver belongs to one `VerosGCMSetupState`, and therefore to one
Veros component/Coupler instance. A separately constructed component performs
its own Veros setup and owns its own solver. Solvers are not shared across
components or Coupler instances, even when their configurations happen to be
equal.

The solver is a static operational resource derived from setup-time grid
geometry. Evolving Veros model state remains exclusively in
`SetupResult.payload` and `StepResult.payload`, preserving VerCOR's functional
payload-ownership contract.

## Considered approaches

1. Retain the setup-created solver on the Veros component and bind it to each
   copied state only while stepping (recommended). This matches the requested
   component lifetime, preserves the native payload, and prevents transient
   copied states from accumulating in Veros' process-level identity cache.
2. Replace the native payload with a wrapper containing both state and solver.
   This would make cache ownership explicit in runtime state, but it would
   broaden the payload contract, require output and snapshot changes, and make
   defensive copying of the opaque solver more complex.
3. Introduce a process-wide cache keyed by a grid/configuration fingerprint.
   This could share solvers between Couplers, but it exceeds the requested
   lifetime and would require complete invalidation rules for every setting and
   geometry value used to assemble the solver.

## Design

`VerosGCMSetupState` captures the solver that Veros creates during
`model.setup()`. Looking it up through Veros' existing memoized accessor does
not construct a second solver because the original setup state is already a
cache key.

The existing private copy-before-mutate step helper receives this component-
owned solver. Immediately before invoking the native Veros step, a focused
private helper temporarily maps the copied `VerosState` identity to that
solver in Veros' own cache. The native step continues to call Veros without
modification and receives the retained solver through its normal accessor.

The temporary association is scoped with `try`/`finally`. If the copied state
had no prior entry, the helper removes it after the step. If an entry already
existed, the helper restores it. Cleanup therefore occurs on both successful
and failing steps, avoids cache growth, and does not disturb Veros' original
setup-state entry.

No public VerCOR API, component declaration, runtime payload type, model
numerics, output contract, or dependency order changes.

## Testing

Tests are written before production changes and cover these behaviors:

- two distinct copied state objects use the same component-owned solver without
  invoking the solver constructor again;
- the copied-state cache key is absent after a successful native step;
- an exception from the native step still removes the copied-state key;
- a pre-existing cache entry is restored instead of overwritten; and
- existing Veros copy isolation, stepping, output, and component tests remain
  unchanged.

The real Veros four-degree reproduction is rerun to confirm that two sequential
copied-state steps emit no additional ILU initialization and leave the solver
cache at its setup-time size. Final verification includes Black, flake8, mypy,
compileall, whitespace checks, focused tests, the fast suite, the full suite,
and branch coverage in the `scipy` conda environment.
