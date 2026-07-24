# Veros Payload Copy Compatibility Design

## Problem

VerCOR copy-owns setup payloads during preparation, runtime-state creation,
host stepping, and validation. The bundled Veros setup returns a native
`VerosState` payload. Its settings container stores `__fields__` as a live
`dict_keys` view, so Python `deepcopy()` fails even though VerCOR's existing
Veros `copy_state()` helper can otherwise construct an independent state.

This prevents `examples/run_veros_with_era5data.py` from completing coupler
preparation.

## Considered approaches

1. Normalize the Veros settings field names in `copy_state()` (recommended).
   Convert the copied settings container's `__fields__` view to an immutable
   tuple. This is private, Veros-specific, and preserves the existing public
   payload contract.
2. Add a public lifecycle payload-cloning callback. This would support more
   opaque third-party states, but expands the public API for one known bundled
   compatibility defect.
3. Preserve opaque payload leaves when `deepcopy()` fails. This would allow
   preparation to continue, but would violate VerCOR's state-isolation and
   validation guarantees.

## Design

Before `copy_state()` returns either its newly constructed jitted-path state or
the original non-jitted-path state, it replaces the returned settings object's
live `dict_keys` field-name view with a tuple holding the same names and order.
The latter path is required by the reported example. Veros metadata is static
after construction, so the tuple retains the required semantics while making
the native state compatible with VerCOR's generic copy-owned payload boundary.

No public API, component contract, payload type, stepping behavior, or output
behavior changes.

## Testing

Add a regression test using a minimal fake `VerosState` constructor whose
settings object reproduces the `dict_keys` member. Test both `jitted=False` and
`jitted=True` to prove that:

- `copy_state()` normalizes the copied settings field names to a tuple;
- the returned native state can be deep-copied; and
- the deep copy owns independent settings and variable containers.

Then run the focused Veros/component tests, the failing example through its
preparation boundary, static checks for changed files, and the fast test suite.
