from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

import vercor.output._runtime as output_runtime_module
from tests._architecture_support import package_import_cycles, source_for
from tests._coverage_support import make_test_grid
from vercor import Clock, ComponentSpec, Coupler, DataComponent, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.exceptions import CouplerError
from vercor._runtime.topology_state import RuntimeTopologyMaps


def test_prepared_coupling_owns_single_normalized_runtime_boundary() -> None:
    prepared_module = importlib.import_module("vercor._runtime.prepared")
    PreparedCoupling = prepared_module.PreparedCoupling

    assert is_dataclass(PreparedCoupling)
    assert getattr(PreparedCoupling, "__dataclass_params__").frozen is True
    assert [field.name for field in fields(PreparedCoupling)] == [
        "components",
        "exchanges",
        "run_order",
        "contracts",
        "topology_maps",
        "dispatch_context",
        "clock",
        "settings",
        "runtime",
        "interrupts",
        "component_fingerprints",
    ]


def test_coupler_owns_one_optional_prepared_coupling() -> None:
    coupler_source = Path("vercor/coupler.py").read_text(encoding="utf-8")

    assert "self._prepared: PreparedCoupling | None = None" in coupler_source
    assert "def _ensure_prepared(" in coupler_source
    assert "RuntimeInputs" not in coupler_source
    assert "_runtime_initialized" not in coupler_source
    assert "CouplerRuntimeResources" not in coupler_source


@pytest.mark.fast_always
def test_contracts_and_dispatch_are_built_once_across_public_runtime_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_module = importlib.import_module("vercor._runtime.prepared")
    initialization_module = importlib.import_module("vercor._runtime.initialization")
    calls = {"contracts": 0, "dispatch": 0}
    original_contracts = initialization_module.build_exchange_contracts
    original_dispatch = prepared_module.build_runtime_dispatch_context

    def counting_contracts(*args: Any, **kwargs: Any) -> Any:
        calls["contracts"] += 1
        return original_contracts(*args, **kwargs)

    def counting_dispatch(*args: Any, **kwargs: Any) -> Any:
        calls["dispatch"] += 1
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(
        initialization_module,
        "build_exchange_contracts",
        counting_contracts,
    )
    monkeypatch.setattr(
        prepared_module,
        "build_runtime_dispatch_context",
        counting_dispatch,
    )
    monkeypatch.setattr(
        output_runtime_module,
        "write_coupler_runtime_outputs",
        lambda **kwargs: None,
    )

    component = DataComponent.from_fields(
        "MODEL",
        make_test_grid(name="prepared-once"),
        {"temperature": 280.0},
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )

    initial_state = coupler.initial_state()
    prepared = coupler._prepared
    final_state = coupler.run(state=initial_state)
    coupler.write_outputs(final_state, write_snapshots=False)

    assert calls == {"contracts": 1, "dispatch": 1}
    assert coupler._prepared is prepared


@pytest.mark.fast_always
def test_write_outputs_rejects_incompatible_supplied_state_before_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1)
    coupler = Coupler(
        clock,
        components=(
            DataComponent.from_fields(
                "MODEL",
                make_test_grid(name="output-state-model"),
                {"temperature": 280.0},
            ),
        ),
        run_order=("MODEL",),
    )
    foreign_coupler = Coupler(
        clock,
        components=(
            DataComponent.from_fields(
                "FOREIGN",
                make_test_grid(name="output-state-foreign"),
                {"temperature": 281.0},
            ),
        ),
        run_order=("FOREIGN",),
    )
    foreign_state = foreign_coupler.initial_state()
    output_calls: list[str] = []
    monkeypatch.setattr(
        output_runtime_module,
        "write_coupler_runtime_outputs",
        lambda **kwargs: output_calls.append("write"),
    )

    error: CouplerError | None = None
    try:
        coupler.write_outputs(foreign_state, write_snapshots=False)
    except CouplerError as exc:
        error = exc

    assert output_calls == []
    assert error is not None
    assert "MODEL" in str(error)
    assert "missing from runtime state" in str(error)


@pytest.mark.fast_always
def test_public_mutator_invalidates_and_rebuilds_preparation() -> None:
    component = DataComponent.from_fields(
        "MODEL",
        make_test_grid(name="prepared-invalidation"),
        {"temperature": 280.0},
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )

    coupler.initial_state()
    first_prepared = coupler._prepared
    coupler.set_run_order(("MODEL",))

    assert first_prepared is not None
    assert coupler._prepared is None

    coupler.initial_state()

    assert coupler._prepared is not None
    assert coupler._prepared is not first_prepared


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "mutation",
    ("name", "grid", "spec", "seeded-field", "settings"),
)
def test_direct_component_mutation_after_preparation_is_rejected(
    mutation: str,
) -> None:
    component = DataComponent.from_fields(
        "MODEL",
        make_test_grid(name="prepared-mutation"),
        {"temperature": 280.0},
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    state = coupler.initial_state()

    if mutation == "name":
        component.name = "RENAMED"
    elif mutation == "grid":
        component.grid = make_test_grid(name="prepared-mutation-replacement")
    elif mutation == "spec":
        component.configure(ComponentSpec(outputs=("temperature",)))
    elif mutation == "seeded-field":
        component.seed_field("temperature", jnp.full(component.grid.shape, 281.0))
    else:
        component.settings.set("missval", -999.0)

    with pytest.raises(
        CouplerError,
        match="changed after preparation.*configure.*before preparation|changed after preparation.*create.*Coupler",
    ):
        coupler.run(state=state)


@pytest.mark.fast_always
def test_runtime_topology_maps_are_frozen_read_only_views() -> None:
    key = ("SRC", "DST", "bilinear")
    maps = RuntimeTopologyMaps(
        regridders={key: object()},
        binary_masks={key: jnp.ones((2, 2))},
        fractional_masks={key: jnp.ones((2, 2))},
    )

    assert getattr(RuntimeTopologyMaps, "__dataclass_params__").frozen is True
    with pytest.raises(TypeError):
        maps.regridders[key] = object()  # type: ignore[index]
    with pytest.raises(TypeError):
        maps.binary_masks[key] = jnp.zeros((2, 2))  # type: ignore[index]
    with pytest.raises(TypeError):
        maps.fractional_masks[key] = jnp.zeros((2, 2))  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        maps.regridders = {}  # type: ignore[misc]


@pytest.mark.fast_always
def test_precision_capability_and_allocation_policy_remain_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_module = importlib.import_module("vercor._runtime.prepared")

    class FakeConfig:
        def __init__(self, enabled: bool) -> None:
            self.enabled = enabled
            self.updates: list[tuple[str, bool]] = []

        def read(self, name: str) -> bool:
            assert name == "jax_enable_x64"
            return self.enabled

        def update(self, name: str, value: bool) -> None:
            assert name == "jax_enable_x64"
            self.updates.append((name, value))
            self.enabled = value

    fake_config = FakeConfig(enabled=False)
    monkeypatch.setattr(prepared_module.jax, "config", fake_config)
    x64_component = DataComponent.from_fields(
        "X64",
        make_test_grid(name="x64-capability"),
        {"temperature": 280.0},
    )
    x64_coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(x64_component,),
        run_order=("X64",),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
    )

    x64_coupler.initial_state()

    assert fake_config.updates == [("jax_enable_x64", True)]

    fake_config.enabled = True
    fake_config.updates.clear()
    f32_component = DataComponent.from_fields(
        "F32",
        make_test_grid(name="f32-policy"),
        {"temperature": 280.0},
    )
    f32_coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(f32_component,),
        run_order=("F32",),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=False)),
    )

    state = f32_coupler.initial_state()

    assert fake_config.updates == []
    assert state.component("F32").field("temperature").dtype == jnp.float32


def test_runtime_preparation_module_owns_runtime_state_preparation() -> None:
    preparation_path = Path("vercor/_runtime/preparation.py")
    assert preparation_path.exists()

    preparation_source = preparation_path.read_text(encoding="utf-8")
    assert "class PreparedRuntimeState" not in preparation_source
    assert "def runtime_state_from_components(" in preparation_source
    assert "def validate_runtime_state(" in preparation_source
    assert "def create_runtime_state(" in preparation_source
    assert "def prepare_runtime_state(" in preparation_source
    assert "RuntimePreparationInputs" not in preparation_source
    assert "return PreparedRuntimeState" not in preparation_source


@pytest.mark.fast_always
def test_component_topology_module_only_owns_role_lookup() -> None:
    component_topology_source = source_for("vercor/_runtime/component_topology.py")
    topology_source = source_for("vercor/_runtime/topology.py")
    surface_masks_source = source_for("vercor/_runtime/surface_masks.py")
    topology_policy_source = source_for("vercor/_runtime/topology_policy.py")
    initialization_source = source_for("vercor/_runtime/initialization.py")

    assert "VALID_TOPOLOGY_COMPONENT_NAMES" not in component_topology_source
    assert "def validate_component_topology_names(" not in component_topology_source
    assert "def require_component(" in component_topology_source
    assert "def get_component(" not in component_topology_source
    assert ".values()" not in component_topology_source
    assert "def validate_component_topology_names(" not in topology_source
    assert "def get_component(" not in topology_source
    assert "def require_component(" not in topology_source
    assert "from vercor._runtime.component_topology import" not in topology_source
    assert "from vercor._runtime.component_topology import" in surface_masks_source
    assert (
        "from vercor._runtime.component_topology import" not in topology_policy_source
    )
    assert "from vercor._runtime.component_topology import" not in initialization_source


@pytest.mark.fast_always
def test_runtime_topology_state_groups_read_only_maps() -> None:
    topology_state_module = importlib.import_module("vercor._runtime.topology_state")
    topology_state_source = source_for("vercor/_runtime/topology_state.py")
    topology_source = source_for("vercor/_runtime/topology.py")

    assert hasattr(topology_state_module, "RuntimeTopologyMaps")
    RuntimeTopologyMaps = topology_state_module.RuntimeTopologyMaps
    assert is_dataclass(RuntimeTopologyMaps)
    assert getattr(RuntimeTopologyMaps, "__dataclass_params__").frozen is True
    assert hasattr(RuntimeTopologyMaps, "__slots__")
    assert [field.name for field in fields(RuntimeTopologyMaps)] == [
        "regridders",
        "binary_masks",
        "fractional_masks",
    ]
    assert "class RuntimeTopologyMaps" in topology_state_source
    assert "topology_maps: RuntimeTopologyMaps" in topology_state_source
    assert "class RuntimeTopologyMaps" not in topology_source
    assert "MappingProxyType" in topology_state_source
    assert not Path("vercor/_runtime/resources.py").exists()


@pytest.mark.fast_always
def test_runtime_topology_policy_boundaries_are_focused() -> None:
    topology_state_module = importlib.import_module("vercor._runtime.topology_state")
    topology_state_source = source_for("vercor/_runtime/topology_state.py")
    exchange_topology_source = source_for("vercor/_runtime/exchange_topology.py")
    surface_masks_source = source_for("vercor/_runtime/surface_masks.py")
    topology_policy_source = source_for("vercor/_runtime/topology_policy.py")
    topology_source = source_for("vercor/_runtime/topology.py")
    ExchangeTopologyState = topology_state_module.ExchangeTopologyState
    assert is_dataclass(ExchangeTopologyState)
    assert [field.name for field in fields(ExchangeTopologyState)] == [
        "topology_maps",
    ]
    assert not hasattr(topology_state_module, "SurfaceExchangeMasks")

    assert "RuntimeRegridder =" not in topology_state_source
    assert "BilinearRectilinearRegridder" not in topology_state_source
    assert "ConservativeRectilinearRegridder" not in topology_state_source
    assert "def build_exchange_topology_maps(" in exchange_topology_source
    assert "def create_surface_exchange_masks(" in surface_masks_source
    assert "def validate_land_mask_consistency(" in surface_masks_source
    assert "def apply_surface_exchange_masks(" not in surface_masks_source
    assert "def apply_topology_policy(" in topology_policy_source
    assert "def build_topology_context(" in topology_policy_source

    for marker in (
        "compute_ocn_lnd_masks_on_atm_grid",
        "check_remap_conservation",
        "check_total_lnd_ocn_mask_sum",
        "ConservativeRectilinearRegridder",
        "jax_ones",
        "def create_exchange_masks(",
        "def validate_land_mask_consistency(",
        "def initialize_regridders_and_masks(",
        "def patch_exchange_masks(",
    ):
        assert marker not in topology_source

    assert "import vercor._runtime.exchange_topology as" in topology_source
    assert "import vercor._runtime.topology_policy as" in topology_source
    assert "import vercor._runtime.surface_masks as" not in topology_source
    assert "from vercor._runtime.topology_state import" in topology_source
    assert "isinstance(policy, SurfaceMaskPolicy)" not in topology_policy_source


@pytest.mark.fast_always
def test_prepared_boundary_replaces_mutable_runtime_resources() -> None:
    prepared_source = source_for("vercor/_runtime/prepared.py")
    facade_source = source_for("vercor/_runtime/facade.py")
    preparation_source = source_for("vercor/_runtime/preparation.py")
    run_context_source = source_for("vercor/_runtime/run_context.py")

    assert not Path("vercor/_runtime/resources.py").exists()
    assert "class PreparedCoupling" in prepared_source
    assert "@dataclass(frozen=True)" in prepared_source
    assert "build_exchange_contracts(" not in preparation_source
    assert "build_runtime_dispatch_context(" not in preparation_source
    assert "build_runtime_dispatch_context(" in prepared_source
    assert "MutableMapping" not in run_context_source
    assert "runtime_cache" not in run_context_source
    assert "CompiledRuntimeCache" not in run_context_source
    for source in (facade_source, preparation_source):
        assert "runtime_resources" not in source
        assert "RuntimeInputs" not in source


@pytest.mark.fast_always
def test_runtime_compilation_cache_is_removed() -> None:
    compilation_path = Path("vercor/_runtime/compilation.py")
    cache_path = Path("vercor/_runtime/cache.py")
    assert not compilation_path.exists()
    assert not cache_path.exists()

    run_context_source = source_for("vercor/_runtime/run_context.py")
    prepared_source = source_for("vercor/_runtime/prepared.py")

    assert "from vercor._runtime.compilation import" not in run_context_source
    assert "from vercor._runtime.compilation import" not in prepared_source
    assert "from vercor._runtime.cache import" not in run_context_source
    assert "from vercor._runtime.cache import" not in prepared_source
    assert "CompiledRuntime" not in run_context_source
    assert "CompiledRuntimeCache" not in prepared_source
    assert "compiled_runtime_cache_key(" not in run_context_source


@pytest.mark.fast_always
def test_runtime_state_validation_module_owns_runtime_topology_validation() -> None:
    state_validation_path = Path("vercor/_runtime/state_validation.py")
    coupler_state_source = source_for("vercor/_runtime/coupler_state.py")
    preparation_source = source_for("vercor/_runtime/preparation.py")
    facade_source = source_for("vercor/_runtime/facade.py")
    coupler_source = source_for("vercor/coupler.py")

    assert state_validation_path.exists()
    state_validation_source = state_validation_path.read_text(encoding="utf-8")
    assert "def validate_runtime_state(" in state_validation_source
    assert "def validate_runtime_state(" not in coupler_state_source
    assert "from vercor._runtime.state_validation import" in preparation_source
    assert "from vercor._runtime.state_validation import" not in facade_source
    assert "from vercor._runtime.state_validation import" not in coupler_source


def test_runtime_facade_reexports_preparation_without_owning_it() -> None:
    facade_source = source_for("vercor/_runtime/facade.py")
    preparation_source = source_for("vercor/_runtime/preparation.py")

    assert "from vercor._runtime.preparation import" in facade_source
    assert "PreparedRuntimeState" not in facade_source
    assert "PreparedRuntimeState" not in preparation_source
    assert "Protocol" not in preparation_source
    assert "RuntimePreparationInputs" not in preparation_source
    assert "RuntimeInputs" not in preparation_source
    assert "from vercor._runtime.prepared import PreparedCoupling" in preparation_source
    assert "def runtime_state_from_components(" not in facade_source
    assert "def validate_runtime_state(" not in facade_source
    assert "def create_runtime_state(" not in facade_source
    assert "def prepare_runtime_state(" not in facade_source


@pytest.mark.fast_always
def test_runtime_runner_splits_path_selection_helpers() -> None:
    runner_source = source_for("vercor/_runtime/runner.py")
    run_coupler_body = runner_source.split("def run_coupler_runtime(", 1)[1].split(
        "\ndef _run_compiled_scanned_runtime(",
        1,
    )[0]

    assert "def _run_compiled_scanned_runtime(" in runner_source
    assert "def _raise_if_donating_host_runtime(" not in runner_source
    assert "compiled_runtime_cache_key(" not in run_coupler_body
    assert "def compiled_runtime_cache_key(" not in runner_source
    assert "compiled_scanned_runtime," not in runner_source
    assert "return compiled_scanned_runtime(" not in runner_source
    assert "get_or_compile_for_context(" not in runner_source
    assert "context.compiled_runtime_cache_key(" not in runner_source
    assert "get_or_compile(" not in runner_source
    assert "donate_state" not in runner_source
    assert "raise CouplerError(" not in run_coupler_body


def test_runtime_package_has_no_top_level_import_cycles() -> None:
    assert package_import_cycles("vercor/_runtime", "vercor._runtime") == []
