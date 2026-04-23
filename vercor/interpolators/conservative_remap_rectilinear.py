from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


@jax.tree_util.register_pytree_node_class
class ConservativeRectilinearRemapper:
    """
    First-order locally conservative area-average remapping.
    Handles arbitrary rectilinear grids, periodicity, and conservation.
    """

    def __init__(
        self,
        src_lon_edges: NDArray,
        src_lat_edges: NDArray,
        dst_lon_edges: NDArray,
        dst_lat_edges: NDArray,
        src_mask: Optional[NDArray] = None,
        normalize: str = "conservation",
        radius: float = 6371.0,
    ):
        """
        Initialize and precompute remapping weights.

        Arguments:
            src_lon_edges (1D array): Source longitude cell edges (Monotonic).
            src_lat_edges (1D array): Source latitude cell edges.
            dst_lon_edges (1D array): Target longitude cell edges (Monotonic).
            dst_lat_edges (1D array): Target latitude cell edges.
            source_mask (2D array, optional): Boolean mask for source grid (True=Invalid).
            normalize (str):
                'conservation': Normalize by total area of target cell. (Mass Preserving)
                'fracarea'    : Normalize by intersection area. (Value Preserving / Extrapolation)
            radius (float): Radius of the sphere (km).
        """

        if normalize not in {"conservation", "fracarea"}:
            raise ValueError(
                "normalize must be either 'conservation' or 'fracarea', "
                f"got {normalize!r}"
            )

        self.radius = float(radius)
        self.normalize = normalize
        self._normalize_fracarea = normalize == "fracarea"

        # 1. Standardize and store bounds
        src_lon_b_np = np.asarray(src_lon_edges, dtype=np.float64)
        src_lat_b_np, self._s_lat_flip = self._standardize_lat(src_lat_edges)
        dst_lon_b_np = np.asarray(dst_lon_edges, dtype=np.float64)
        dst_lat_b_np, self._d_lat_flip = self._standardize_lat(dst_lat_edges)

        self.src_lon_b = jnp.asarray(src_lon_b_np, dtype=jnp.float64)
        self.src_lat_b = jnp.asarray(src_lat_b_np, dtype=jnp.float64)
        self.dst_lon_b = jnp.asarray(dst_lon_b_np, dtype=jnp.float64)
        self.dst_lat_b = jnp.asarray(dst_lat_b_np, dtype=jnp.float64)

        self.n_src_lon = len(src_lon_b_np) - 1
        self.n_src_lat = len(src_lat_b_np) - 1
        self.n_dst_lon = len(dst_lon_b_np) - 1
        self.n_dst_lat = len(dst_lat_b_np) - 1
        self._n_src_cells = self.n_src_lat * self.n_src_lon
        self._n_dst_cells = self.n_dst_lat * self.n_dst_lon

        # 2. Compute 1D overlaps
        lon_dst_idx, lon_src_idx, lon_overlap = self._compute_lon_overlaps(
            src_lon_b_np, dst_lon_b_np
        )

        src_sin_lat = np.round(np.sin(np.deg2rad(src_lat_b_np)), 14)
        dst_sin_lat = np.round(np.sin(np.deg2rad(dst_lat_b_np)), 14)
        lat_dst_idx, lat_src_idx, lat_overlap = self._compute_interval_overlaps(
            src_sin_lat, dst_sin_lat
        )

        # 3. Combine 1D overlaps into 2D remapping triplets
        dst_indices_np = (
            lat_dst_idx[:, None] * self.n_dst_lon + lon_dst_idx[None, :]
        ).reshape(-1)
        src_indices_np = (
            lat_src_idx[:, None] * self.n_src_lon + lon_src_idx[None, :]
        ).reshape(-1)
        overlap_weights_np = (
            (self.radius**2) * (lat_overlap[:, None] * lon_overlap[None, :])
        ).reshape(-1)

        # 4. Apply source mask eagerly by dropping invalid source entries
        if src_mask is not None:
            src_mask_np = np.asarray(src_mask, dtype=bool)
            if self._s_lat_flip:
                src_mask_np = src_mask_np[::-1, :]
            valid_src = (~src_mask_np).reshape(-1)
            keep = valid_src[src_indices_np]
            dst_indices_np = dst_indices_np[keep]
            src_indices_np = src_indices_np[keep]
            overlap_weights_np = overlap_weights_np[keep]

        self.dst_indices = jnp.asarray(dst_indices_np, dtype=jnp.int32)
        self.src_indices = jnp.asarray(src_indices_np, dtype=jnp.int32)
        self.overlap_weights = jnp.asarray(overlap_weights_np, dtype=jnp.float64)

        dst_lon_diff = np.abs(np.diff(np.deg2rad(dst_lon_b_np)))
        dst_lat_diff = np.abs(np.diff(dst_sin_lat))
        dst_areas_np = (
            (self.radius**2) * np.outer(dst_lat_diff, dst_lon_diff)
        ).reshape(-1)
        dst_areas_np[dst_areas_np <= 1e-15] = np.inf
        self.dst_areas = jnp.asarray(dst_areas_np, dtype=jnp.float64)
        self.fracarea_norm = self._segment_sum(
            self.overlap_weights,
            self.dst_indices,
            self._n_dst_cells,
        )

    @staticmethod
    def _segment_sum(values: jax.Array, indices: jax.Array, size: int) -> jax.Array:
        return jnp.zeros((size,), dtype=jnp.float64).at[indices].add(values)

    @staticmethod
    def _standardize_lat(bounds: NDArray) -> tuple[NDArray, bool]:
        """Ensure latitude bounds are monotonically increasing."""

        b = np.array(bounds, dtype=np.float64)
        is_flipped = False

        if b[0] > b[-1]:
            b = b[::-1]
            is_flipped = True

        return b, is_flipped

    @staticmethod
    def _merge_duplicate_entries(
        row_ind: list[int], col_ind: list[int], data: list[float]
    ) -> tuple[NDArray, NDArray, NDArray]:
        if not data:
            empty_i = np.asarray([], dtype=np.int64)
            empty_f = np.asarray([], dtype=np.float64)
            return empty_i, empty_i, empty_f

        rows = np.asarray(row_ind, dtype=np.int64)
        cols = np.asarray(col_ind, dtype=np.int64)
        values = np.asarray(data, dtype=np.float64)
        keys = np.stack((rows, cols), axis=1)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        merged = np.zeros(unique_keys.shape[0], dtype=np.float64)
        np.add.at(merged, inverse, values)
        return unique_keys[:, 0], unique_keys[:, 1], merged

    def _compute_interval_overlaps(
        self, src_edges: NDArray, dst_edges: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        """1D overlap calculation (latitude in sin-space)."""

        n_src = len(src_edges) - 1
        n_dst = len(dst_edges) - 1
        row_ind: list[int] = []
        col_ind: list[int] = []
        data: list[float] = []

        for i in range(n_dst):
            d1, d2 = dst_edges[i], dst_edges[i + 1]
            if abs(d2 - d1) < 1e-15:
                continue

            idx_start = np.searchsorted(src_edges, d1, side="left") - 1
            idx_start = max(0, idx_start)
            idx_end = np.searchsorted(src_edges, d2, side="right")
            idx_end = min(n_src, idx_end)

            for j in range(idx_start, idx_end):
                s1, s2 = src_edges[j], src_edges[j + 1]
                overlap = max(0.0, min(d2, s2) - max(d1, s1))
                if overlap > 1e-15:
                    row_ind.append(i)
                    col_ind.append(j)
                    data.append(overlap)

        return self._merge_duplicate_entries(row_ind, col_ind, data)

    def _compute_lon_overlaps(
        self, src_edges: NDArray, dst_edges: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        """1D longitude overlap with periodicity check."""

        n_src = len(src_edges) - 1
        n_dst = len(dst_edges) - 1
        row_ind: list[int] = []
        col_ind: list[int] = []
        data: list[float] = []

        for i in range(n_dst):
            d1, d2 = dst_edges[i], dst_edges[i + 1]
            if d1 > d2:
                d1, d2 = d2, d1

            for shift in (0.0, 360.0, -360.0):
                t_s, t_e = d1 + shift, d2 + shift

                if t_e <= src_edges[0] or t_s >= src_edges[-1]:
                    continue

                idx_start = np.searchsorted(src_edges, t_s, side="left") - 1
                idx_start = max(0, idx_start)
                idx_end = np.searchsorted(src_edges, t_e, side="right")
                idx_end = min(n_src, idx_end)

                for j in range(idx_start, idx_end):
                    s1, s2 = src_edges[j], src_edges[j + 1]
                    overlap = max(0.0, min(t_e, s2) - max(t_s, s1))

                    if overlap > 1e-15:
                        row_ind.append(i)
                        col_ind.append(j)
                        data.append(overlap)

        rows, cols, values = self._merge_duplicate_entries(row_ind, col_ind, data)
        return rows, cols, np.deg2rad(values)

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[jax.Array, ...], tuple[float, str, bool, bool, int, int, int, int]
    ]:
        children = (
            self.src_lon_b,
            self.src_lat_b,
            self.dst_lon_b,
            self.dst_lat_b,
            self.dst_areas,
            self.dst_indices,
            self.src_indices,
            self.overlap_weights,
            self.fracarea_norm,
        )
        aux_data = (
            self.radius,
            self.normalize,
            self._s_lat_flip,
            self._d_lat_flip,
            self.n_src_lon,
            self.n_src_lat,
            self.n_dst_lon,
            self.n_dst_lat,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[float, str, bool, bool, int, int, int, int],
        children: tuple[jax.Array, ...],
    ) -> "ConservativeRectilinearRemapper":
        (
            radius,
            normalize,
            s_lat_flip,
            d_lat_flip,
            n_src_lon,
            n_src_lat,
            n_dst_lon,
            n_dst_lat,
        ) = aux_data
        (
            src_lon_b,
            src_lat_b,
            dst_lon_b,
            dst_lat_b,
            dst_areas,
            dst_indices,
            src_indices,
            overlap_weights,
            fracarea_norm,
        ) = children

        obj = object.__new__(cls)
        obj.radius = radius
        obj.normalize = normalize
        obj._normalize_fracarea = normalize == "fracarea"
        obj._s_lat_flip = s_lat_flip
        obj._d_lat_flip = d_lat_flip
        obj.n_src_lon = n_src_lon
        obj.n_src_lat = n_src_lat
        obj.n_dst_lon = n_dst_lon
        obj.n_dst_lat = n_dst_lat
        obj._n_src_cells = n_src_lon * n_src_lat
        obj._n_dst_cells = n_dst_lon * n_dst_lat
        obj.src_lon_b = src_lon_b
        obj.src_lat_b = src_lat_b
        obj.dst_lon_b = dst_lon_b
        obj.dst_lat_b = dst_lat_b
        obj.dst_areas = dst_areas
        obj.dst_indices = dst_indices
        obj.src_indices = src_indices
        obj.overlap_weights = overlap_weights
        obj.fracarea_norm = fracarea_norm
        return obj

    def get_src_areas(self) -> jax.Array:
        """Returns the exact source cell areas (useful for mass verification)."""

        dlon = jnp.diff(jnp.deg2rad(self.src_lon_b))
        sin_lat = jnp.sin(jnp.deg2rad(self.src_lat_b))
        dsinlat = jnp.abs(jnp.diff(sin_lat))
        areas = (self.radius**2) * dsinlat[:, None] * dlon[None, :]

        if self._s_lat_flip:
            areas = areas[::-1, :]

        return areas

    def apply_scalar(self, field: Any) -> jax.Array:
        """Apply conservative remapping to a scalar field."""

        expected_shape = (self.n_src_lat, self.n_src_lon)
        if np.shape(field) != expected_shape:
            raise ValueError(
                f"Shape mismatch: {np.shape(field)} vs grid {expected_shape}"
            )

        field_array = jnp.asarray(field, dtype=jnp.float64)
        if self._s_lat_flip:
            field_array = field_array[::-1, :]

        flat_field = field_array.reshape(-1)
        clean_field = jnp.where(jnp.isnan(flat_field), 0.0, flat_field)
        weighted_values = self.overlap_weights * clean_field[self.src_indices]
        weighted_sum = self._segment_sum(
            weighted_values,
            self.dst_indices,
            self._n_dst_cells,
        )

        if self._normalize_fracarea:
            valid = jnp.where(jnp.isnan(flat_field), 0.0, 1.0)
            norm = self._segment_sum(
                self.overlap_weights * valid[self.src_indices],
                self.dst_indices,
                self._n_dst_cells,
            )
        else:
            norm = self.dst_areas

        result = jnp.where(norm > 1e-15, weighted_sum / norm, jnp.nan)
        result_grid = result.reshape((self.n_dst_lat, self.n_dst_lon))

        if self._d_lat_flip:
            result_grid = result_grid[::-1, :]

        return result_grid

    def apply_vector(self, u_src: Any, v_src: Any) -> tuple[Any, Any]:
        # To satisfy the interface
        # Avoid errors from MyPy static type checking
        raise RuntimeError("Conservative remapping for vectors not implemented.")

    def get_src_total_mass(self, field_on_src: Any) -> float:
        """Calculate total mass on source grid given field values."""

        result = jnp.nansum(
            jnp.asarray(field_on_src, dtype=jnp.float64) * self.get_src_areas()
        )
        return float(result)

    def get_dst_total_mass(self, field_on_dst: Any) -> float:
        """Calculate total mass on destination grid given field values."""

        clean_areas = jnp.where(jnp.isinf(self.dst_areas), 0.0, self.dst_areas)
        result = jnp.nansum(
            jnp.asarray(field_on_dst, dtype=jnp.float64).reshape(-1) * clean_areas
        )
        return float(result)
