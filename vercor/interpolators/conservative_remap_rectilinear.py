from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
import scipy


class ConservativeRectilinearRemapper:
    """
    First-order locally conservative area-average remapping.
    Handles arbitrary rectilinear grids, periodicity, and conservation.
    """

    def __init__(
        self,
        src_lon_bnds: NDArray,
        src_lat_bnds: NDArray,
        dst_lon_bnds: NDArray,
        dst_lat_bnds: NDArray,
        src_mask: Optional[NDArray] = None,
        normalize: str = "conservation",
        radius: float = 6371.0,
    ):
        """
        Initialize and precompute remapping weights.

        Arguments:
            src_lon_bnds (1D array): Source longitude cell boundaries (Monotonic).
            src_lat_bnds (1D array): Source latitude cell boundaries.
            dst_lon_bnds (1D array): Target longitude cell boundaries (Monotonic).
            dst_lat_bnds (1D array): Target latitude cell boundaries.
            source_mask (2D array, optional): Boolean mask for source grid (True=Invalid).
            normalize (str):
                'conservation': Normalize by total area of target cell. (Mass Preserving)
                'fracarea'    : Normalize by intersection area. (Value Preserving / Extrapolation)
            radius (float): Radius of the sphere (km).
        """

        self.radius = radius
        self.normalize = normalize

        # 1. Standardize and Store Bounds
        self.src_lon_b = np.asarray(src_lon_bnds, dtype=np.float64)
        self.src_lat_b, self._s_lat_flip = self._standardize_lat(src_lat_bnds)

        self.dst_lon_b = np.asarray(dst_lon_bnds, dtype=np.float64)
        self.dst_lat_b, self._d_lat_flip = self._standardize_lat(dst_lat_bnds)

        self.n_src_lon = len(self.src_lon_b) - 1
        self.n_src_lat = len(self.src_lat_b) - 1
        self.n_dst_lon = len(self.dst_lon_b) - 1
        self.n_dst_lat = len(self.dst_lat_b) - 1

        # 2. Compute Overlaps
        W_lon = self._compute_lon_overlaps(self.src_lon_b, self.dst_lon_b)

        # Latitude Overlaps (Sin space)
        s_sin_lat = np.sin(np.deg2rad(self.src_lat_b))
        d_sin_lat = np.sin(np.deg2rad(self.dst_lat_b))

        # Fix potential float noise at poles
        s_sin_lat = np.round(s_sin_lat, 14)
        d_sin_lat = np.round(d_sin_lat, 14)

        W_lat = self._compute_interval_overlaps(s_sin_lat, d_sin_lat)

        # 3. Combine Weights
        self.weights: scipy.sparse._csr.csr_matrix = sparse.kron(
            W_lat, W_lon, format="csr"
        )

        # 4. Masking
        if src_mask is not None:
            if self._s_lat_flip:
                src_mask = src_mask[::-1, :]
            flat_mask = src_mask.flatten()
            valid_diag = sparse.diags((~flat_mask).astype(int))
            self.weights = self.weights @ valid_diag
            self.weights.eliminate_zeros()

        # 5. Destination Areas
        # Calculate geometric area based on provided bounds (even if they overhang)
        dst_lon_diff = np.abs(np.diff(np.deg2rad(self.dst_lon_b)))
        dst_lat_diff = np.abs(np.diff(d_sin_lat))

        self.dst_areas = (
            self.radius**2 * np.outer(dst_lat_diff, dst_lon_diff)
        ).flatten()
        self.dst_areas[self.dst_areas <= 1e-15] = np.inf

        self.weights *= self.radius**2

    def _standardize_lat(self, bounds: NDArray) -> tuple[NDArray, bool]:
        """Ensure latitude bounds are monotonically increasing."""

        b = np.array(bounds, dtype=np.float64)
        is_flipped = False

        if b[0] > b[-1]:
            b = b[::-1]
            is_flipped = True

        return b, is_flipped

    def _compute_interval_overlaps(
        self, src_bnds: NDArray, dst_bnds: NDArray
    ) -> sparse.csr_matrix:
        """1D Overlap calculation (Latitude)."""
        n_src = len(src_bnds) - 1
        n_dst = len(dst_bnds) - 1
        row_ind, col_ind, data = [], [], []

        for i in range(n_dst):
            d1, d2 = dst_bnds[i], dst_bnds[i + 1]
            if abs(d2 - d1) < 1e-15:
                continue

            idx_start = np.searchsorted(src_bnds, d1, side="left") - 1
            idx_start = max(0, idx_start)
            idx_end = np.searchsorted(src_bnds, d2, side="right")
            idx_end = min(n_src, idx_end)

            for j in range(idx_start, idx_end):
                s1, s2 = src_bnds[j], src_bnds[j + 1]
                overlap = max(0.0, min(d2, s2) - max(d1, s1))
                if overlap > 1e-15:
                    row_ind.append(i)
                    col_ind.append(j)
                    data.append(overlap)

        return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n_dst, n_src))

    def _compute_lon_overlaps(
        self, src_bnds: NDArray, dst_bnds: NDArray
    ) -> sparse.csr_matrix:
        """1D Longitude overlap with periodicity check."""
        n_src = len(src_bnds) - 1
        n_dst = len(dst_bnds) - 1
        row_ind, col_ind, data = [], [], []

        for i in range(n_dst):
            d1, d2 = dst_bnds[i], dst_bnds[i + 1]
            if d1 > d2:
                d1, d2 = d2, d1

            # Check overlap against Source, Source-360, Source+360
            # This handles destination cells that are defined outside 0..360 (e.g. -182.5)
            # but physically map to the source domain.
            shifts = [0.0, 360.0, -360.0]

            for shift in shifts:
                t_s, t_e = d1 + shift, d2 + shift

                # Bounding box check optimization
                if t_e <= src_bnds[0] or t_s >= src_bnds[-1]:
                    continue

                idx_start = np.searchsorted(src_bnds, t_s, side="left") - 1
                idx_start = max(0, idx_start)
                idx_end = np.searchsorted(src_bnds, t_e, side="right")
                idx_end = min(n_src, idx_end)

                for j in range(idx_start, idx_end):
                    s1, s2 = src_bnds[j], src_bnds[j + 1]
                    overlap = max(0.0, min(t_e, s2) - max(t_s, s1))

                    if overlap > 1e-15:
                        row_ind.append(i)
                        col_ind.append(j)
                        data.append(overlap)

        W = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n_dst, n_src))
        W.sum_duplicates()
        W.data = np.deg2rad(W.data)

        return W

    def get_src_areas(self) -> NDArray:
        """Returns the exact source cell areas (useful for mass verification)."""
        dlon = np.diff(np.deg2rad(self.src_lon_b))
        s_sin_lat = np.sin(np.deg2rad(self.src_lat_b))
        dsinlat = np.abs(np.diff(s_sin_lat))
        areas = self.radius**2 * np.outer(dsinlat, dlon)

        if self._s_lat_flip:
            areas = areas[::-1, :]

        return areas

    def apply_scalar(self, field: NDArray) -> NDArray:
        """Apply conservative remapping to a scalar field."""

        if field.shape != (self.n_src_lat, self.n_src_lon):
            raise ValueError(
                f"Shape mismatch: {field.shape} vs grid ({self.n_src_lat}, {self.n_src_lon})"
            )

        if self._s_lat_flip:
            field = field[::-1, :]
        flat_field = field.flatten()
        nan_mask = np.isnan(flat_field)

        if np.any(nan_mask):
            data_clean = np.where(nan_mask, 0.0, flat_field)
            weighted_sum = self.weights @ data_clean
            if self.normalize == "fracarea":
                validity_vec = (~nan_mask).astype(float)
                norm = self.weights @ validity_vec
            else:
                norm = self.dst_areas
        else:
            weighted_sum = self.weights @ flat_field
            if self.normalize == "fracarea":
                norm = np.array(self.weights.sum(axis=1)).flatten()
            else:
                norm = self.dst_areas

        result: NDArray = np.zeros_like(weighted_sum)
        valid = norm > 1e-15
        result[valid] = weighted_sum[valid] / norm[valid]
        result[~valid] = np.nan

        result_grid: NDArray = result.reshape((self.n_dst_lat, self.n_dst_lon))

        if self._d_lat_flip:
            result_grid = result_grid[::-1, :]

        return result_grid

    def apply_vector(self, u_src: NDArray, v_src: NDArray) -> tuple[NDArray, NDArray]:
        # To satisfy the interface
        # Avoid errors from MyPy static type checking
        raise RuntimeError("Conservative remapping for vectors not implemented.")

    def get_src_total_mass(self, field_on_src: NDArray) -> float:
        """Calculate total mass on source grid given field values."""

        src_areas: NDArray = self.get_src_areas()
        result: float = np.nansum(field_on_src * src_areas)
        return result

    def get_dst_total_mass(self, field_on_dst: NDArray) -> float:
        """Calculate total mass on destination grid given field values."""

        clean_areas: NDArray = np.where(np.isinf(self.dst_areas), 0.0, self.dst_areas)
        result: float = np.nansum(field_on_dst.flatten() * clean_areas)
        return result
