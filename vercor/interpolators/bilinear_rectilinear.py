from typing import Any, Tuple
import numpy as np


def _wrap_like(lon_deg: np.ndarray, base0_deg: float) -> np.ndarray:
    """
    Map longitudes (deg) into the [base0, base0+360) interval.
    """
    return base0_deg + np.mod(lon_deg - base0_deg, 360.0)


def _unit_east_north(
    lon_rad: np.ndarray, lat_rad: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return unit vectors (east, north) in 3-D for given lon/lat (radians).
    east = d r / d lon normalized; north = d r / d lat normalized.
    east is independent of latitude on the unit sphere.
    Shapes follow broadcasting of lon_rad/lat_rad.
    """
    slon, clon = np.sin(lon_rad), np.cos(lon_rad)
    slat, clat = np.sin(lat_rad), np.cos(lat_rad)

    # east: (-sin lon, cos lon, 0)
    e_east = np.stack((-slon, clon, np.zeros_like(lon_rad)), axis=-1)

    # north: (-sin lat cos lon, -sin lat sin lon, cos lat)
    e_north = np.stack((-slat * clon, -slat * slon, clat), axis=-1)
    return e_east, e_north


def _great_circle_distance_rad(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> Any:
    """
    Great-circle distance (radians) between points (supports broadcasting).
    Haversine, numerically stable.
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    sdlat2 = np.sin(dlat * 0.5)
    sdlon2 = np.sin(dlon * 0.5)
    a = sdlat2 * sdlat2 + np.cos(lat1) * np.cos(lat2) * sdlon2 * sdlon2

    # Clamp for safety
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


class Bilinear:
    """
    Bilinear interpolator for rectilinear lat/lon grids with:

      - periodic longitude handling
      - ascending or descending latitude
      - non-uniform spacing
      - optional NaN-aware renormalization
      - support for scalar and vector (u,v) fields
      - source/target masks
      - nearest / IDW extrapolation for masked-out areas

    Coordinates:
      - lon_src: shape (nlon,), degrees, strictly monotonic ascending (wrap allowed across dateline)
      - lat_src: shape (nlat,), degrees, strictly monotonic (ascending or descending)
      - lon_tgt, lat_tgt: target coordinates, any shape (broadcastable to the same)

    Masks:
      - src_mask: boolean array (nlat, nlon) where True means valid; if None, all valid
      - tgt_mask: boolean array like lon_tgt/lat_tgt shape; True means compute/keep output, False -> fill_value

    Vector interpolation:
      - (u,v) are eastward and northward components on the sphere (m/s etc.)
      - Rotate vectors properly by projecting each corner vector to 3-D (using local EN basis),
        bilinear-blending in 3-D, then projecting onto the target EN basis.

    Extrapolation:
      - If all four bilinear corners are invalid, we extrapolate from sources:
         - mode='nearest': nearest valid source point
         - mode='idw': inverse-distance weighted blend of k nearest valid sources
    """

    def __init__(
        self,
        field_in: dict[str, np.ndarray],
        field_out: dict[str, np.ndarray],
        periodic_longitude: bool = True,
        nan_renorm: bool = True,
        extrapolation_mode: str | None = "idw",  # 'nearest' | 'idw'
        idw_k: int = 8,
        idw_eps: float = 1e-12,
        fill_value=np.nan,
    ):

        # Store config
        self.periodic = bool(periodic_longitude)
        self.nan_renorm = bool(nan_renorm)
        self.extrapolation_mode = extrapolation_mode
        self.idw_k = int(idw_k)
        self.idw_eps = float(idw_eps)
        self.fill_value = fill_value

        # Source grid (1D)
        lon_src_deg = np.asarray(field_in["lon"], dtype=float)
        lat_src_deg = np.asarray(field_in["lat"], dtype=float)
        assert (
            lon_src_deg.ndim == 1 and lat_src_deg.ndim == 1
        ), "lon_src, lat_src must be 1-D"
        self.nlon = lon_src_deg.size
        self.nlat = lat_src_deg.size

        # Ensure monotonicity assumptions
        if not (np.all(np.diff(lon_src_deg) > 0) or np.all(np.diff(lon_src_deg) < 0)):
            raise ValueError(
                "lon_src must be strictly monotonic (ascending or descending)."
            )
        if np.all(np.diff(lon_src_deg) < 0):
            # Prefer ascending longitudes internally
            lon_src_deg = lon_src_deg[::-1]
            self._lon_flipped = True
        else:
            self._lon_flipped = False

        # For latitude we support ascending or descending; track and keep original order
        self.lat_ascending = np.all(np.diff(lat_src_deg) > 0)
        self.lat_descending = np.all(np.diff(lat_src_deg) < 0)
        if not (self.lat_ascending or self.lat_descending):
            raise ValueError(
                "lat_src must be strictly monotonic (ascending or descending)."
            )

        self.lon_src_deg = lon_src_deg
        self.lat_src_deg = lat_src_deg
        self.lon_src_rad = np.deg2rad(lon_src_deg)
        self.lat_src_rad = np.deg2rad(lat_src_deg)

        # Target grid (any shape)
        lon_tgt = np.asarray(field_out["lon"], dtype=float)
        lat_tgt = np.asarray(field_out["lat"], dtype=float)
        lon_tgt_deg, lat_tgt_deg = np.meshgrid(lon_tgt, lat_tgt)
        tgt_mask = field_out.get("mask", None)

        self.tshape = np.broadcast_shapes(lon_tgt_deg.shape, lat_tgt_deg.shape)
        self.lon_tgt_deg = np.broadcast_to(lon_tgt_deg, self.tshape).copy()
        self.lat_tgt_deg = np.broadcast_to(lat_tgt_deg, self.tshape).copy()
        self.lon_tgt_rad = np.deg2rad(self.lon_tgt_deg)
        self.lat_tgt_rad = np.deg2rad(self.lat_tgt_deg)

        # Target mask
        if tgt_mask is None:
            self.tgt_mask = np.ones(self.tshape, dtype=bool)
        else:
            self.tgt_mask = np.broadcast_to(
                np.asarray(tgt_mask, dtype=bool), self.tshape
            )

        # Precompute bilinear cell indices and weights for all target points
        self._precompute_cells_and_weights()

        # Precompute target EN bases (for vector projection back)
        self._e_east_t, self._e_north_t = _unit_east_north(
            self.lon_tgt_rad, self.lat_tgt_rad
        )

        # Precompute source EN bases (for vector projection from source)
        # Shapes: (nlat, nlon, 3)
        lon_src_2d, lat_src_2d = np.meshgrid(self.lon_src_rad, self.lat_src_rad)
        self._e_east_src, self._e_north_src = _unit_east_north(lon_src_2d, lat_src_2d)

        # Precompute source positions for extrapolation (geometry only)
        self._lon_src_2d = lon_src_2d
        self._lat_src_2d = lat_src_2d

    # --------------------------- Precomputation --------------------------- #

    def _precompute_cells_and_weights(self):
        """
        For each target point, compute (i0,i1,j0,j1) and (fx,fy) and corner weights.
        """
        nlon, nlat = self.nlon, self.nlat
        lon_src = self.lon_src_deg
        lat_src = self.lat_src_deg

        # Map target lons into the source range for searchsorted
        if self.periodic:
            base0 = lon_src[0]
            lon_tgt_mapped = _wrap_like(self.lon_tgt_deg, base0)
        else:
            lon_tgt_mapped = np.clip(self.lon_tgt_deg, lon_src.min(), lon_src.max())

        # Longitudinal indices (assume lon_src ascending internally)
        i1 = np.searchsorted(lon_src, lon_tgt_mapped, side="right")
        i0 = i1 - 1
        # clamp for non-periodic
        if self.periodic:
            i0 = np.mod(i0, nlon)
            i1 = np.mod(i1, nlon)
        else:
            i0 = np.clip(i0, 0, nlon - 2)
            i1 = i0 + 1

        # Latitudinal indices (support ascending or descending)
        if self.lat_ascending:
            j1 = np.searchsorted(lat_src, self.lat_tgt_deg, side="right")
            j0 = j1 - 1
        else:  # descending
            # invert for search, then map back
            lat_inv = lat_src[::-1]
            j1_inv = np.searchsorted(lat_inv, self.lat_tgt_deg, side="right")
            j0_inv = j1_inv - 1
            # indices in inverted array -> original
            j0 = (nlat - 1) - np.clip(j0_inv, 0, nlat - 2) - 1
            j1 = j0 + 1

        # Clamp to valid interior rows
        j0 = np.clip(j0, 0, nlat - 2)
        j1 = j0 + 1

        # Fractions fx, fy (0..1), careful with wrap row in longitude
        lon0 = self.lon_src_rad[i0]
        lon1 = self.lon_src_rad[i1]
        lam_t = self.lon_tgt_rad

        # Forward difference from i0 to i1 along ascending axis (handle wrap)
        dlon = lon1 - lon0
        wrap = i1 <= i0  # wrapped cell
        dlon = np.where(wrap, (lon1 + 2 * np.pi) - lon0, dlon)

        dlam = lam_t - lon0
        dlam = np.where(
            dlam < 0, dlam + 2 * np.pi, dlam
        )  # ensure forward within [0, 2π)

        fx = np.where(dlon != 0.0, dlam / dlon, 0.0)
        fx = np.clip(fx, 0.0, 1.0)

        # Latitude fraction
        lat0 = self.lat_src_rad[j0]
        lat1 = self.lat_src_rad[j1]
        dphi = lat1 - lat0
        # avoid division by zero if degenerate (shouldn't happen in well-formed grids)
        fy = np.where(dphi != 0.0, (self.lat_tgt_rad - lat0) / dphi, 0.0)
        fy = np.clip(fy, 0.0, 1.0)

        # Corner weights
        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy

        # Save
        self.i0 = i0.astype(np.int64)
        self.i1 = i1.astype(np.int64)
        self.j0 = j0.astype(np.int64)
        self.j1 = j1.astype(np.int64)
        self.fx = fx
        self.fy = fy
        self.w00 = w00
        self.w10 = w10
        self.w01 = w01
        self.w11 = w11

    # --------------------------- Utilities --------------------------- #

    @staticmethod
    def _ensure_src_mask(src, src_mask):
        if src_mask is None:
            return np.isfinite(src)
        else:
            return np.asarray(src_mask, dtype=bool) & np.isfinite(src)

    def _apply_bilinear_scalar(self, src, src_mask=None):
        """
        Core bilinear (with optional NaN/mask renormalization).
        Returns (out, valid_weight_sum).
        """
        src = np.asarray(src, dtype=float)
        if src.shape != (self.nlat, self.nlon):
            raise ValueError(
                f"src field must have shape (nlat,nlon)=({self.nlat},{self.nlon})"
            )
        valid = self._ensure_src_mask(src, src_mask)

        # Gather corners
        v00 = src[self.j0, self.i0]
        v10 = src[self.j0, self.i1]
        v01 = src[self.j1, self.i0]
        v11 = src[self.j1, self.i1]

        m00 = valid[self.j0, self.i0]
        m10 = valid[self.j0, self.i1]
        m01 = valid[self.j1, self.i0]
        m11 = valid[self.j1, self.i1]

        if self.nan_renorm:
            w00 = self.w00 * m00
            w10 = self.w10 * m10
            w01 = self.w01 * m01
            w11 = self.w11 * m11
            wsum = w00 + w10 + w01 + w11

            num = np.zeros_like(wsum, dtype=float)
            num += np.where(m00, w00 * v00, 0.0)
            num += np.where(m10, w10 * v10, 0.0)
            num += np.where(m01, w01 * v01, 0.0)
            num += np.where(m11, w11 * v11, 0.0)

            out = np.where(wsum > 0.0, num / wsum, np.nan)
            return out, wsum
        else:
            num = self.w00 * v00 + self.w10 * v10 + self.w01 * v01 + self.w11 * v11
            # If any corner NaN, result NaN
            any_nan = ~(m00 & m10 & m01 & m11)
            out = np.where(any_nan, np.nan, num)
            # weight sum is 1 everywhere (but we mark 0 if any invalid to trigger extrapolation)
            wsum = np.where(any_nan, 0.0, 1.0)
            return out, wsum

    def _extrapolate_scalar(self, src, src_mask, where_nan):
        """
        Extrapolate scalar to positions where_nan (boolean mask in target shape).
        Modes: 'nearest' or 'idw'. Returns filled array only at where_nan positions.
        """
        if self.extrapolation_mode is None:
            return np.full(where_nan.shape, self.fill_value, dtype=float)

        # Flatten list of target indices that need extrapolation and are inside tgt_mask
        need = np.where(where_nan & self.tgt_mask)
        if need[0].size == 0:
            return np.full(where_nan.shape, self.fill_value, dtype=float)

        # Valid source points
        valid = self._ensure_src_mask(src, src_mask)
        vmask = valid
        if not np.any(vmask):
            # Nothing to extrapolate from
            out = np.full(where_nan.shape, self.fill_value, dtype=float)
            return out

        # Coordinates of valid sources
        js, is_ = np.where(vmask)
        lonv = self._lon_src_2d[js, is_]
        latv = self._lat_src_2d[js, is_]
        vals = src[js, is_]

        # Prepare outputs
        out_fill = np.full(where_nan.shape, self.fill_value, dtype=float)

        # For each target needing extrapolation, compute nearest / IDW from valid sources
        t_lon = self.lon_tgt_rad[need]
        t_lat = self.lat_tgt_rad[need]

        if self.extrapolation_mode == "nearest":
            # Compute distances to all valid sources; pick argmin
            # (Do this in chunks to cap memory for very large arrays)
            chunk = max(1, 2000)
            for s in range(0, t_lon.size, chunk):
                e = min(s + chunk, t_lon.size)
                lon_blk = t_lon[s:e, None]
                lat_blk = t_lat[s:e, None]
                d = _great_circle_distance_rad(
                    lon_blk, lat_blk, lonv[None, :], latv[None, :]
                )
                jmin = np.argmin(d, axis=1)
                out_fill[need[0][s:e], need[1][s:e]] = vals[jmin]
            return out_fill

            # (IDW below)
        elif self.extrapolation_mode == "idw":
            k = min(self.idw_k, vals.size)
            chunk = max(1, 1000)
            for s in range(0, t_lon.size, chunk):
                e = min(s + chunk, t_lon.size)
                lon_blk = t_lon[s:e, None]
                lat_blk = t_lat[s:e, None]
                d = _great_circle_distance_rad(
                    lon_blk, lat_blk, lonv[None, :], latv[None, :]
                )  # (m,kAll)
                # Select k nearest
                idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]  # (m,k)
                d_k = np.take_along_axis(d, idx, axis=1)  # (m,k)
                v_k = vals[idx]  # (m,k)
                w = 1.0 / (d_k + self.idw_eps)
                wsum = np.sum(w, axis=1)
                out_fill[need[0][s:e], need[1][s:e]] = np.sum(w * v_k, axis=1) / wsum
            return out_fill
        else:
            raise ValueError("extrapolation_mode must be 'nearest', 'idw', or None")

    # --------------------------- Public API --------------------------- #

    def apply_scalar(self, src, src_mask=None):
        """
        Interpolate a scalar field defined on the source grid to the target grid.

        Parameters
        ----------
        src : (nlat, nlon) array-like
            Source scalar field.
        src_mask : (nlat, nlon) boolean, optional
            True where source is valid. If None, validity = isfinite(src).

        Returns
        -------
        out : target-shaped float array
        """
        out, wsum = self._apply_bilinear_scalar(src, src_mask)
        # Extrapolate where bilinear had no valid corners
        need = ~np.isfinite(out)
        if np.any(need):
            ext = self._extrapolate_scalar(np.asarray(src, float), src_mask, need)
            out = np.where(need, ext, out)

        # Apply target mask and fill value
        out = np.where(self.tgt_mask, out, self.fill_value)
        return out.reshape(self.tshape)

    def apply_vector(self, u_src, v_src, src_mask=None):
        """
        Interpolate a vector field (u,v) in east/north components.

        Steps:
          1) Project each source corner (u,v) to 3-D using that corner's local EN basis.
          2) Bilinear blend those 3-D vectors with NaN/mask renormalization.
          3) Project blended 3-D vector to the target EN basis to get (u_t, v_t).
          4) Extrapolate where needed using scalar fallback on |V| and direction from nearest.

        Parameters
        ----------
        u_src, v_src : (nlat, nlon) arrays
            Eastward and northward components on source grid.
        src_mask : (nlat, nlon) boolean, optional
            True where vector is valid. If None, validity = isfinite(u) & isfinite(v).

        Returns
        -------
        u_t, v_t : target-shaped arrays
        """
        u_src = np.asarray(u_src, dtype=float)
        v_src = np.asarray(v_src, dtype=float)
        if u_src.shape != (self.nlat, self.nlon) or v_src.shape != (
            self.nlat,
            self.nlon,
        ):
            raise ValueError(
                f"(u_src,v_src) must both have shape (nlat,nlon)=({self.nlat},{self.nlon})"
            )

        if src_mask is None:
            valid = np.isfinite(u_src) & np.isfinite(v_src)
        else:
            valid = (
                np.asarray(src_mask, dtype=bool)
                & np.isfinite(u_src)
                & np.isfinite(v_src)
            )

        # Build 3-D vector field at sources: V = u*e_east + v*e_north
        V3 = (u_src[..., None] * self._e_east_src) + (
            v_src[..., None] * self._e_north_src
        )  # (nlat,nlon,3)

        # Gather corners
        V00 = V3[self.j0, self.i0, :]
        V10 = V3[self.j0, self.i1, :]
        V01 = V3[self.j1, self.i0, :]
        V11 = V3[self.j1, self.i1, :]

        m00 = valid[self.j0, self.i0]
        m10 = valid[self.j0, self.i1]
        m01 = valid[self.j1, self.i0]
        m11 = valid[self.j1, self.i1]

        if self.nan_renorm:
            w00 = self.w00 * m00
            w10 = self.w10 * m10
            w01 = self.w01 * m01
            w11 = self.w11 * m11
            wsum = (w00 + w10 + w01 + w11)[..., None]  # (..,1)

            num = np.zeros_like(V00)
            num += np.where(m00[..., None], w00[..., None] * V00, 0.0)
            num += np.where(m10[..., None], w10[..., None] * V10, 0.0)
            num += np.where(m01[..., None], w01[..., None] * V01, 0.0)
            num += np.where(m11[..., None], w11[..., None] * V11, 0.0)

            Vt3 = np.where(wsum > 0.0, num / wsum, np.nan)
            need = ~np.isfinite(Vt3[..., 0])
        else:
            num = (
                self.w00[..., None] * V00
                + self.w10[..., None] * V10
                + self.w01[..., None] * V01
                + self.w11[..., None] * V11
            )
            all_ok = m00 & m10 & m01 & m11
            Vt3 = np.where(all_ok[..., None], num, np.nan)
            need = ~all_ok

        # Project to target EN basis
        u_t = np.sum(Vt3 * self._e_east_t, axis=-1)
        v_t = np.sum(Vt3 * self._e_north_t, axis=-1)

        # Extrapolate where needed (vector-aware): we take nearest/IDW of u and v separately in source space
        if np.any(need):
            # For vector extrapolation, re-use scalar extrapolation on u and v components independently.
            u_fill = self._extrapolate_scalar(u_src, valid, need)
            v_fill = self._extrapolate_scalar(v_src, valid, need)
            u_t = np.where(need, u_fill, u_t)
            v_t = np.where(need, v_fill, v_t)

        # Apply target mask & fill
        u_t = np.where(self.tgt_mask, u_t, self.fill_value)
        v_t = np.where(self.tgt_mask, v_t, self.fill_value)

        return u_t.reshape(self.tshape), v_t.reshape(self.tshape)
