import numpy as np
from numpy.typing import NDArray


def _wrap_like(lon_deg: NDArray, base0_deg: float) -> NDArray:
    r"""Maps longitudes (deg) into the [base0, base0+360) interval.

    When longitude is treated as periodic, wrap every target longitude
    :math: `\lambda^{*}_{deg}` into the same half-open interval
    :math: `[ \lambda^{*}_{0}, \lambda^{*}_{0} + 360 )`
    of the (internally ascending) source grid:

    .. math::
        \tilde{\lambda}^{*}_{deg} = \lambda^{deg}_{0} + \mathrm{mod}(\lambda^{*}_{deg} - \lambda^{deg}_{0}, 360)
        \text{where } \lambda^{deg}_{0} = \text{base0\_deg}

    This guarantees consistent bracketing even across the dateline.

    Arguments:
        lon_deg (ndarray): 1-D array of longitudes (degrees)
        base0_deg (float): base longitude (degrees) for wrapping
    Returns:
        (ndarray): array of same shape as lon_deg with wrapped longitudes
    """

    return base0_deg + np.mod(lon_deg - base0_deg, 360.0)


def _unit_east_north(lon_rad: NDArray, lat_rad: NDArray) -> tuple[NDArray, NDArray]:
    r"""Computes unit vectors (east, north) in 3-D for given lon/lat (radians).

    At any geographic point (λ,φ) on the unit sphere, define:
    - Radial unit vector (position on the unit sphere):
        .. math::
            \mathbf{r(\lambda, \phi)} = \begin{pmatrix}
                \cos\phi \cos\lambda \\
                \cos\phi \sin\lambda \\
                \sin\phi
            \end{pmatrix}

    - Orthonormal tangent basis:
        .. math::
            \mathbf{e}_{\text{east}} = \frac{\partial \mathbf{r}}{\partial \lambda} =
            \begin{pmatrix}
                -\sin\lambda \\
                \cos\lambda \\
                0
            \end{pmatrix}, \quad
            \mathbf{e}_{\text{north}} = \frac{\partial \mathbf{r}}{\partial \phi} =
            \begin{pmatrix}
                -\sin\phi \cos\lambda \\
                -\sin\phi \sin\lambda \\
                \cos\phi
            \end{pmatrix} 

    These satisfy:
        .. math::
            \mathbf{e}_{\text{east}} \cdot \mathbf{e}_{\text{north}} = 0 \quad \text{and} \quad
            \|\mathbf{e}_{\text{east}}\| = \|\mathbf{e}_{\text{north}}\| = 1.

    Arguments:
        lon_rad (ndarray): array of longitudes (radians)
        lat_rad (ndarray): array of latitudes (radians)

    Returns:
        (tuple): tuple containing:
            - e_east (ndarray): array of shape (...,3) with east unit vectors
            - e_north (ndarray): array of shape (...,3) with north unit vectors
    """

    slon, clon = np.sin(lon_rad), np.cos(lon_rad)
    slat, clat = np.sin(lat_rad), np.cos(lat_rad)

    # east: (-sin (lon), cos (lon), 0)
    e_east = np.stack((-slon, clon, np.zeros_like(lon_rad)), axis=-1)

    # north: (-sin (lat) * cos (lon), -sin (lat) * sin (lon), cos (lat))
    e_north = np.stack((-slat * clon, -slat * slon, clat), axis=-1)
    return (e_east, e_north)


def _geo_to_cart(lon_rad: NDArray, lat_rad: NDArray) -> NDArray:
    r"""
    Convert geographic coordinates (longitude, latitude) in radians to 3-D
    Cartesian unit vectors on the unit sphere.

    Mathematics:
        For longitude λ and latitude φ (radians) the corresponding point on the
        unit sphere is

            x = cos(φ) * cos(λ)
            y = cos(φ) * sin(λ)
            z = sin(φ)

        These follow from the spherical-to-Cartesian coordinate transform with the
        convention that latitude is the angle north of the equator and longitude
        is the angle east of the prime meridian.

    Arguments:
        lon_rad (ndarray):
            Longitudes in radians. Can be scalar or an array;
            must be broadcastable with lat_rad.
        lat_rad (ndarray):
            Latitudes in radians. Can be scalar or an array;
            must be broadcastable with lon_rad.

    Returns:
        (ndarray): array of shape (..., 3) (broadcasted shape of the inputs plus a trailing
        length-3 axis) containing the Cartesian unit vectors [x, y, z] for each
        input coordinate pair.

    Notes
    -----
    - The returned vectors are unit length up to floating-point precision.
    - Inputs may be of any shape that follows NumPy broadcasting rules; the
      last axis of the output corresponds to (x, y, z).
    """

    slon, clon = np.sin(lon_rad), np.cos(lon_rad)
    slat, clat = np.sin(lat_rad), np.cos(lat_rad)
    x = clat * clon
    y = clat * slon
    z = slat
    return np.stack((x, y, z), axis=-1)


def _great_circle_distance_rad(
    lon1: NDArray, lat1: NDArray, lon2: NDArray, lat2: NDArray
) -> NDArray:
    r"""Haversine great-circle distance (radians) between points on the unit sphere.

    Mathematics:
        The haversine formula gives the central angle (great-circle distance in
        radians) between two points with geographic coordinates
        (longitude, latitude) = (λ, φ):

        .. math::
            \Delta\lambda &= \lambda_2 - \lambda_1,\\
            \Delta\varphi &= \varphi_2 - \varphi_1,\\[6pt]
            a &= \sin^2\!\left(\frac{\Delta\varphi}{2}\right)
                + \cos\varphi_1\,\cos\varphi_2\,
                \sin^2\!\left(\frac{\Delta\lambda}{2}\right),\\[6pt]
            c &= 2\,\operatorname{atan2}\!\left(\sqrt{a},\sqrt{1-a}\right).
        
        The returned value is the central angle :math:`c` in radians. For an
        earth-radius-scaled distance multiply :math:`c` by the desired radius.

    Arguments:
        lon1 (ndarray): 
            Longitudes of the first point(s) in radians. May be scalar or array.
        lat1 (ndarray): 
            Latitudes of the first point(s) in radians. Must be broadcastable
            with ``lon1``.
        lon2 (ndarray): 
            Longitudes of the second point(s) in radians. May be scalar or array.
            Must be broadcastable with ``lat2`` and the other inputs.
        lat2 (ndarray): 
            Latitudes of the second point(s) in radians. Must be broadcastable
            with ``lon2`` and the other inputs.

    Returns:
        (ndarray): Array of great-circle distances (central angles) in radians. 
            The shape is the result of NumPy broadcasting of the inputs.

    Notes:
        - The implementation uses the haversine formulation for numerical
        robustness for small distances.
        - Intermediate value ``a`` is clamped to ``[0, 1]`` to avoid NaNs from
        floating point round-off when computing ``atan2``.
    """

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    sdlat2 = np.sin(dlat * 0.5)
    sdlon2 = np.sin(dlon * 0.5)
    a = sdlat2 * sdlat2 + np.cos(lat1) * np.cos(lat2) * sdlon2 * sdlon2

    # Clamp for safety
    a = np.clip(a, 0.0, 1.0)
    result: NDArray = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return result


class BilinearRectilinearInterpolator:
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
        - lon_src: shape (NX,), degrees, strictly monotonic ascending (wrap allowed across dateline)
        - lat_src: shape (NY,), degrees, strictly monotonic (ascending or descending)
        - lon_tgt, lat_tgt: target coordinates, any shape (broadcastable to the same)

    Masks:
        - src_mask: boolean array (NY, NX) where True means valid; if None, all valid
        - tgt_mask: boolean array like lon_tgt/lat_tgt shape; True means compute/keep output, False -> fill_value

    Vector interpolation:
        - (u,v) are eastward and northward components on the sphere (m/s etc.)
        - We rotate vectors properly by projecting each corner vector to 3-D (using local EN basis),
          bilinear-blending in 3-D, then projecting onto the target EN basis.

    Extrapolation:
        - If all four bilinear corners are invalid, we extrapolate from sources:
            - mode='nearest': nearest valid source point
            - mode='idw': inverse-distance weighted blend of k nearest valid sources
    """

    def __init__(
        self,
        lon_src: NDArray,
        lat_src: NDArray,
        lon_tgt: NDArray,
        lat_tgt: NDArray,
        src_mask: NDArray | None = None,
        tgt_mask: NDArray | None = None,
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
        lon_src_deg = np.asarray(lon_src, dtype=float)
        lat_src_deg = np.asarray(lat_src, dtype=float)
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
        lon_tgt = np.asarray(lon_tgt, dtype=float)
        lat_tgt = np.asarray(lat_tgt, dtype=float)
        lon_tgt_deg, lat_tgt_deg = np.meshgrid(lon_tgt, lat_tgt)

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

    def _precompute_cells_and_weights(self) -> None:
        r"""For each target point, compute (i0,i1,j0,j1) and (fx,fy) and corner weights.

        Mathematics:
            For each target :math: `(\tilde{\lambda}^{*}, \phi^{*})` we find bracketing indices
            .. math::
                (i_0, i_1) \in \{0, \ldots, N_x - 1\}^2, \quad
                (j_0, j_1) \in \{0, \ldots, N_y - 1\}^2,

            such that :math: `(i_0, i_1)` are consecutive longitudes around :math: `(\tilde{\lambda}^*)`,
            and :math: `(j_0, j_1)` are consecutive latitudes around :math: `(\varphi^*)`.

            If the target lies beyond the non-periodic ends, indices are clamped; for periodic longitude, indices wrap modulo :math: `(N_x)`.

            Let
            .. math::
                \lambda_0 = \lambda_{i_0}, \quad
                \lambda_1 = \lambda_{i_1}, \quad
                \varphi_0 = \varphi_{j_0}, \quad
                \varphi_1 = \varphi_{j_1}.

            **Forward (wrapped) longitudinal difference**

            Across the dateline, we must measure the **forward** difference from :math: `(i_0)` to :math: `(i_1)`.
            Because longitude wraps, we define the **forward** cell width

            .. math::
                \Delta \lambda_{\text{cell}} =
                \begin{cases}
                    (\lambda_1 + 2\pi) - \lambda_0, & \text{if } i_1 \le i_0 \text{ (wrapped cell)}, \\
                    \lambda_1 - \lambda_0, & \text{otherwise.}
                \end{cases}

            and the **forward displacement**

            .. math::
                \Delta \tilde{\lambda}^* = \tilde{\lambda}^* - \lambda_0, \quad
                \Delta \tilde{\lambda}^* \leftarrow
                \begin{cases}
                    \Delta \tilde{\lambda}^* + 2\pi, & \text{if } \Delta \tilde{\lambda}^* < 0, \\
                    \Delta \tilde{\lambda}^*, & \text{otherwise.}
                \end{cases}

            Then the fractional longitudinal coordinate is  

            .. math::
                f_x = \frac{\Delta \tilde{\lambda}^*}{\Delta \lambda_{\text{cell}}} \in [0, 1],

            (after clipping if needed).

            **Latitudinal fraction**

            Regardless of ascending/descending latitude ordering,

            .. math::
                \Delta \varphi_{\text{cell}} = \varphi_1 - \varphi_0,
                \quad
                f_y = \frac{\varphi^* - \varphi_0}{\Delta \varphi_{\text{cell}}}.

            and then clip :math: `f_y` to :math: `[0, 1]`. If latitudes are descending, 
            :math: `(\Delta \varphi_{\text{cell}} < 0)`, and the fraction remains consistent after clipping.

            **Bilinear shape functions (weights)**

            On the rectangle :math: `(i_0, i_1) \times (j_0, j_1)`, the four standard bilinear basis functions are

            .. math::
                w_{00} = (1 - f_x)(1 - f_y), \quad
                w_{10} = f_x(1 - f_y),
            .. math::
                w_{01} = (1 - f_x)f_y, \quad
                w_{11} = f_x f_y.

            They satisfy :math: `w_{ab} \ge 0` and :math: `\sum w_{ab} = 1`.

            **Corner mapping:**

            .. math::
                (0,0) \mapsto (j_0, i_0), \quad
                (1,0) \mapsto (j_0, i_1), \quad
                (0,1) \mapsto (j_1, i_0), \quad
                (1,1) \mapsto (j_1, i_1).
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
    def _ensure_src_mask(src: NDArray, src_mask: NDArray | None) -> NDArray:
        result: NDArray = np.empty(src.shape, dtype=bool)
        if src_mask is None:
            result[...] = np.isfinite(src)
        else:
            result[...] = np.asarray(src_mask, dtype=bool) & np.isfinite(src)
        return result

    def _apply_bilinear_scalar(
        self, src: NDArray, src_mask: NDArray | None
    ) -> tuple[NDArray, NDArray]:
        r"""Core bilinear (with optional NaN/mask renormalization).

        Mathematics:
            **Mask/NaN-aware renormalization**

            Let the corner validity be

            .. math::
                \mu_{00} = m^{\text{src}}_{j_0, i_0}, \quad
                \mu_{10} = m^{\text{src}}_{j_0, i_1}, \quad
                \mu_{01} = m^{\text{src}}_{j_1, i_0}, \quad
                \mu_{11} = m^{\text{src}}_{j_1, i_1}
                \in \{0, 1\}.

            (If values are NaN, take the corresponding :math: `\mu = 0`.)

            We down-weight invalid corners:

            .. math::
                \tilde{w}_{ab} = w_{ab} \mu_{ab}, \quad
                W = \sum_{a,b \in \{0,1\}} \tilde{w}_{ab}.

            * If :math: `W > 0` (some valid corners): **renormalize**

            .. math::
                \hat{w}_{ab} = \frac{\tilde{w}_{ab}}{W}, \quad
                \sum \hat{w}_{ab} = 1,

            and the scalar interpolation is

            .. math::
                s^* = \sum_{a,b} \hat{w}_{ab} \, s_{j_b, i_a},

            where :math: `i_{0/1} = i_0/i_1` and :math: `j_{0/1} = j_0/j_1`.

            * If :math: `W = 0`: all four corners invalid ⇒ **extrapolate** (Section 7).

            (If renormalization is **disabled**, then :math: `s^* = \sum w_{ab} s_{j_b, i_a}`
            only if all four corners are valid; otherwise :math: `s^* = \text{NaN}` and we fall back to extrapolation.)

        Arguments:
            src (ndarray): source scalar field (NY, NX)
            src_mask (ndarray or None): optional boolean mask (NY, NX) where True means valid

        Returns:
            tuple (ndarray, ndarray): Interpolated values and valid weight sum.
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

    def _extrapolate_scalar(
        self, src: NDArray, src_mask: NDArray | None, where_nan: NDArray
    ) -> NDArray:
        r"""Extrapolate scalar to positions where_nan (boolean mask in target shape).

        Mathematics:
            Extrapolation on the sphere (when all 4 corners are invalid)

            Let
            .. math::
                \mathcal{S} = \{(\lambda_p, \varphi_p) : m^{\text{src}}_p = 1\}

            be all valid source points (flattened index :math: `p` maps to :math: `(j, i)`).
            For a target :math: `(\lambda^*, \varphi^*)`, we compute **great-circle distances** using the haversine formula.

            ---

            Two supported modes:

            #### Nearest neighbor

            .. math::
                p^* = \arg \min_{p \in \mathcal{S}} \delta_p, \quad
                s^* = s_{p^*} \quad \text{or} \quad (u^*, v^*) = (u_{p^*}, v_{p^*}).

            ---

            #### Inverse-distance weighting (IDW)

            Choose the :math: `K` nearest valid sources :math: `\mathcal{N}_K \subset \mathcal{S}`.
            With a small :math: `\varepsilon > 0` to avoid division by zero, define:

            .. math::
                \tilde{w}_p = \frac{1}{\delta_p + \varepsilon}, \quad
                W = \sum_{p \in \mathcal{N}_K} \tilde{w}_p, \quad
                \hat{w}_p = \frac{\tilde{w}_p}{W}.

            Then

            .. math::
                s^* = \sum_{p \in \mathcal{N}_K} \hat{w}_p \, s_p,
                \quad
                u^* = \sum_{p \in \mathcal{N}_K} \hat{w}_p \, u_p,
                \quad
                v^* = \sum_{p \in \mathcal{N}_K} \hat{w}_p \, v_p.

            (The code extrapolates $u$ and $v$ separately for this fallback.)

            > **Note:** IDW preserves constants and reduces to nearest neighbor as :math: `K \to 1`
            > or when one :math: `\delta_p \ll` others.

        Arguments:
            src (ndarray): source scalar field (NY, NX)
            src_mask (ndarray or None): optional boolean mask (NY, NX) where True means valid
            where_nan (ndarray): boolean mask in target shape where extrapolation is needed

        Returns:
            (ndarray): filled array only at where_nan positions.
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

    def apply_scalar(self, src: NDArray, src_mask: NDArray | None = None) -> NDArray:
        """Interpolate a scalar field defined on the source grid to the target grid.

        Arguments:
            src (ndarray(NY, NX)): Source scalar field.
            src_mask (ndarray(NY, NX), optional):
                True where source is valid. If None, validity = isfinite(src).

        Returns:
            (ndarray): target-shaped float array
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

    def apply_vector(
        self, u_src: NDArray, v_src: NDArray, src_mask: NDArray | None = None
    ) -> tuple[NDArray, NDArray]:
        r"""Interpolate a vector field (u,v) in east/north components.

        Steps:
            1) Project each source corner (u,v) to 3-D using that corner's local EN basis.
            2) Bilinear blend those 3-D vectors with NaN/mask renormalization.
            3) Project blended 3-D vector to the target EN basis to get (u_t, v_t).
            4) Extrapolate where needed using scalar fallback on |V| and direction from nearest.

        Mathematics:
            At each source corner :math: `(\lambda_{i_a}, \varphi_{j_b})`, convert :math: `(u, v)` to a 3-D vector:

            .. math::
                \mathbf{V}_{ab}
                = u_{j_b, i_a} \, \mathbf{e}_{\text{east}}(\lambda_{i_a}, \varphi_{j_b})
                + v_{j_b, i_a} \, \mathbf{e}_{\text{north}}(\lambda_{i_a}, \varphi_{j_b}).

            Then apply the **same mask-aware bilinear blend** to the 3-D vectors:

            .. math::
                \mathbf{V}^* = \sum_{a,b} \hat{w}_{ab} \, \mathbf{V}_{ab}
                \quad (\text{if } W > 0; \text{ else extrapolate}).

            Finally, **project** the blended 3-D vector onto the target tangent basis at :math: `(\lambda^*, \varphi^*)`:

            .. math::
                u^* = \mathbf{V}^* \cdot \mathbf{e}_{\text{east}}(\lambda^*, \varphi^*),
                \quad
                v^* = \mathbf{V}^* \cdot \mathbf{e}_{\text{north}}(\lambda^*, \varphi^*).

            This procedure automatically rotates vectors correctly across the dateline and anywhere on the sphere
            (because the local bases vary with :math: `\lambda, \varphi`), while keeping the interpolation linear.

        Arguments:
            u_src (ndarray(NY, NX)): Eastward components on source grid.
            v_src (ndarray(NY, NX)): Northward components on source grid.
            src_mask (ndarray(NY, NX), optional): True where vector is valid. If None, validity = isfinite(u) & isfinite(v).

        Returns:
            tuple (ndarray, ndarray): target-shaped arrays
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

        return (u_t.reshape(self.tshape), v_t.reshape(self.tshape))
