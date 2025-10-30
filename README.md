# VerCOR
Versatile Earth system COupleR (VerCOR)

## Bilinear interpolation on a unit sphere

### 1) Grids, indexing, and notation

#### Source (rectilinear) grid

Longitudes $`\{\lambda_i\}_{i=0}^{N_x-1}`$ (strictly monotone in the code’s internal representation, ascending after an optional flip).  
Latitudes $`\{\varphi_j\}_{j=0}^{N_y-1}`$ (strictly monotone; may be ascending or descending).

**Index ranges:**

$$
    i \in \{0, \ldots, N_x - 1\}, \quad
    j \in \{0, \ldots, N_y - 1\}.
$$

A scalar field on the source grid is $`s_{j,i}`$.
A vector field is $`(u_{j,i}, v_{j,i})`$ in local east/north components.

---

#### Target points

A (possibly non-rectangular) target set $`{(\lambda^*, \varphi^*)}`$ with broadcast shape $`\mathcal{T}`$.
All formulas below apply pointwise over $`\mathcal{T}`$.

---

#### Masks

Source mask $`m^{\text{src}}_{j,i} \in \{0,1\}`$ (True/False in code) indicates validity of $`s_{j,i}`$
(and of vector components similarly).

Target mask $`m^{\text{tgt}}(\lambda^*, \varphi^*) \in \{0,1\}`$ indicates whether to keep the output or place  
`fill_value`.

### 2) Periodic longitude wrapping

When longitude is treated as periodic, wrap every target longitude 
    $$ \lambda^{*}_{deg} $$ into the same half-open interval
    $$ [ \lambda^{*}_{0}, \lambda^{*}_{0} + 360 ) $$

of the (internally ascending) source grid:

$$
    \tilde{\lambda}^{*}_{deg} = \lambda^{deg}_{0} + \mathrm{mod}(\lambda^{*}_{deg} - \lambda^{deg}_{0}, 360)
    \text{where } \lambda^{deg}_{0} = \text{base0\_deg} 
$$

and convert to radians when needed,
$$
    \tilde{\lambda}^{*} = \tilde{\lambda}^{*}_{deg} \cdot \pi / 180
$$

This guarantees consistent bracketing even across the dateline.

### 3) Cell search and local bilinear coordinates

For each target $$(\tilde{\lambda}^{*}, \phi^{*})$$ we find bracketing indices
    $$
        (i_0, i_1) \in \{0, \ldots, N_x - 1\}^2, \quad
        (j_0, j_1) \in \{0, \ldots, N_y - 1\}^2,
    $$

such that $(i_0, i_1)$ are consecutive longitudes around $(\tilde{\lambda}^*)$, and $(j_0, j_1)$ are consecutive latitudes around $(\varphi^*)$.

If the target lies beyond the non-periodic ends, indices are clamped; for periodic longitude, indices wrap modulo $(N_x)$.

Let
$$
    \lambda_0 = \lambda_{i_0}, \quad
    \lambda_1 = \lambda_{i_1}, \quad
    \varphi_0 = \varphi_{j_0}, \quad
    \varphi_1 = \varphi_{j_1}.
$$

#### 3.1) Forward (wrapped) longitudinal difference

Across the dateline, we must measure the **forward** difference from $i_0$ to $i_1$. Because longitude wraps, we define the **forward** cell width

$$
    \Delta \lambda_{\text{cell}} =
    \begin{cases}
        (\lambda_1 + 2\pi) - \lambda_0, & \text{if } i_1 \le i_0 \text{ (wrapped cell)}, \\
        \lambda_1 - \lambda_0, & \text{otherwise.}
    \end{cases}
$$

and the **forward displacement**

$$
    \Delta \tilde{\lambda}^* = \tilde{\lambda}^* - \lambda_0, \quad
    \Delta \tilde{\lambda}^* \leftarrow
    \begin{cases}
        \Delta \tilde{\lambda}^* + 2\pi, & \text{if } \Delta \tilde{\lambda}^* < 0, \\
        \Delta \tilde{\lambda}^*, & \text{otherwise.}
    \end{cases}
$$

Then the fractional longitudinal coordinate is  

$$
f_x = \frac{\Delta \tilde{\lambda}^*}{\Delta \lambda_{\text{cell}}} \in [0, 1],
$$

(after clipping if needed).

#### 3.2) Latitudinal fraction

Regardless of ascending/descending latitude ordering,

$$
    \Delta \varphi_{\text{cell}} = \varphi_1 - \varphi_0,
    \quad
    f_y = \frac{\varphi^* - \varphi_0}{\Delta \varphi_{\text{cell}}}.
$$

and then clip $f_y$ to $[0, 1]$. If latitudes are descending, $(\Delta \varphi_{\text{cell}} < 0)$, and the fraction remains consistent after clipping.

### 4) Bilinear shape functions (weights)

On the rectangle $(i_0, i_1) \times (j_0, j_1)$, the four standard bilinear basis functions are  

$$
    w_{00} = (1 - f_x)(1 - f_y), \quad
    w_{10} = f_x(1 - f_y),
$$
$$
    w_{01} = (1 - f_x)f_y, \quad
    w_{11} = f_x f_y.
$$

They satisfy $w_{ab} \ge 0$ and $\sum w_{ab} = 1.$

**Corner mapping:**

$$
    (0,0) \mapsto (j_0, i_0), \quad
    (1,0) \mapsto (j_0, i_1), \quad
    (0,1) \mapsto (j_1, i_0), \quad
    (1,1) \mapsto (j_1, i_1).
$$

---

### 5) Mask/NaN-aware renormalization

Let the corner validity be  

$$
    \mu_{00} = m^{\text{src}}_{j_0, i_0}, \quad
    \mu_{10} = m^{\text{src}}_{j_0, i_1}, \quad
    \mu_{01} = m^{\text{src}}_{j_1, i_0}, \quad
    \mu_{11} = m^{\text{src}}_{j_1, i_1}
    \in \{0, 1\}.
$$

(If values are NaN, take the corresponding $\mu = 0$.)

We down-weight invalid corners:

$$
    \tilde{w}_{ab} = w_{ab} \mu_{ab}, \quad
    W = \sum_{a,b \in \{0,1\}} \tilde{w}_{ab}.
$$

* If $W > 0$ (some valid corners): **renormalize**

$$
    \hat{w}_{ab} = \frac{\tilde{w}_{ab}}{W}, \quad
    \sum \hat{w}_{ab} = 1,
$$

and the scalar interpolation is  

$$
    s^* = \sum_{a,b} \hat{w}_{ab} \, s_{j_b, i_a},
$$

where $i_{0/1} = i_0/i_1$ and $j_{0/1} = j_0/j_1.$

* If $W = 0$: all four corners invalid ⇒ **extrapolate** (Section 7).

(If renormalization is **disabled**, then $s^* = \sum w_{ab} s_{j_b, i_a}$ only if all four corners are valid; otherwise $s^* = \text{NaN}$ and we fall back to extrapolation.)

### 6) Vector fields: correct spherical rotation via 3-D projection

Let $(u, v)$ be **eastward** and **northward** components (tangent to the sphere).  
At any geographic point $(\lambda, \varphi)$ on the unit sphere, define:

---

#### • Radial unit vector (position on the unit sphere)

$$
    \mathbf{r(\lambda, \phi)} = \begin{pmatrix}
        \cos\phi \cos\lambda \\
        \cos\phi \sin\lambda \\
        \sin\phi
    \end{pmatrix}.
$$

---

#### • Orthonormal tangent basis

$$
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
    \end{pmatrix}.
$$

These satisfy  
$\mathbf{e}_{\text{east}} \cdot \mathbf{e}_{\text{north}} = 0$ and  
$\|\mathbf{e}_{\text{east}}\| = \|\mathbf{e}_{\text{north}}\| = 1.$

---

At each source corner $(\lambda_{i_a}, \varphi_{j_b})$, convert $(u, v)$ to a 3-D vector:

$$
    \mathbf{V}_{ab}
    = u_{j_b, i_a} \, \mathbf{e}_{\text{east}}(\lambda_{i_a}, \varphi_{j_b})
    + v_{j_b, i_a} \, \mathbf{e}_{\text{north}}(\lambda_{i_a}, \varphi_{j_b}).
$$

Then apply the **same mask-aware bilinear blend** to the 3-D vectors:

$$
    \mathbf{V}^* = \sum_{a,b} \hat{w}_{ab} \, \mathbf{V}_{ab}
    \quad (\text{if } W > 0; \text{ else extrapolate}).
$$

Finally, **project** the blended 3-D vector onto the target tangent basis at $`(\lambda^{*}, \varphi^{*})`$:

$$
    u^* = \mathbf{V}^* \cdot \mathbf{e}_{\text{east}}(\lambda^*, \varphi^*),
    \quad
    v^* = \mathbf{V}^* \cdot \mathbf{e}_{\text{north}}(\lambda^*, \varphi^*).
$$

This procedure automatically rotates vectors correctly across the dateline and anywhere on the sphere  
(because the local bases vary with $\lambda, \varphi$), while keeping the interpolation linear.


### 7) Extrapolation on the sphere (when all 4 corners are invalid)

Let  
$$
    \mathcal{S} = \{(\lambda_p, \varphi_p) : m^{\text{src}}_p = 1\}
$$  
be all valid source points (flattened index $p$ maps to $(j, i)$).  
For a target $(\lambda^*, \varphi^*)$, we compute **great-circle distances** using the haversine formula:

$$
    \delta_p = d_{\text{gc}}\big((\lambda^*, \varphi^*), (\lambda_p, \varphi_p)\big)
    = 2 \arctan 2\!\left(\sqrt{a_p}, \sqrt{1 - a_p}\right),
$$

$$
    a_p = \sin^2 \frac{\varphi_p - \varphi^*}{2}
    + \cos\varphi^* \cos\varphi_p \sin^2 \frac{\lambda_p - \lambda^*}{2}.
$$

---

Two supported modes:

#### 7.1 Nearest neighbor

$$
    p^* = \arg \min_{p \in \mathcal{S}} \delta_p, \quad
    s^* = s_{p^*} \quad \text{or} \quad (u^*, v^*) = (u_{p^*}, v_{p^*}).
$$

---

#### 7.2 Inverse-distance weighting (IDW)

Choose the $K$ nearest valid sources $\mathcal{N}_K \subset \mathcal{S}$.  
With a small $\varepsilon > 0$ to avoid division by zero, define:

$$
    \tilde{w}_p = \frac{1}{\delta_p + \varepsilon}, \quad
    W = \sum_{p \in \mathcal{N}_K} \tilde{w}_p, \quad
    \hat{w}_p = \frac{\tilde{w}_p}{W}.
$$

Then

$$
    s^* = \sum_{p \in \mathcal{N}_K} \hat{w}_p \, s_p,
    \quad
    u^* = \sum_{p \in \mathcal{N}_K} \hat{w}_p \, u_p,
    \quad
    v^* = \sum_{p \in \mathcal{N}_K} \hat{w}_p \, v_p.
$$

(The code extrapolates $u$ and $v$ separately for this fallback.)

> **Note:** IDW preserves constants and reduces to nearest neighbor as $K \to 1$  
> or when one $\delta_p \ll$ others.

---

### 8) Target mask and fill value

After interpolation/extrapolation, the final output applies the target mask:

$$
    s^{\text{out}}(\lambda^*, \varphi^*) =
    \begin{cases}
        s^*(\lambda^*, \varphi^*), & m^{\text{tgt}}(\lambda^*, \varphi^*) = 1, \\
        \text{fill\_value}, & \text{otherwise,}
    \end{cases}
$$

and similarly for $(u, v)$.

### 9) Computational complexity (per target field)

- **Precompute (once):**  
  Cell search by binary search  
  $$
    \mathcal{O}(|\mathcal{T}| \log N_x + |\mathcal{T}| \log N_y);
  $$
  vector bases:  
  $$
    \mathcal{O}(N_x N_y)
  $$
  to form $\mathbf{e}_{\text{east}}, \mathbf{e}_{\text{north}}$.

- **Apply scalar field:**  
  $$
    \mathcal{O}(|\mathcal{T}|)
  $$

- **Apply vector field:**  
  $$
    \mathcal{O}(|\mathcal{T}|) \text{ arithmetic using cached bases.}
  $$

- **Extrapolation (only where needed):**  
  Nearest: naive  
  $$
    \mathcal{O}(|\mathcal{T}|_{\text{need}} \cdot N_x N_y)
  $$
  with small practical chunks;  
  IDW adds a partial sort for $K$.

---

### 10) Properties & remarks

- **Partition of unity.**  
  If at least one valid corner exists and renormalization is used,  
  $\sum \hat{w}_{ab} = 1.$

- **Constants are preserved.**  
  If $s_{j,i} \equiv c$, then $s^* = c.$

- **Linearity.**  
  The map $s \mapsto s^*$ is linear; vector interpolation is linear in $(u, v)$.

- **Periodic consistency.**  
  The wrapped forward differences ensure cells that straddle $\lambda = 180^\circ$ behave exactly like any other cell.

- **Non-uniform spacing.**  
  Only the local bracketing grid lines enter $f_x, f_y$; spacing may vary arbitrarily.

- **Vector correctness.**  
  Converting to 3-D and back projects between **local tangent planes** and correctly rotates directions with longitude/latitude — crucial near the dateline or at high latitudes.

- **Extrapolation locality.**  
  Extrapolation is applied **only** where all four corners are invalid; elsewhere you get true bilinear interpolation (with or without renormalization).

