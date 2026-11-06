"""Theoretical chip flow angle on the rake face for OD turning with a nosed insert.

Coordinates in the tool reference plane (seen from above, along the cutting speed):
    z = feed direction (towards the chuck), x = radially outward.
The nose bottom touches the machined surface at (0, 0); the stock surface is at x = ap.

Uncut chip area for one revolution = tool silhouette now, minus everything the tool covered
one feed (f) earlier. The engaged cutting edge runs from the major edge at x = ap, round the
nose, to the cusp point where the current trailing nose meets the previous revolution.

eta (deg) = direction of chip flow across the rake face, measured from -z (straight back over
the insert, i.e. perpendicular to a 90-degree major edge) towards +x (radially outward).

  Colwell (1954): flow is perpendicular to the chord joining the two ends of the engaged edge.
  Young, Mathew & Oxley (1987): each edge element pushes chip along its own normal, weighted
  by the chip area it removes; flow follows the resultant.
"""
import math
import numpy as np

# Assumed tooling (insert code unknown): CCMT 09T304 (80 deg rhombic, r_eps = 0.4 mm, screw
# clamped, as seen in the videos) in an SCLCR 95-degree holder.
DEFAULT_TOOL = dict(r_eps=0.4, kappa_r=95.0, eps_r=80.0)


def tool_branches(r_eps, kappa_r, eps_r, x_max, n=4000):
    """Front (leading) and back (trailing) boundary of the tool as z(x) for x in [0, x_max]."""
    kmin = 180.0 - kappa_r - eps_r                     # minor-edge entering angle
    c = np.array([0.0, r_eps])                         # nose centre
    a_maj = math.radians(90.0 - kappa_r)               # outward normal angle of major edge (from +z)
    a_min = math.radians(-90.0 - kmin)                 # outward normal angle of minor edge
    xs = np.linspace(0.0, x_max, n)

    def branch(a_end, edge_dir):
        # arc from the nose bottom (normal -90 deg) to the tangent point, then the straight edge
        ang = np.linspace(-math.pi / 2, a_end, 2000)
        arc = c[None, :] + r_eps * np.stack([np.cos(ang), np.sin(ang)], 1)
        t = arc[-1]
        L = (x_max + 1.0) / max(edge_dir[1], 1e-6)
        line = t[None, :] + np.linspace(0, L, 2000)[:, None] * np.array(edge_dir)[None, :]
        pts = np.vstack([arc, line[1:]])
        order = np.argsort(pts[:, 1])
        return np.interp(xs, pts[order, 1], pts[order, 0])

    zf = branch(a_maj, (math.cos(math.radians(kappa_r)), math.sin(math.radians(kappa_r))))
    zb = branch(a_min, (-math.cos(math.radians(kmin)), math.sin(math.radians(kmin))))
    return xs, zf, zb


def flow_angles(f, ap, r_eps=0.4, kappa_r=95.0, eps_r=80.0):
    xs, zf, zb = tool_branches(r_eps, kappa_r, eps_r, ap)
    lo = np.maximum(zb, zf - f)                        # chip at height x spans [lo, zf]
    width = zf - lo
    # cusp: highest x where the back boundary is still the current tool (zb > zf - f)
    own_back = zb >= zf - f
    xc = xs[own_back].max() if own_back.any() else 0.0
    A = np.array([zf[-1], ap])
    B = np.array([np.interp(xc, xs, zb), xc])
    chord = A - B
    d_col = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)      # perpendicular, into the tool
    eta_col = math.degrees(math.atan2(d_col[1], -d_col[0]))

    # Young-Mathew-Oxley: split the chip area into cells, each assigned to the nearest engaged-edge
    # point; sum the inward edge normals weighted by cell area.
    edge = [np.stack([zf, xs], 1)]
    back = xs <= xc
    if back.any():
        edge.append(np.stack([zb[back], xs[back]], 1))
    edge = np.vstack(edge)
    normals = []
    if back.sum() < 3:
        edge = edge[: len(zf)]
    for zz, sign in ((zf, 1), (zb[back], -1)):
        if len(zz) < 3:
            continue
        xx = xs[: len(zz)]
        dz, dx = np.gradient(zz), np.gradient(xx)
        # tangent (dz, dx) rotated +90 deg points into the tool for the front branch, -90 for the back
        nrm = np.stack([-dx, dz], 1) * sign
        normals.append(nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12))
    normals = np.vstack(normals)
    # sample the chip area
    zs = np.linspace(min(lo.min(), zf.min()), zf.max(), 300)
    Z, X = np.meshgrid(zs, xs[::20])
    LO, ZF = np.interp(X, xs, lo), np.interp(X, xs, zf)
    inside = (Z >= LO) & (Z <= ZF)
    P = np.stack([Z[inside], X[inside]], 1)
    from scipy.spatial import cKDTree
    _, idx = cKDTree(edge).query(P)
    vec = normals[idx].sum(0)
    d_ymo = vec / np.linalg.norm(vec)
    eta_ymo = math.degrees(math.atan2(d_ymo[1], -d_ymo[0]))

    area = float(np.trapezoid(width, xs))
    L_edge = float(np.sum(np.linalg.norm(np.diff(edge, axis=0), axis=1)))
    return dict(eta_colwell=eta_col, eta_ymo=eta_ymo, chip_area=area, edge_len=L_edge,
                h_eq=area / L_edge, cusp_x=xc)


if __name__ == "__main__":
    print("sanity: no nose effect when ap >> r_eps -> eta -> 90 - kappa_r =", 90 - DEFAULT_TOOL["kappa_r"])
    for f, ap in [(0.08, 0.9), (0.15, 0.9), (0.08, 0.3), (0.15, 0.3), (0.08, 0.6), (0.15, 0.6), (0.1, 20.0)]:
        r = flow_angles(f, ap, **DEFAULT_TOOL)
        print(f"f={f:.2f} ap={ap:4.1f}  Colwell={r['eta_colwell']:6.2f}  YMO={r['eta_ymo']:6.2f}  "
              f"area={r['chip_area']:.4f} (f*ap={f*ap:.4f})  h_eq={r['h_eq']:.4f}")
