"""Extract 18 dimensionless traits from binary stripe masks.
Width L, height H and area L*H define normalization.
Invalid masks return NaN features with valid=False and an invalid_reason.
Undefined quantities, including pure-ring tortuosity, are represented by NaN."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy import fft, ndimage as ndi
from skimage.measure import euler_number, label, regionprops
from skimage.morphology import skeletonize


FEATURE_NAMES = (
    "melanistic_ratio", "rel_char_wavelength", "spectral_entropy",
    "dominant_orientation_sin2", "anisotropy_index", "fractal_dimension",
    "lacunarity", "ap_density_gradient", "dv_density_gradient",
    "loop_to_stripe_ratio", "rel_width_median", "width_iqr_ratio",
    "rel_spacing_median", "bifurcation_rate", "mean_tortuosity",
    "mean_dorsoventral_span", "solidity_median", "isolated_spot_fraction",
)

PRUNE_HALF_WIDTH_MULTIPLIER = 1.2
_NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))


def _invalid(reason: str, shape: tuple[int, ...]) -> tuple[dict, dict]:
    return ({name: float("nan") for name in FEATURE_NAMES},
            {"valid": False, "invalid_reason": reason, "shape": shape,
             "feature_names": FEATURE_NAMES, "warnings": [reason]})


def _frequency_features(mask: np.ndarray) -> tuple[dict, dict]:
    """Hann-windowed spectrum with unit-width bins in cycles per torso length.

    The radial PSD is integrated shell power, not the shell average. Entropy
    is Shannon entropy in nats; it is not divided by a resolution-dependent
    logarithm. Frequencies are retained only within the inscribed Nyquist
    circle. The zero-frequency shell is excluded from peak/entropy estimates.
    """
    h, w = mask.shape
    window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    centered = (mask.astype(float) - mask.mean()) * window
    power = np.abs(fft.fftshift(fft.fft2(centered))) ** 2
    fy = fft.fftshift(fft.fftfreq(h))
    fx = fft.fftshift(fft.fftfreq(w))
    squared_radius = fy[:, None] ** 2 + fx[None, :] ** 2
    radius_rel = np.sqrt(squared_radius) * w
    shell = np.floor(radius_rel + 0.5).astype(np.int32)
    usable = (squared_radius <= 0.25 + 1e-12) & (shell > 0)
    limit = int(np.floor(w / 2))
    usable &= shell <= limit
    radial = np.bincount(shell[usable], weights=power[usable], minlength=limit + 1)
    counts = np.bincount(shell[usable], minlength=limit + 1)
    frequencies_rel = np.arange(len(radial), dtype=float)
    radial[0] = 0.0
    total = radial.sum()
    values = {name: float("nan") for name in
              ("rel_char_wavelength", "spectral_entropy", "dominant_orientation_sin2", "anisotropy_index")}
    angle = cos2 = float("nan")
    if total > 0 and np.isfinite(total):
        probabilities = radial / total
        occupied = probabilities > 0
        values["spectral_entropy"] = float(-np.sum(probabilities[occupied] * np.log(probabilities[occupied])))
        peak = int(np.argmax(radial))
        delta = 0.0
        if 1 < peak < len(radial) - 1 and np.all(radial[peak - 1:peak + 2] > 0):
            a, b, c = np.log(radial[peak - 1:peak + 2])
            denominator = a - 2 * b + c
            if denominator < -1e-15:
                delta = float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))
        peak_rel = peak + delta
        values["rel_char_wavelength"] = float(1.0 / peak_rel)
        angular = np.zeros(mask.shape, dtype=complex)
        np.divide((fx[None, :] + 1j * fy[:, None]) ** 2, squared_radius,
                  out=angular, where=squared_radius > 0)
        z = np.sum(power[usable] * angular[usable]) / np.sum(power[usable])
        anisotropy = float(np.clip(abs(z), 0, 1))
        values["anisotropy_index"] = anisotropy
        if anisotropy > 1e-6:
            # Stripe angle minus the vertical axis equals wave-vector angle,
            # modulo pi. Image row coordinates increase downwards.
            values["dominant_orientation_sin2"] = float(z.imag / abs(z))
            cos2 = float(z.real / abs(z))
            angle = float(0.5 * np.angle(z))
    else:
        peak_rel = float("nan")
    diagnostics = {
        "fft_power": power, "fft_fx": fx, "fft_fy": fy,
        "radial_frequency": frequencies_rel / w,
        "radial_frequency_rel": frequencies_rel, "radial_psd": radial,
        "radial_psd_shell_mean": np.divide(radial, counts, out=np.zeros_like(radial), where=counts > 0),
        "dominant_frequency_rel": peak_rel,
        "dominant_orientation_cos2": cos2,
        "dominant_orientation_relative_dv_radians": angle,
        "spectrum_definition": "Hann-windowed mean-centered mask; shell-integrated power; non-DC inscribed Nyquist circle",
        "spectral_entropy_units": "nats",
    }
    return values, diagnostics


def _fractal_dimension(mask: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Occupied foreground boxes at dyadic fractions of the shorter dimension."""
    short = min(mask.shape)
    sizes = np.unique([max(1, int(round(short / (2 ** k)))) for k in range(1, 8)])
    counts = []
    for size in sizes:
        padded = np.pad(mask, ((0, (-mask.shape[0]) % size), (0, (-mask.shape[1]) % size)))
        occupied = padded.reshape(padded.shape[0] // size, size,
                                  padded.shape[1] // size, size).any(axis=(1, 3))
        counts.append(int(occupied.sum()))
    counts = np.asarray(counts)
    eligible = counts > 1
    value = float("nan")
    if eligible.sum() >= 3:
        value = float(np.polyfit(np.log(mask.shape[1] / sizes[eligible]), np.log(counts[eligible]), 1)[0])
    return value, sizes, counts


def _lacunarity(mask: np.ndarray) -> tuple[float, int]:
    r = max(1, min(min(mask.shape), int(round(0.05 * mask.shape[1]))))
    background = (~mask).astype(float)
    integral = np.pad(background.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    masses = integral[r:, r:] - integral[:-r, r:] - integral[r:, :-r] + integral[:-r, :-r]
    mean = masses.mean()
    value = float(np.mean(masses ** 2) / mean ** 2) if mean > 0 else float("nan")
    return value, r


def _density_gradients(mask: np.ndarray) -> tuple[float, float, np.ndarray]:
    row_groups, col_groups = np.array_split(np.arange(mask.shape[0]), 3), np.array_split(np.arange(mask.shape[1]), 3)
    densities = np.array([[mask[np.ix_(ys, xs)].mean() for xs in col_groups] for ys in row_groups])
    xs = np.array([(g.mean() + 0.5) / mask.shape[1] for g in col_groups])
    ys = np.array([(g.mean() + 0.5) / mask.shape[0] for g in row_groups])
    ap = float(np.polyfit(xs, densities.mean(axis=0), 1)[0])
    dv = float(np.polyfit(ys, densities.mean(axis=1), 1)[0])
    return ap, dv, densities


def _pixel_graph(skeleton: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Eight-neighbor graph without redundant diagonal corner shortcuts.

    A diagonal edge is omitted when an orthogonal intermediate skeleton pixel
    already joins the same corners. Connectivity is retained, but digital
    right-angle bends do not become three-edge triangles or false Y nodes.
    """
    coords = np.argwhere(skeleton)
    index = np.full(skeleton.shape, -1, dtype=np.int32)
    if len(coords):
        index[coords[:, 0], coords[:, 1]] = np.arange(len(coords))
    h, w = skeleton.shape
    adjacency: list[list[int]] = [[] for _ in coords]
    for i, (y, x) in enumerate(coords):
        for dy, dx in _NEIGHBORS:
            yy, xx = int(y + dy), int(x + dx)
            if not (0 <= yy < h and 0 <= xx < w) or index[yy, xx] < 0:
                continue
            if dy and dx and (skeleton[y, xx] or skeleton[yy, x]):
                continue
            adjacency[i].append(int(index[yy, xx]))
    return coords, adjacency


def _components_in_subset(vertices: set[int], adjacency: list[list[int]]) -> int:
    remaining = set(vertices)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            for nxt in adjacency[stack.pop()]:
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _skeleton_graph(skeleton: np.ndarray) -> dict[str, Any]:
    coords, adjacency = _pixel_graph(skeleton)
    degrees = np.array([len(a) for a in adjacency])
    raw_neighbors = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - skeleton
    candidates = skeleton & (raw_neighbors >= 3)
    candidate_labels, candidate_count = ndi.label(candidates, structure=np.ones((3, 3)))
    by_candidate: dict[int, list[int]] = defaultdict(list)
    for i, (y, x) in enumerate(coords):
        if candidate_labels[y, x]:
            by_candidate[int(candidate_labels[y, x])].append(i)
    groups = []
    for ids in by_candidate.values():
        inside = set(ids)
        boundary = {j for i in ids for j in adjacency[i] if j not in inside}
        if _components_in_subset(boundary, adjacency) >= 3:
            groups.append(ids)
    membership = np.full(len(coords), -1, int)
    graph_coords = [coords[ids].mean(axis=0) for ids in groups]
    pixel_members = [list(ids) for ids in groups]
    for j, ids in enumerate(groups):
        membership[ids] = j
    for i in range(len(coords)):
        if membership[i] < 0:
            membership[i] = len(graph_coords)
            graph_coords.append(coords[i].astype(float))
            pixel_members.append([i])
    graph_coords_array = np.asarray(graph_coords, dtype=float).reshape((-1, 2))
    graph_adjacency = [set() for _ in graph_coords]
    length = 0.0
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            if j <= i:
                continue
            length += float(np.linalg.norm(coords[i] - coords[j]))
            a, b = int(membership[i]), int(membership[j])
            if a != b:
                graph_adjacency[a].add(b)
                graph_adjacency[b].add(a)
    junction_labels = np.zeros_like(candidate_labels)
    for j, ids in enumerate(groups, 1):
        junction_labels[coords[ids, 0], coords[ids, 1]] = j
    return {"pixel_coords": coords, "pixel_adjacency": adjacency, "pixel_degrees": degrees,
            "coords": graph_coords_array, "adjacency": graph_adjacency,
            "pixel_members": pixel_members, "junction_count": len(groups),
            "raw_junction_candidates": candidates, "raw_junction_cluster_count": int(candidate_count),
            "junction_labels": junction_labels, "skeleton_length_pixels": length}


def _segments(graph: dict) -> list[dict]:
    coords, adj = graph["coords"], graph["adjacency"]
    anchors = {i for i, neighbors in enumerate(adj) if len(neighbors) != 2}
    visited: set[tuple[int, int]] = set()
    result = []

    def edge(a: int, b: int) -> tuple[int, int]:
        return min(a, b), max(a, b)

    def walk(start: int, nxt: int) -> None:
        path = [start, nxt]
        visited.add(edge(start, nxt))
        previous, current = start, nxt
        while current not in anchors:
            options = [v for v in adj[current] if v != previous]
            if not options:
                break
            following = options[0]
            if edge(current, following) in visited:
                break
            visited.add(edge(current, following))
            path.append(following)
            previous, current = current, following
            if current == start:
                break
        points = coords[path]
        arc = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        chord = float(np.linalg.norm(points[-1] - points[0]))
        closed = path[-1] == path[0]
        result.append({"node_ids": path, "coords": points, "arc_length": arc,
                       "chord_length": chord, "closed": closed,
                       "tortuosity": float(arc / chord) if not closed and chord > 1e-12 else float("nan")})

    for start in sorted(anchors):
        for nxt in sorted(adj[start]):
            if edge(start, nxt) not in visited:
                walk(start, nxt)
    for start in range(len(adj)):
        for nxt in sorted(adj[start]):
            if edge(start, nxt) not in visited:
                walk(start, nxt)
    return result


def _prune_skeleton(skeleton: np.ndarray, edt_fg: np.ndarray) -> tuple[np.ndarray, int]:
    skeleton = skeleton.copy()
    removed_total = 0
    # Each iteration removes at least one pixel; usually a single pass suffices.
    for _ in range(32):
        graph = _skeleton_graph(skeleton)
        remove: set[int] = set()
        for branch in _segments(graph):
            if branch["closed"]:
                continue
            start, end = branch["node_ids"][0], branch["node_ids"][-1]
            ds, de = len(graph["adjacency"][start]), len(graph["adjacency"][end])
            if ds == 1 and de >= 3:
                junction = end
            elif de == 1 and ds >= 3:
                junction = start
            else:
                continue  # Never remove a complete endpoint-to-endpoint stripe.
            joint_pixels = graph["pixel_coords"][graph["pixel_members"][junction]]
            # A merged node includes off-center pixels. Its largest inscribed
            # radius measures attachment half-width without dilution by those
            # peripheral pixels (which otherwise leaves digital cap spurs).
            half_width = float(np.max(edt_fg[joint_pixels[:, 0], joint_pixels[:, 1]]))
            if branch["arc_length"] < PRUNE_HALF_WIDTH_MULTIPLIER * half_width:
                for node in branch["node_ids"]:
                    if node != junction:
                        remove.update(graph["pixel_members"][node])
        if not remove:
            break
        pixels = graph["pixel_coords"][sorted(remove)]
        skeleton[pixels[:, 0], pixels[:, 1]] = False
        removed_total += len(pixels)
    return skeleton, removed_total


def extract_features(mask: np.ndarray) -> tuple[dict, dict]:
    """Extract 18 dimensionless features and numeric/array diagnostics.

    ``mask`` must be an already cropped 2-D boolean or binary 0/1 array.
    Invalid input returns ``{feature: NaN}`` plus a readable diagnostic reason.
    Foreground uses 8-connectivity and background holes the dual 4-connectivity.
    Junction candidates are adjacent degree>=3 pixels merged into nodes; digital
    bend clusters without three external graph ports are discarded.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        return _invalid("Expected a two-dimensional foreground mask.", array.shape)
    if min(array.shape, default=0) < 3:
        return _invalid("Mask width and height must each be at least three pixels.", array.shape)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        return _invalid("Mask must contain only finite binary 0/1 values; preprocess the image first.", array.shape)
    mask = array.astype(bool, copy=True)
    if not mask.any():
        return _invalid("No foreground stripe pixels remain in the mask.", mask.shape)
    if mask.all():
        return _invalid("All pixels are foreground; no stripe/background contrast is available.", mask.shape)
    h, w = mask.shape
    features = {name: float("nan") for name in FEATURE_NAMES}
    features["melanistic_ratio"] = float(mask.mean())
    frequency, diagnostics = _frequency_features(mask)
    features.update(frequency)
    features["fractal_dimension"], box_sizes, box_counts = _fractal_dimension(mask)
    features["lacunarity"], lacunarity_box = _lacunarity(mask)
    features["ap_density_gradient"], features["dv_density_gradient"], density_grid = _density_gradients(mask)

    component_labels = label(mask, connectivity=2)
    regions = regionprops(component_labels)
    beta0 = len(regions)
    chi = int(euler_number(mask, connectivity=2))
    beta1 = int(beta0 - chi)
    holes = ndi.binary_fill_holes(mask, structure=ndi.generate_binary_structure(2, 1)) & ~mask
    hole_labels, hole_count = ndi.label(holes, structure=ndi.generate_binary_structure(2, 1))
    features["loop_to_stripe_ratio"] = float(beta1 / beta0)

    areas = np.array([r.area for r in regions], dtype=float)
    aspect = np.array([r.axis_major_length / r.axis_minor_length if r.axis_minor_length > 0 else np.inf for r in regions])
    stripe_areas = areas[aspect >= 2.0]
    median_stripe_area = float(np.median(stripe_areas if len(stripe_areas) else areas))
    major = [r for r in regions if r.area >= 0.25 * median_stripe_area]
    features["mean_dorsoventral_span"] = float(np.mean([(r.bbox[2] - r.bbox[0]) / h for r in major]))
    features["solidity_median"] = float(np.median([r.solidity for r in regions]))
    features["isolated_spot_fraction"] = float(np.mean((aspect < 2.0) & (areas < 0.25 * median_stripe_area)))

    # Padding foreground with white makes its crop-border convention explicit.
    edt_fg = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    edt_bg = ndi.distance_transform_edt(~mask)
    skeleton_unpruned = skeletonize(mask)
    skeleton, removed = _prune_skeleton(skeleton_unpruned, edt_fg)
    graph = _skeleton_graph(skeleton)
    branches = _segments(graph)
    widths = 2.0 * edt_fg[skeleton]
    if len(widths):
        median_width = float(np.median(widths))
        features["rel_width_median"] = median_width / w
        features["width_iqr_ratio"] = float(np.subtract(*np.percentile(widths, [75, 25])) / median_width)
    background_skeleton = skeletonize(~mask)
    yy, xx = np.indices(mask.shape)
    edge_distance = np.minimum.reduce([yy + 0.5, h - yy - 0.5, xx + 0.5, w - xx - 0.5])
    spacing_samples = background_skeleton & (edt_bg < edge_distance)
    if spacing_samples.any():
        features["rel_spacing_median"] = float(np.median(2 * edt_bg[spacing_samples]) / w)
    total_length = float(graph["skeleton_length_pixels"])
    junction_count = int(graph["junction_count"])
    features["bifurcation_rate"] = float(junction_count / (total_length / w)) if total_length > 0 else 0.0
    eligible = [b for b in branches if np.isfinite(b["tortuosity"]) and b["arc_length"] > 0]
    if eligible:
        features["mean_tortuosity"] = float(np.average([b["tortuosity"] for b in eligible], weights=[b["arc_length"] for b in eligible]))
    junctions = graph["coords"][:junction_count]
    endpoints = graph["coords"][[i for i, adj in enumerate(graph["adjacency"]) if len(adj) == 1]]
    warnings = []
    if hole_count != beta1:
        warnings.append("Euler hole count and dual-connectivity hole labeling differ; inspect the digital boundary.")
    if not eligible:
        warnings.append("No open skeleton branch has a nonzero chord; mean_tortuosity is undefined.")
    if not spacing_samples.any():
        warnings.append("No crop-interior background medial-axis sample is available; spacing is undefined.")
    if not np.isfinite(features["dominant_orientation_sin2"]):
        warnings.append("Axial orientation is undefined for an isotropic or uninformative spectrum.")
    diagnostics.update({
        "valid": True, "invalid_reason": None, "warnings": warnings,
        "feature_names": FEATURE_NAMES, "mask": mask, "normalization": {"width": w, "height": h, "area": w * h},
        "edt_fg": edt_fg, "edt_bg": edt_bg,
        "skeleton": skeleton, "skeleton_unpruned": skeleton_unpruned,
        "background_skeleton": background_skeleton, "spacing_sample_mask": spacing_samples,
        "junctions": junctions, "endpoints": endpoints,
        "junction_mask": graph["junction_labels"] > 0, "junction_labels": graph["junction_labels"],
        "raw_junction_candidate_mask": graph["raw_junction_candidates"],
        "raw_junction_cluster_count": graph["raw_junction_cluster_count"],
        "junction_count": junction_count, "endpoint_count": len(endpoints),
        "loops": holes, "hole_labels": hole_labels,
        "beta0": beta0, "beta1": beta1, "euler_characteristic": chi,
        "component_labels": component_labels,
        "major_component_labels": np.array([r.label for r in major], dtype=int),
        "median_stripe_area_pixels": median_stripe_area,
        "stripe_area_reference": "elongated components with moment-axis aspect>=2; all components if none",
        "skeleton_length_pixels": total_length, "skeleton_length_rel": total_length / w,
        "junctions_per_component": junction_count / beta0,
        "bifurcation_rate_definition": "J / (unique skeleton graph edge length in pixels / cropped width L)",
        "branch_prune_half_width_multiplier": PRUNE_HALF_WIDTH_MULTIPLIER,
        "pruned_skeleton_pixel_count": removed, "branches": branches,
        "closed_skeleton_segment_count": sum(b["closed"] for b in branches),
        "valid_tortuosity_branch_count": len(eligible),
        "box_counting_sizes_pixels": box_sizes, "box_counting_occupied_boxes": box_counts,
        "lacunarity_box_width_pixels": lacunarity_box, "density_grid_3x3": density_grid,
        "spacing_definition": "2*background EDT on background skeleton, excluding crop-boundary-censored samples",
    })
    assert tuple(features) == FEATURE_NAMES
    return features, diagnostics
