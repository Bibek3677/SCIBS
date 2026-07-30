"""Map electrode surface labels onto the volumetric mesh nodes.

``trielec`` produces a per-face electrode label (``face_labels``) over the
*surface* triangulation.  ``tetwarp`` then imprints those boundaries into the
volumetric tetrahedral mesh and returns ``el_map`` (``new_el_map`` in the
pipeline), which maps a surface-node index onto its volumetric-node index.

``elecpotsurf`` combines the two: it produces a dense per-node array over the
volumetric mesh where each node that belongs to an electrode patch carries that
electrode's id and every other node is ``-1`` (background).  This is the
"surface potential" field consumed by :func:`visualize_potentials` and written
alongside the mesh output.
"""

import numpy as np


def elecpotsurf(
    tri_ids: np.ndarray,
    face_labels: np.ndarray,
    n_nodes: int,
    el_map: np.ndarray,
    background: int = -1,
) -> np.ndarray:
    """Project surface electrode labels onto volumetric nodes.

    Parameters
    ----------
    tri_ids:
        ``(F, 3)`` surface triangles, indexed into the *surface* node set
        (i.e. the faces returned by ``trielec``).
    face_labels:
        ``(F,)`` electrode id per surface triangle; ``-1`` for non-electrode
        faces.
    n_nodes:
        Number of nodes in the volumetric mesh (``len(tet_pts)`` after warping).
    el_map:
        ``(P,)`` mapping from surface-node index to volumetric-node index
        (``new_el_map`` from ``tetwarp``).
    background:
        Value assigned to nodes that do not belong to any electrode.

    Returns
    -------
    np.ndarray
        ``(n_nodes,)`` integer array of electrode ids.
    """
    tri_ids = np.asarray(tri_ids)
    face_labels = np.asarray(face_labels)
    el_map = np.asarray(el_map)

    pots = np.full(int(n_nodes), background, dtype=np.int64)

    labeled = face_labels >= 0
    if not labeled.any():
        return pots

    # Surface-node ids of every labeled face -> volumetric-node ids.
    surf_nodes = tri_ids[labeled]              # (L, 3) surface indices
    vol_nodes = el_map[surf_nodes]             # (L, 3) volumetric indices
    labels = np.repeat(face_labels[labeled], 3)

    flat_nodes = vol_nodes.reshape(-1)

    # Guard against a malformed el_map pointing past the volumetric node set.
    in_range = flat_nodes < len(pots)
    if not in_range.all():
        dropped = int((~in_range).sum())
        print(
            f"Warning: elecpotsurf dropped {dropped} node label(s) whose "
            f"volumetric index exceeded n_nodes={n_nodes}."
        )

    pots[flat_nodes[in_range]] = labels[in_range]
    return pots
