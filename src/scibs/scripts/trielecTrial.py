import argparse
from typing import Optional

import numpy as np

from scibs.utilities.file import read_tri, write_pot, write_tri
from scibs.utilities.graphics import visualize_mesh


def trielec(
    verts: np.ndarray,
    faces: np.ndarray,
    electrode_pts: np.ndarray,
    electrode_radii: np.ndarray,
):
    """
    Imprints electrode boundaries onto a surface mesh using a
    Vectorized Deterministic Cascade.

    Generates strict N=2 bisections for a .warp file without
    requiring a dynamic global adjacency graph.
    """

    print("=== TRIELEC STARTED ===", flush=True)
    print(f"verts shape: {verts.shape}", flush=True)
    print(f"faces shape: {faces.shape}", flush=True)
    print(f"electrode_pts shape: {electrode_pts.shape}", flush=True)
    print(f"electrode_radii shape: {electrode_radii.shape}", flush=True)

    warp_lines = []

    verts_list = verts.tolist()
    faces_list = faces.tolist()

    # ============================================================
    # PHASE 1: Vectorized Global Edge Hash
    # ============================================================

    print("\n=== PHASE 1: BUILDING EDGE HASH ===", flush=True)

    edges = np.vstack(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        )
    )

    print(f"Raw edges shape: {edges.shape}", flush=True)

    edges = np.sort(edges, axis=1)

    print("Edges sorted", flush=True)

    unique_edges = np.unique(edges, axis=0)

    print(
        f"Unique edges: {len(unique_edges)}",
        flush=True,
    )

    edge_to_new_v = {}
    moved_nodes = set()

    # ------------------------------------------------------------
    # Process electrodes
    # ------------------------------------------------------------

    print(
        f"\nNumber of electrodes: {len(electrode_pts)}",
        flush=True,
    )

    for el_idx, (center, radius) in enumerate(
        zip(electrode_pts, electrode_radii)
    ):

        print(
            f"\n--- Processing electrode {el_idx} ---",
            flush=True,
        )

        print(
            f"Center: {center}",
            flush=True,
        )

        print(
            f"Radius: {radius}",
            flush=True,
        )

        # Distance from every vertex to electrode surface
        print(
            "Calculating vertex distances...",
            flush=True,
        )

        dists = np.linalg.norm(
            verts - center,
            axis=1,
        ) - radius

        print(
            f"Distances calculated. Shape: {dists.shape}",
            flush=True,
        )

        d1 = dists[unique_edges[:, 0]]
        d2 = dists[unique_edges[:, 1]]

        print(
            "Edge distances calculated",
            flush=True,
        )

        # Only evaluate edges that physically cross
        # the electrode boundary
        crossings = (d1 < 0) != (d2 < 0)

        crossing_edges = unique_edges[crossings]
        crossing_d1 = d1[crossings]
        crossing_d2 = d2[crossings]

        print(
            f"Crossing edges: {len(crossing_edges)}",
            flush=True,
        )

        # --------------------------------------------------------
        # SLIVER PREVENTION
        # --------------------------------------------------------

        eps = 0.15

        print(
            f"Processing {len(crossing_edges)} crossing edges...",
            flush=True,
        )

        for i, (vA, vB) in enumerate(crossing_edges):

            # Progress message every 10,000 edges
            if i % 10000 == 0:
                print(
                    f"  Crossing edge {i}/{len(crossing_edges)}",
                    flush=True,
                )

            if (vA, vB) in edge_to_new_v:
                continue

            t = crossing_d1[i] / (
                crossing_d1[i] - crossing_d2[i]
            )

            p_new_calc = (
                verts[vA]
                + t * (verts[vB] - verts[vA])
            )

            if t < eps:

                if vA not in moved_nodes:

                    verts_list[vA] = p_new_calc.tolist()

                    warp_lines.append(
                        f"m {vA} "
                        f"{p_new_calc[0]:.15g} "
                        f"{p_new_calc[1]:.15g} "
                        f"{p_new_calc[2]:.15g}"
                    )

                    moved_nodes.add(vA)

            elif t > 1.0 - eps:

                if vB not in moved_nodes:

                    verts_list[vB] = p_new_calc.tolist()

                    warp_lines.append(
                        f"m {vB} "
                        f"{p_new_calc[0]:.15g} "
                        f"{p_new_calc[1]:.15g} "
                        f"{p_new_calc[2]:.15g}"
                    )

                    moved_nodes.add(vB)

            else:

                new_idx = len(verts_list)

                verts_list.append(
                    p_new_calc.tolist()
                )

                edge_to_new_v[(vA, vB)] = new_idx

                warp_lines.append(
                    f"a {new_idx} "
                    f"{p_new_calc[0]:.15g} "
                    f"{p_new_calc[1]:.15g} "
                    f"{p_new_calc[2]:.15g}"
                )

        print(
            f"Finished electrode {el_idx}",
            flush=True,
        )

        print(
            f"Total new vertices so far: "
            f"{len(verts_list) - len(verts)}",
            flush=True,
        )

    # ============================================================
    # PHASE 2: Isolated Deterministic Face Bisection
    # ============================================================

    print("\n=== PHASE 2: FACE BISECTION ===", flush=True)

    num_original_faces = len(faces)

    print(
        "what is this:",
        num_original_faces,
        flush=True,
    )

    print(
        f"Original faces: {num_original_faces}",
        flush=True,
    )

    print(
        f"Edges with new vertices: {len(edge_to_new_v)}",
        flush=True,
    )

    for i in range(num_original_faces):

        if i % 10000 == 0:
            print(
                f"Processing face {i}/{num_original_faces}",
                flush=True,
            )

        face = faces[i]

        v0, v1, v2 = face

        edges_of_face = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0))),
        ]

        splits_needed = [
            (edge, edge_to_new_v[edge])
            for edge in edges_of_face
            if edge in edge_to_new_v
        ]

        if not splits_needed:
            continue

        face_queue = [
            (i, face.tolist())
        ]

        for (e_vA, e_vB), P_new in splits_needed:

            for q_idx, (curr_f_idx, nodes) in enumerate(
                face_queue
            ):

                if e_vA in nodes and e_vB in nodes:

                    n0, n1, n2 = nodes

                    if (
                        (n0 == e_vA and n1 == e_vB)
                        or
                        (n0 == e_vB and n1 == e_vA)
                    ):

                        t1 = [
                            n0,
                            P_new,
                            n2,
                        ]

                        t2 = [
                            P_new,
                            n1,
                            n2,
                        ]

                    elif (
                        (n1 == e_vA and n2 == e_vB)
                        or
                        (n1 == e_vB and n2 == e_vA)
                    ):

                        t1 = [
                            n0,
                            n1,
                            P_new,
                        ]

                        t2 = [
                            n0,
                            P_new,
                            n2,
                        ]

                    else:

                        t1 = [
                            n0,
                            n1,
                            P_new,
                        ]

                        t2 = [
                            P_new,
                            n1,
                            n2,
                        ]

                    new_f_idx = len(faces_list)

                    warp_lines.append(
                        f"s {curr_f_idx} "
                        f"{n0} {n1} {n2} 2 "
                        f"{curr_f_idx} "
                        f"{t1[0]} {t1[1]} {t1[2]} "
                        f"{new_f_idx} "
                        f"{t2[0]} {t2[1]} {t2[2]}"
                    )

                    faces_list[curr_f_idx] = t1

                    faces_list.append(t2)

                    face_queue.pop(q_idx)

                    face_queue.append(
                        (
                            curr_f_idx,
                            t1,
                        )
                    )

                    face_queue.append(
                        (
                            new_f_idx,
                            t2,
                        )
                    )

                    break

    print(
        "Finished Phase 2",
        flush=True,
    )

    print(
        f"Final number of faces: {len(faces_list)}",
        flush=True,
    )

    # ============================================================
    # PHASE 3: Vectorized Post-Topology Centroid Labeling
    # ============================================================

    print(
        "\n=== PHASE 3: CENTROID LABELING ===",
        flush=True,
    )

    final_verts = np.array(
        verts_list
    )

    final_faces = np.array(
        faces_list
    )

    print(
        f"Final vertices: {final_verts.shape}",
        flush=True,
    )

    print(
        f"Final faces: {final_faces.shape}",
        flush=True,
    )

    face_labels = np.full(
        len(final_faces),
        -1,
        dtype=np.int32,
    )

    # Centroids only need to be calculated once
    centroids = final_verts[
        final_faces
    ].mean(axis=1)

    print(
        f"Centroids calculated: {centroids.shape}",
        flush=True,
    )

    for el_idx, (center, radius) in enumerate(
        zip(electrode_pts, electrode_radii)
    ):

        print(
            f"Labeling electrode {el_idx}",
            flush=True,
        )

        dists = np.linalg.norm(
            centroids - center,
            axis=1,
        )

        # --------------------------------------------------------
        # 1. Candidate faces
        # --------------------------------------------------------

        inside_mask = dists < (
            radius + 1e-5
        )

        candidate_indices = np.where(
            inside_mask
        )[0]

        print(
            f"Candidate faces: "
            f"{len(candidate_indices)}",
            flush=True,
        )

        if len(candidate_indices) == 0:
            continue

        # --------------------------------------------------------
        # 2. Local Adjacency Hash
        # --------------------------------------------------------

        edge_to_faces = {}

        for f_idx in candidate_indices:

            face = final_faces[f_idx]

            for j in range(3):

                edge = tuple(
                    sorted(
                        (
                            face[j],
                            face[(j + 1) % 3],
                        )
                    )
                )

                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []

                edge_to_faces[edge].append(
                    f_idx
                )

        print(
            f"Adjacency edges: "
            f"{len(edge_to_faces)}",
            flush=True,
        )

        # --------------------------------------------------------
        # 3. Seed Identification
        # --------------------------------------------------------

        seed_idx = candidate_indices[
            np.argmin(
                dists[inside_mask]
            )
        ]

        print(
            f"Seed face: {seed_idx}",
            flush=True,
        )

        # --------------------------------------------------------
        # 4. Region Grow BFS
        # --------------------------------------------------------

        visited = set(
            [seed_idx]
        )

        queue = [
            seed_idx
        ]

        while queue:

            curr_idx = queue.pop(0)

            face = final_faces[curr_idx]

            for j in range(3):

                edge = tuple(
                    sorted(
                        (
                            face[j],
                            face[(j + 1) % 3],
                        )
                    )
                )

                for neighbor_idx in edge_to_faces.get(
                    edge,
                    [],
                ):

                    if neighbor_idx not in visited:

                        visited.add(
                            neighbor_idx
                        )

                        queue.append(
                            neighbor_idx
                        )

        print(
            f"Region size: {len(visited)} faces",
            flush=True,
        )

        # --------------------------------------------------------
        # 5. Label Assignment
        # --------------------------------------------------------

        for f_idx in visited:
            face_labels[f_idx] = el_idx

    print(
        "\n=== TRIELEC FINISHED ===",
        flush=True,
    )

    print(
        f"Final vertices: {len(final_verts)}",
        flush=True,
    )

    print(
        f"Final faces: {len(final_faces)}",
        flush=True,
    )

    print(
        f"Warp commands: {len(warp_lines)}",
        flush=True,
    )

    print(
        f"Labeled faces: "
        f"{np.sum(face_labels >= 0)}",
        flush=True,
    )

    return (
        final_verts,
        final_faces,
        face_labels,
        warp_lines,
    )


def main(
    elec_descr_in: str,
    tri_in: str,
    tri_out: str,
    warp_name: Optional[str] = None,
    plot: bool = False,
):

    print("\n=== MAIN STARTED ===", flush=True)

    print(
        f"Electrode file: {elec_descr_in}",
        flush=True,
    )

    print(
        f"Input TRI: {tri_in}",
        flush=True,
    )

    print(
        f"Output TRI: {tri_out}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Read TRI
    # ------------------------------------------------------------

    print(
        "\nReading TRI file...",
        flush=True,
    )

    tri_verts, tri_ids = read_tri(
        tri_in
    )

    print(
        f"TRI vertices: {tri_verts.shape}",
        flush=True,
    )

    print(
        f"TRI faces: {tri_ids.shape}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Read electrode description
    # ------------------------------------------------------------

    print(
        "\nReading electrode description...",
        flush=True,
    )

    electrode = np.loadtxt(
        elec_descr_in,
        skiprows=1,
        usecols=(1, 2, 3, 4),
        ndmin=2,
        comments=["height", "n layers"],
    )

    print(
        f"Electrode array shape: {electrode.shape}",
        flush=True,
    )

    electrode_pts, electrode_radii = np.split(
        electrode,
        [3],
        axis=1,
    )

    electrode_radii = electrode_radii.flatten()

    print(
        f"Electrode points:\n{electrode_pts}",
        flush=True,
    )

    print(
        f"Electrode radii:\n{electrode_radii}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Run TRIELEC
    # ------------------------------------------------------------

    print(
        "\nCalling trielec()...",
        flush=True,
    )

    (
        tri_verts,
        tri_ids,
        face_labels,
        warp_lines,
    ) = trielec(
        tri_verts,
        tri_ids,
        electrode_pts,
        electrode_radii,
    )

    print(
        "\nReturned from trielec()",
        flush=True,
    )

    # ------------------------------------------------------------
    # Write TRI
    # ------------------------------------------------------------

    print(
        f"Writing TRI file: {tri_out}",
        flush=True,
    )

    write_tri(
        tri_out,
        tri_verts,
        tri_ids,
    )

    print(
        "TRI file written",
        flush=True,
    )

    # ------------------------------------------------------------
    # Write WARP
    # ------------------------------------------------------------

    if warp_name:

        print(
            f"Writing WARP file: {warp_name}",
            flush=True,
        )

        with open(
            warp_name,
            "w",
        ) as f:

            f.write(
                "\n".join(warp_lines)
                + "\n"
            )

        print(
            "WARP file written",
            flush=True,
        )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    if plot:

        print(
            "Visualizing mesh...",
            flush=True,
        )

        visualize_mesh(
            tri_verts,
            tri_ids,
            face_labels,
            electrode_pts,
        )

    print(
        "\n=== MAIN FINISHED ===",
        flush=True,
    )


if __name__ == "__main__":
    print(
        "=== SCRIPT STARTED ===",
        flush=True,
    )

    # Your argparse section should call main().