import argparse
import os
from typing import Optional
import numpy as np
from scibs.utilities.file import read_tri, write_pot, write_tri
from scibs.utilities.graphics import visualize_mesh


def load_electrode_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Robustly parses electrode text files while skipping variable headers."""
    valid_rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Expecting 5 numbers: [id, x, y, z, radius]
            if len(parts) == 5:
                try:
                    nums = [float(p) for p in parts]
                    valid_rows.append(nums[1:])  # Keep x, y, z, radius
                except ValueError:
                    continue

    if not valid_rows:
        raise ValueError(f"Could not parse valid electrode data from {filepath}")

    data = np.array(valid_rows)
    electrode_pts = data[:, :3]
    electrode_radii = data[:, 3]
    return electrode_pts, electrode_radii


def trielec(verts: np.ndarray, faces: np.ndarray, electrode_pts: np.ndarray, electrode_radii: np.ndarray):
    """
    Imprints electrode boundaries onto a surface mesh using a Vectorized Deterministic Cascade.
    Generates strict N=2 bisections for a .warp file without requiring a dynamic global adjacency graph.
    """
    warp_lines = []
    verts_list = verts.tolist()
    faces_list = faces.tolist()

    # --- PHASE 1: Vectorized Global Edge Hash ---
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique_edges = np.unique(edges, axis=0)

    # DEGENERATE VERTEX GUARD
    # A single corrupted/outlier vertex (e.g. a placeholder coordinate baked into
    # the source mesh) creates edges dozens of times longer than the rest of the
    # mesh. Treating those as legitimate electrode-boundary crossings makes the
    # sliver-prevention snap below drag real scalp vertices toward the bad point,
    # punching holes in otherwise round electrodes. Drop such edges from crossing
    # detection entirely so a corrupted vertex can't distort its real neighbors.
    edge_lengths = np.linalg.norm(verts[unique_edges[:, 0]] - verts[unique_edges[:, 1]], axis=1)
    median_edge_len = np.median(edge_lengths)
    degenerate_mask = edge_lengths > 20 * median_edge_len
    if degenerate_mask.any():
        # The corrupted vertex is the one shared by every bad edge (its neighbors
        # each show up on only one bad edge, since only their shared edge with the
        # corrupted vertex is abnormally long).
        touched, counts = np.unique(unique_edges[degenerate_mask].flatten(), return_counts=True)
        hub_verts = touched[counts > 1].tolist()
        print(
            f"WARNING: Ignoring {degenerate_mask.sum()} degenerate mesh edge(s) "
            f"(>20x the median edge length of {median_edge_len:.4f}). "
            f"This usually means the input mesh has corrupted vertex position(s) "
            f"-- check vertex indices {hub_verts} in the source .tri file."
        )
        unique_edges = unique_edges[~degenerate_mask]

    edge_to_new_v = {}
    moved_nodes = set()

    for el_idx, (center, radius) in enumerate(zip(electrode_pts, electrode_radii)):
        print(f"\n--- Processing electrode {el_idx + 1} ---")
        print(f"Center: [{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}]")
        print(f"Radius: {radius}")
        print("Calculating vertex distances...")

        dists = np.linalg.norm(verts - center, axis=1) - radius
        print(f"Distances calculated. Shape: {dists.shape}")
        print("Edge distances calculated")

        d1 = dists[unique_edges[:, 0]]
        d2 = dists[unique_edges[:, 1]]
        
        # Only evaluate edges that physically cross the boundary threshold
        crossings = (d1 < 0) != (d2 < 0)
        
        crossing_edges = unique_edges[crossings]
        crossing_d1 = d1[crossings]
        crossing_d2 = d2[crossings]

        num_crossings = len(crossing_edges)
        print(f"Crossing edges: {num_crossings}")
        print(f"Processing {num_crossings} crossing edges...")
        if num_crossings > 0:
            print(f"  Crossing edge 0/{num_crossings}")

        # SLIVER PREVENTION (Epsilon Snapping)
        # Matches the MINLM2 = 0.15 threshold from the original C++ code.
        eps = 0.15 

        for i, (vA, vB) in enumerate(crossing_edges):
            if (vA, vB) in edge_to_new_v:
                continue 

            t = crossing_d1[i] / (crossing_d1[i] - crossing_d2[i])
            p_new_calc = verts[vA] + t * (verts[vB] - verts[vA])

            if t < eps:
                if vA not in moved_nodes:
                    verts_list[vA] = p_new_calc.tolist()
                    warp_lines.append(f"m {vA} {p_new_calc[0]:.15g} {p_new_calc[1]:.15g} {p_new_calc[2]:.15g}")
                    moved_nodes.add(vA)
            elif t > 1.0 - eps:
                if vB not in moved_nodes:
                    verts_list[vB] = p_new_calc.tolist()
                    warp_lines.append(f"m {vB} {p_new_calc[0]:.15g} {p_new_calc[1]:.15g} {p_new_calc[2]:.15g}")
                    moved_nodes.add(vB)
            else:
                new_idx = len(verts_list)
                verts_list.append(p_new_calc.tolist())
                edge_to_new_v[(vA, vB)] = new_idx
                warp_lines.append(f"a {new_idx} {p_new_calc[0]:.15g} {p_new_calc[1]:.15g} {p_new_calc[2]:.15g}")

        print(f"Finished electrode {el_idx + 1}")
        print(f"Total new vertices so far: {len(edge_to_new_v)}")

    # --- PHASE 2: Isolated Deterministic Face Bisection ---
    num_original_faces = len(faces)
    print("\n=== PHASE 2: FACE BISECTION ===")
    print(f"what is this: {num_original_faces}")
    print(f"Original faces: {num_original_faces}")
    print(f"Edges with new vertices: {len(edge_to_new_v)}")

    for i in range(num_original_faces):
        if i == 0 or i == 10000:
            print(f"Processing face {i}/{num_original_faces}")

        face = faces[i]
        v0, v1, v2 = face
        edges_of_face = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]

        splits_needed = [(edge, edge_to_new_v[edge]) for edge in edges_of_face if edge in edge_to_new_v]

        if not splits_needed:
            continue

        face_queue = [(i, face.tolist())]

        for (e_vA, e_vB), P_new in splits_needed:
            for q_idx, (curr_f_idx, nodes) in enumerate(face_queue):
                if e_vA in nodes and e_vB in nodes:
                    n0, n1, n2 = nodes

                    if (n0 == e_vA and n1 == e_vB) or (n0 == e_vB and n1 == e_vA):
                        t1 = [n0, P_new, n2]
                        t2 = [P_new, n1, n2]
                    elif (n1 == e_vA and n2 == e_vB) or (n1 == e_vB and n2 == e_vA):
                        t1 = [n0, n1, P_new]
                        t2 = [n0, P_new, n2]
                    else:
                        t1 = [n0, n1, P_new]
                        t2 = [P_new, n1, n2]

                    new_f_idx = len(faces_list)

                    warp_lines.append(
                        f"s {curr_f_idx} {n0} {n1} {n2} 2 "
                        f"{curr_f_idx} {t1[0]} {t1[1]} {t1[2]} "
                        f"{new_f_idx} {t2[0]} {t2[1]} {t2[2]}"
                    )

                    faces_list[curr_f_idx] = t1
                    faces_list.append(t2)
                    
                    face_queue.pop(q_idx)
                    face_queue.append((curr_f_idx, t1))
                    face_queue.append((new_f_idx, t2))
                    break

    print("phase 2 ends here **__**__")
    print(f"total faces: {len(faces_list)}")

    # --- PHASE 3: Vectorized Post-Topology Centroid Labeling ---
    print("\n centroid labeling third phase")
    final_verts = np.array(verts_list)
    final_faces = np.array(faces_list)
    print(f"Final vertices: {final_verts.shape}")
    print(f"Final faces: {final_faces.shape}")

    face_labels = np.full(len(final_faces), -1, dtype=np.int32)

    centroids = final_verts[final_faces].mean(axis=1)
    print(f"Centroids calculated: {centroids.shape}")

    for el_idx, (center, radius) in enumerate(zip(electrode_pts, electrode_radii)):
        electrode_label = el_idx + 1
        print(f"Labeling electrode {electrode_label}")
        dists = np.linalg.norm(centroids - center, axis=1)

        # 1. Candidate faces
        # Phase 1/2 already cut every boundary-crossing edge exactly at the
        # sphere, so a correctly-split face should never straddle it. Checking
        # the CENTROID distance instead of every vertex lets a triangle with one
        # genuinely-outside vertex sneak in whenever the other two vertices sit
        # close enough to the boundary to pull the average back under the
        # threshold -- that produces a thin spike poking out of the disc.
        vert_dists = np.linalg.norm(final_verts - center, axis=1)
        inside_mask = np.all(vert_dists[final_faces] < (radius + 1e-5), axis=1)
        candidate_indices = np.where(inside_mask)[0]
        print(f"Candidate faces: {len(candidate_indices)}")
        
        if len(candidate_indices) == 0:
            continue

        # 2. Local Adjacency Hash
        edge_to_faces = {}
        for f_idx in candidate_indices:
            face = final_faces[f_idx]
            for i in range(3):
                edge = tuple(sorted((face[i], face[(i+1)%3])))
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(f_idx)

        print(f"Adjacency edges: {len(edge_to_faces)}")

        # 3. Seed Identification
        seed_idx = candidate_indices[np.argmin(dists[inside_mask])]
        print(f"Seed face: {seed_idx}")

        # 4. Region-Grow (BFS)
        visited = set([seed_idx])
        queue = [seed_idx]

        while queue:
            curr_idx = queue.pop(0)
            face = final_faces[curr_idx]

            for i in range(3):
                edge = tuple(sorted((face[i], face[(i+1)%3])))
                for neighbor_idx in edge_to_faces.get(edge, []):
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)

        print(f"Region size: {len(visited)} faces")

        # 5. Label Assignment
        for f_idx in visited:
            face_labels[f_idx] = electrode_label

    labeled_count = np.sum(face_labels != -1)
    print("\n=== TRIELEC FINISHED ===")
    print(f"Final vertices: {len(final_verts)}")
    print(f"Final faces: {len(final_faces)}")
    print(f"Warp commands: {len(warp_lines)}")
    print(f"Labeled faces: {labeled_count}")

    # Force direct extraction dump to C:\S.R\SCIBS\extraction_trielec.txt during function execution
    extraction_path = r"C:\S.R\SCIBS\extraction_trielec.txt"
    print(f"Dumping extraction data directly to: {extraction_path}")
    with open(extraction_path, "w") as f:
        f.write("=== FINAL VERTICES ===\n")
        np.savetxt(f, final_verts, fmt="%.15g")

        f.write("\n=== FINAL FACES ===\n")
        np.savetxt(f, final_faces, fmt="%d")

        f.write("\n=== FACE LABELS ===\n")
        np.savetxt(f, face_labels.reshape(-1, 1), fmt="%d")

        f.write("\n=== WARP LINES ===\n")
        for line in warp_lines:
            f.write(line + "\n")

    return final_verts, final_faces, face_labels, warp_lines


def main(elec_descr_in: str, tri_in: str, tri_out: str, warp_name: Optional[str] = None, plot: bool = False):
    tri_verts, tri_ids = read_tri(tri_in)
    electrode_pts, electrode_radii = load_electrode_file(elec_descr_in)

    tri_verts, tri_ids, face_labels, warp_lines = trielec(
        tri_verts, tri_ids, electrode_pts, electrode_radii
    )

    write_tri(tri_out, tri_verts, tri_ids)

    if warp_name:
        with open(warp_name, "w") as f:
            f.write("\n".join(warp_lines) + "\n")

    if plot:
        visualize_mesh(tri_verts, tri_ids, face_labels, electrode_pts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imprint electrode geometry onto surface mesh.")
    parser.add_argument("elec_descr_in", type=str, help="Path to electrode description file")
    parser.add_argument("tri_in", type=str, help="Path to input .tri surface mesh")
    parser.add_argument("tri_out", type=str, help="Path for output .tri surface mesh")
    parser.add_argument("--warp_name", type=str, default=None, help="Path to output .warp file")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot interactive 3D mesh")

    args = parser.parse_args()
    main(
        elec_descr_in=args.elec_descr_in,
        tri_in=args.tri_in,
        tri_out=args.tri_out,
        warp_name=args.warp_name,
        plot=args.plot
    )