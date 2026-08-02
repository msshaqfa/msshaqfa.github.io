import os, json, sys, glob, re
import numpy as np
import meshio, trimesh

def convert(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    # Find VTU files
    vtu_files = sorted(glob.glob(os.path.join(src_dir, "*.vtu")))
    if not vtu_files:
        print(f"No .vtu files found in {src_dir}")
        return

    # Detect common prefix (e.g., "disk_morph_opt_stage_" or "shape_morph_opt_stage_")
    names = [os.path.basename(f) for f in vtu_files]
    prefix = os.path.commonprefix(names).rstrip("_0123456789")
    print(f"Found {len(vtu_files)} VTU files, prefix: '{prefix}'")

    all_data = []
    for path in vtu_files:
        m = meshio.read(path)
        pts = m.points.astype(np.float32)
        cells = m.cells_dict.get("triangle")
        if cells is None:
            cells = m.cells_dict.get("tetra")
            if cells is not None:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                cells = hull.simplices
        if cells is None:
            print(f"  WARNING: no triangle/tetra cells in {os.path.basename(path)}, skipping")
            continue

        ts = {"pts": pts, "cells": cells, "arrays": {}}
        for name in ["actuation", "bending_moments", "membrane_forces"]:
            arr = m.point_data.get(name)
            if arr is None: continue
            ts["arrays"][name] = np.linalg.norm(arr, axis=1) if arr.ndim > 1 else arr
        all_data.append(ts)

    if not all_data:
        print("No valid meshes found")
        return

    data_json = {"fields": {}, "timesteps": []}

    # Global ranges
    for name in ["actuation", "bending_moments", "membrane_forces"]:
        vals = [ts["arrays"][name] for ts in all_data if name in ts["arrays"]]
        if vals:
            vmin = min(arr.min() for arr in vals)
            vmax = max(arr.max() for arr in vals)
            data_json["fields"][name] = {"min": float(vmin), "max": float(vmax)}

    # Bounds
    all_pts = np.concatenate([ts["pts"] for ts in all_data])
    data_json["bounds"] = {"min": all_pts.min(axis=0).tolist(), "max": all_pts.max(axis=0).tolist()}

    # Generate GLBs
    for i, ts in enumerate(all_data):
        tm = trimesh.Trimesh(vertices=ts["pts"], faces=ts["cells"])
        glb_name = f"stage_{i+1:02d}.glb"
        tm.export(os.path.join(dst_dir, glb_name))
        entry = {"file": glb_name, "vertices": len(ts["pts"]), "tris": len(ts["cells"])}
        for name, arr in ts["arrays"].items():
            entry[name] = arr.astype(np.float32).tolist()
        data_json["timesteps"].append(entry)
        print(f"  {glb_name} — {len(ts['pts'])}v, {len(ts['cells'])}t")

    with open(os.path.join(dst_dir, "data.json"), "w") as f:
        json.dump(data_json, f)

    total = sum(os.path.getsize(os.path.join(dst_dir, e["file"])) for e in data_json["timesteps"])
    print(f"\nDone: {len(all_data)} GLBs ({total/1024:.0f} KB) + data.json → {dst_dir}/")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        convert(sys.argv[1], sys.argv[2])
    else:
        # Default: convert all _morph folders in assets/data/
        for name in sorted(os.listdir("assets/data")):
            src = os.path.join("assets/data", name)
            if os.path.isdir(src) and name.endswith("_morph"):
                dst = os.path.join("assets/data", name + "_glb")
                print(f"\n=== Converting {name} → {name}_glb ===")
                convert(src, dst)
