# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# def inspect_binary_tensor(bin_file, tile_index=0, obs_dim=185):
#     print(f"Loading tensor from {bin_file}...")
#
#     # 1. Load the raw binary data as 32-bit floats
#     # Note: numpy defaults to the system's byte order, which perfectly
#     # matches Rust's bytemuck::cast_slice default behavior.
#     raw_data = np.fromfile(bin_file, dtype=np.float32)
#
#     # 2. File Size Sanity Check
#     total_floats = len(raw_data)
#     if total_floats % obs_dim != 0:
#         print(f"🚨 CRITICAL ERROR: File contains {total_floats} floats, "
#               f"which is not divisible by {obs_dim}.")
#         print("Your Rust saving function is either truncating data or adding padding!")
#         return
#
#     num_tiles = total_floats // obs_dim
#     print(f"✅ Successfully decoded {num_tiles} tiles.")
#
#     if tile_index >= num_tiles:
#         print(f"🚨 ERROR: Requested tile {tile_index}, but only {num_tiles} exist.")
#         return
#
#     # 3. Reshape and extract the target tile
#     tensor = raw_data.reshape((num_tiles, obs_dim))
#     tile = tensor[tile_index]
#
#     # 4. Slice the 185-dimensional vector into its biological components
#     ripley = tile[0:20]
#     enrichment = tile[20:56].reshape((6, 6))
#     halo = tile[56:184].reshape((4, 32))  # 4 edges, 32 bins each
#     cell_count = tile[184]
#
#     print(f"\n--- TILE {tile_index} BIOLOGICAL SUMMARY ---")
#     print(f"Total Cells Detected: {cell_count}")
#     print(f"Halo Boundary Activations (Top, Right, Bottom, Left): "
#           f"{np.sum(halo[0]):.0f}, {np.sum(halo[1]):.0f}, "
#           f"{np.sum(halo[2]):.0f}, {np.sum(halo[3]):.0f}")
#
#     # 5. Visual Diagnostics
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#
#     # Plot A: Ripley's L-Curve
#     # Expected: Should rise above 0 initially (clustering), then dip.
#     # If it is a perfectly straight line at negative values, the tile is empty.
#     r_vals = np.arange(1.5, 30.1, 1.5)
#     ax1.plot(r_vals, ripley, marker='o', color='crimson', linewidth=2)
#     ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
#     ax1.set_title(f"Ripley's L-Curve (Spatial Clustering)", fontsize=14)
#     ax1.set_xlabel("Radius (µm)")
#     ax1.set_ylabel("L(r) - r")
#     ax1.grid(True, alpha=0.3)
#
#     # Plot B: Adjacency Enrichment Heatmap
#     # Expected: High values along the diagonal (auto-adjacency) for structural cells.
#     # If it is totally blank, ID clamping or spatial hashing failed.
#     sns.heatmap(enrichment, annot=True, fmt=".0f", cmap="viridis",
#                 cbar_kws={'label': 'Adjacent Pairs'}, ax=ax2)
#     ax2.set_title("Cell-Type Adjacency Matrix", fontsize=14)
#     ax2.set_xlabel("Cell Type B")
#     ax2.set_ylabel("Cell Type A")
#
#     plt.tight_layout()
#     plt.show()
#
#
# # --- RUN THE TEST ---
# # Point this at your generated binary file and pick a tile near the center
# inspect_binary_tensor("test.bin", tile_index=271)
#
# # import numpy as np
#
#
# def sweep_binary_tensor(bin_file, obs_dim=185):
#     print(f"Loading tensor from {bin_file}...")
#
#     # Load the raw binary data
#     raw_data = np.fromfile(bin_file, dtype=np.float32)
#
#     total_floats = len(raw_data)
#     num_tiles = total_floats // obs_dim
#     print(f"✅ Successfully decoded {num_tiles} tiles.")
#
#     tensor = raw_data.reshape((num_tiles, obs_dim))
#
#     total_cells_detected = 0
#     non_empty_tiles = 0
#     densest_tile_idx = -1
#     max_cells_in_a_tile = -1
#
#     # Sweep all tiles
#     for idx, tile in enumerate(tensor):
#         cell_count = tile[184]  # The 185th float is our global count
#
#         if cell_count > 0:
#             total_cells_detected += int(cell_count)
#             non_empty_tiles += 1
#
#             if cell_count > max_cells_in_a_tile:
#                 max_cells_in_a_tile = int(cell_count)
#                 densest_tile_idx = idx
#
#     print("\n--- 🧹 SWEEP RESULTS ---")
#     print(f"Total Cells Detected Across All Tiles: {total_cells_detected}")
#     print(f"Number of Tiles Containing Tissue: {non_empty_tiles} / {num_tiles}")
#     print(f"Number of Empty Glass Tiles: {num_tiles - non_empty_tiles}")
#
#     if densest_tile_idx != -1:
#         print(f"\n🔥 The Densest Tile is Tile #{densest_tile_idx} with {max_cells_in_a_tile} cells.")
#         print(f"Run inspect_binary_tensor('test.bin', tile_index={densest_tile_idx}) to see the biology!")
#
#
# # Run the sweep
# # sweep_binary_tensor("test.bin")


import numpy as np


def extract_top_10_tiles(bin_file, obs_dim=185):
    print(f"Loading tensor from {bin_file}...\n")

    # Load and reshape
    raw_data = np.fromfile(bin_file, dtype=np.float32)
    num_tiles = len(raw_data) // obs_dim
    tensor = raw_data.reshape((num_tiles, obs_dim))

    # Store tile data: (index, cell_count, ripley, enrichment)
    tile_stats = []

    for idx, tile in enumerate(tensor):
        cell_count = tile[184]
        if cell_count > 0:
            ripley = tile[0:20]
            # Reshape the 36 floats back into your 6x6 Broad ID matrix
            enrichment = tile[20:56].reshape((6, 6))
            tile_stats.append((idx, int(cell_count), ripley, enrichment))

    # Sort by cell count descending
    tile_stats.sort(key=lambda x: x[1], reverse=True)
    top_10 = tile_stats[:10]

    print("--- 🔬 TOP 10 DENSEST TILES FOR AI ANALYSIS ---\n")
    for rank, (idx, count, ripley, enrichment) in enumerate(top_10):
        print(f"### Rank {rank + 1}: Tile {idx} ({count} cells) ###")

        # Round Ripley's values so they don't take up massive screen space
        rounded_ripley = [round(val, 2) for val in ripley]
        print(f"Ripley's L: {rounded_ripley}")

        print("Adjacency Matrix:")
        for row in enrichment:
            # Format matrix rows into clean, aligned columns
            print(" [" + ", ".join([f"{int(val):>4}" for val in row]) + "]")
        print("-" * 50 + "\n")


# Run it!
extract_top_10_tiles("test.bin")