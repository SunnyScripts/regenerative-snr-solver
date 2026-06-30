import struct
import json
import numpy as np
import os


def verify_binaries():
    print("🔬 RUNNING BINARY DIAGNOSTICS...\n")

    # 1. Load the JSON map to know our expected matrix dimensions
    if not os.path.exists("../../models/rna_cell_classifier/Dualv2/granular_class_map.json"):
        print("❌ Cannot find granular_class_map.json. Run preprocessor first.")
        return

    with open("../../models/rna_cell_classifier/Dualv2/granular_class_map.json", "r") as f:
        gran_array = json.load(f)
    num_gran = len(gran_array)

    print(f"📌 Expected Matrix Dimensions: {num_gran} x {num_gran} ({num_gran ** 2} floats)")
    expected_bytes = (num_gran ** 2) * 4  # 4 bytes per float32

    # ==========================================
    # 2. CHECK 1D/2D MATRICES (Numpy)
    # ==========================================
    print("\n--- MATRICES ---")

    def check_matrix(filename):
        if not os.path.exists(filename):
            print(f"❌ Missing: {filename}")
            return

        file_size = os.path.getsize(filename)
        if file_size != expected_bytes:
            print(f"❌ SIZE MISMATCH in {filename}: Expected {expected_bytes} bytes, got {file_size}")
            return

        # Read the raw bytes back into a 2D float array
        mat = np.fromfile(filename, dtype=np.float32).reshape((num_gran, num_gran))
        print(f"✅ {filename}: Shape {mat.shape}, Max Value: {mat.max():.4f}, Min Value: {mat.min():.4f}")

    def check_1d(filename):
        if not os.path.exists(filename):
            print(f"❌ Missing: {filename}")
            return

        arr = np.fromfile(filename, dtype=np.float32)
        print(f"✅ {filename}: Length {len(arr)}, Sample [0]: {arr[0]:.4f}")

    check_matrix("adhesion_matrix.bin")
    check_matrix("conductance_matrix.bin")
    check_1d("ideal_depths.bin")
    check_1d("strat_weights.bin")

    # ==========================================
    # 3. CHECK THE COMPLEX STRUCT
    # ==========================================
    print("\n--- SDE INTERACTOME ---")
    interactome_file = "gpu_interactome.bin"

    if not os.path.exists(interactome_file):
        print(f"❌ Missing: {interactome_file}")
        return

    with open(interactome_file, "rb") as f:
        # Read the first 256 bytes (16 RuleOffsets * 16 bytes)
        offset_data = f.read(256)
        if len(offset_data) < 256:
            print("❌ gpu_interactome.bin is too small! Offset header corrupted.")
            return

        print("✅ Successfully read 256-byte Offset Header.")

        total_rules_expected = 0
        for i in range(16):
            chunk = offset_data[i * 16: (i + 1) * 16]
            # '<IIII' = Little-Endian, 4 Unsigned Integers (start, count, pad, pad)
            start_idx, count, pad1, pad2 = struct.unpack('<IIII', chunk)

            if count > 0:
                print(f"   Broad Pair {i:02d}: Starts at index {start_idx:03d}, Contains {count:02d} rules.")
                total_rules_expected += count

        # Read the remaining bytes (The actual GpuInteractomeRules)
        rule_data = f.read()
        bytes_remaining = len(rule_data)
        rules_found = bytes_remaining // 16

        print(f"\n✅ Extracted {rules_found} rules ({bytes_remaining} bytes).")

        if rules_found != total_rules_expected:
            print(f"❌ MISMATCH: Header expected {total_rules_expected} rules, but found {rules_found}.")

        if rules_found > 0:
            # Unpack the very first rule to verify the float conversion worked
            # '<IffI' = Little-Endian, 1 Uint, 2 Floats, 1 Uint
            first_rule = struct.unpack('<IffI', rule_data[:16])
            print(f"✅ Sample Rule 0 -> Target Granular ID: {first_rule[0]}")
            print(f"                    Total Drift:      {first_rule[1]:.6f}")
            print(f"                    Total Diffusion:  {first_rule[2]:.6f}")


if __name__ == "__main__":
    verify_binaries()