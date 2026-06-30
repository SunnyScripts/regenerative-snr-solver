import struct
import numpy as np

def squash_interactome_drift(scale_factor=100.0):
    filename = "gpu_interactome.bin"
    print(f"🔧 Squashing Drift in {filename}...")

    with open(filename, "rb") as f:
        data = f.read()

    # The first 256 bytes are the offsets (untouched)
    offset_bytes = data[:256]
    rule_data = data[256:]

    num_rules = len(rule_data) // 16
    new_rule_bytes = bytearray()

    max_original_drift = 0.0

    for i in range(num_rules):
        chunk = rule_data[i*16 : (i+1)*16]
        # Unpack the rule: Target ID (u32), Drift (f32), Diffusion (f32), Pad (u32)
        target_id, drift, diff, pad = struct.unpack('<IffI', chunk)

        if drift > max_original_drift:
            max_original_drift = drift

        # THE SQUASH: You can use division, log1p, or whatever fits your SDE math
        # Dividing by 100 turns a 1061 drift into 10.61 (perfect for Gumbel noise)
        squashed_drift = drift / scale_factor

        # Repack the rule with the new drift
        new_rule_bytes += struct.pack('<IffI', target_id, squashed_drift, diff, pad)

    # Write the safely squashed binary back to disk
    with open("gpu_interactome_squashed.bin", "wb") as f:
        f.write(offset_bytes)
        f.write(new_rule_bytes)

    print(f"✅ Success! Max Original Drift was {max_original_drift:.2f}.")
    print(f"   Max New Drift is {max_original_drift / scale_factor:.2f}.")
    print("   Saved to gpu_interactome_squashed.bin")

if __name__ == "__main__":
    # max_allowed_drift = 20.0
    # scale_factor = max_original_drift / max_allowed_drift
    #
    # # Now apply this factor to every rule:
    # squashed_drift = drift / scale_factor

    squash_interactome_drift(scale_factor=100.0)