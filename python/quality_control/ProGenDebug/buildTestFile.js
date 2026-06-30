const fs = require('fs');

const nodes = [];
const cryptRadius = 60;
const layerSpacing = 12;

console.log("Generating 3D epithelial crypt...");

// Build 3 distinct layers (Basal, Spinous, Granular)
for (let layer = 0; layer < 3; layer++) {
    const cellType = layer + 1; // Types 1, 2, and 3
    const zOffset = layer * layerSpacing;

    // Spin around the center in a circle
    for (let angle = 0; angle < Math.PI * 2; angle += 0.1) {

        // Expand outwards from the center
        for (let r = 0; r < cryptRadius; r += 4) {

            // Mathematical parabola to create the "bowl" / crypt invagination
            const baseZ = (r * r) / 40;

            // Convert polar to Cartesian coordinates
            const x = Math.cos(angle) * r;
            const y = Math.sin(angle) * r;
            const z = baseZ + zOffset;

            // Add organic jitter so it looks like real biology, not a perfect grid
            const jitter = () => (Math.random() - 0.5) * 5.0;

            nodes.push({
                x: +(x + jitter()).toFixed(2),
                y: +(y + jitter()).toFixed(2),
                z: +(z + jitter()).toFixed(2),
                type: cellType
            });
        }
    }
}

// Write to the exact file your HTML viewer is looking for
const filename = 'test_tissue_01.json';
fs.writeFileSync(filename, JSON.stringify(nodes, null, 2));

console.log(`Success! Saved ${nodes.length} cells to ${filename}.`);
console.log("Start your local server (e.g., npx serve) and open the HTML viewer to see it.");