pub struct TileGrid {
    // Key: (Grid X, Grid Y), Value: List of Cell Indices in that tile
    pub bins: HashMap<(i32, i32), Vec<usize>>,
    pub tile_size: f32,
}

impl TileGrid {
    pub fn build(nodes: &[CellNode], interaction_radius: f32) -> Self {
        // The tile size should exactly match your maximum interaction radius.
        // This guarantees we only ever need to check the home tile and the 8 surrounding tiles.
        let tile_size = interaction_radius;
        let mut bins: HashMap<(i32, i32), Vec<usize>> = HashMap::new();

        for (i, node) in nodes.iter().enumerate() {
            let grid_x = (node.x / tile_size).floor() as i32;
            let grid_y = (node.y / tile_size).floor() as i32;

            bins.entry((grid_x, grid_y)).or_insert_with(Vec::new).push(i);
        }

        Self { bins, tile_size }
    }
}