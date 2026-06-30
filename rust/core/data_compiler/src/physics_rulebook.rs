#[derive(Deserialize)]
struct RawCellRules {
    ideal_depth: f32,
    stratification_weight: f32,
}

#[derive(Deserialize)]
struct RawAdhesion {
    type_a: String,
    type_b: String,
    penalty: f32,
}

#[derive(Deserialize)]
struct RawPhysicsRulebook {
    cell_types: HashMap<String, RawCellRules>,
    adhesion_rules: Vec<RawAdhesion>, // MRF Cell-Type rules
    gap_junction_genes: Vec<String>,
    // adhesion_pairs: Vec<(String, String)> // Gene-Level Ligand/Receptor pairs
}

// ---------------------------------------------------------
// 2. COMPILED STRUCT (What the MRF actually uses)
// ---------------------------------------------------------
#[derive(Clone, Default)]
struct PhysicsRulebook {
    ideal_depths: Vec<f32>,
    strat_weights: Vec<f32>,
    adhesion_matrix: Vec<f32>,
    num_types: usize,
    gap_junction_genes: Vec<usize>,
    // adhesion_pairs: Vec<(usize, usize)>
}

impl PhysicsRulebook {
    // This replaces your old 'load_physics_rulebook' function
    fn compile(path: &str, type_map: &HashMap<String, u32>, gene_map: &HashMap<String, usize>) -> Self {
        let file_content = fs::read_to_string(path).expect("Failed to read rulebook");
        let raw: RawPhysicsRulebook = serde_json::from_str(&file_content).expect("Invalid JSON");

        let num_types = type_map.len();

        // Initialize flat arrays filled with defaults
        let mut ideal_depths = vec![0.0; num_types];
        let mut strat_weights = vec![0.0; num_types];
        let mut adhesion_matrix = vec![0.0; num_types * num_types];

        // Compile Cell Properties
        for (name, rules) in raw.cell_types {
            if let Some(&type_id) = type_map.get(&name) {
                let idx = type_id as usize;
                ideal_depths[idx] = rules.ideal_depth;
                strat_weights[idx] = rules.stratification_weight;
            }
        }

        // Compile Adhesion Rules
        for rule in raw.adhesion_rules {
            if let (Some(&id_a), Some(&id_b)) = (type_map.get(&rule.type_a), type_map.get(&rule.type_b)) {
                let idx_a = id_a as usize;
                let idx_b = id_b as usize;

                // Write to both sides of the symmetric matrix
                adhesion_matrix[(idx_a * num_types) + idx_b] = rule.penalty;
                adhesion_matrix[(idx_b * num_types) + idx_a] = rule.penalty;
            }
        }

        let mut gap_junction_genes = Vec::new();
        for gene_name in raw.gap_junction_genes {
            if let Some(&gene_idx) = gene_map.get(&gene_name) {
                gap_junction_genes.push(gene_idx);
            } else {
                // Good practice to log this, in case the JSON asks for a gene
                // that doesn't exist in this specific Xenium panel.
                // println!("Warning: Gap junction gene '{}' not found in dataset.", gene_name);
            }
        }

        // let mut adhesion_pairs = Vec::new();
        // for (ligand, receptor) in raw.adhesion_pairs {
        //     // We only push the pair if BOTH genes exist in this specific CSV dataset
        //     if let (Some(&l_idx), Some(&r_idx)) = (gene_map.get(&ligand), gene_map.get(&receptor)) {
        //         adhesion_pairs.push((l_idx, r_idx));
        //     } else {
        //         println!("Warning: Ligand/Receptor pair ('{}', '{}') not found in dataset.", ligand, receptor);
        //     }
        // }

        PhysicsRulebook {
            ideal_depths,
            strat_weights,
            adhesion_matrix,
            num_types,
            gap_junction_genes,
            // adhesion_pairs
        }
    }

    // ---------------------------------------------------------
    // 3. THE GETTERS (Called by execute_mrf)
    // ---------------------------------------------------------
    #[inline(always)]
    fn get_ideal_depth(&self, type_id: u32) -> f32 {
        self.ideal_depths[type_id as usize]
    }

    #[inline(always)]
    fn get_stratification_weight(&self, type_id: u32) -> f32 {
        self.strat_weights[type_id as usize]
    }

    #[inline(always)]
    fn get_adhesion(&self, type_a: u32, type_b: u32) -> f32 {
        let idx = (type_a as usize * self.num_types) + (type_b as usize);
        self.adhesion_matrix[idx]
    }
}