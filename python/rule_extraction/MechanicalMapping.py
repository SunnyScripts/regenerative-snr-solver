import json


def generate_mechanical_mapping(input_json_path, output_json_path):
    with open(input_json_path, 'r') as iFile:
        class_mapping = json.load(iFile)

        # REFINED RULES: TICKET TO THE FINISH LINE
        mechanical_rules = {
            # 5: Anomalous / Cancer
            5: ["cancer", "tumor", "malignant", "carcinoma", "melanoma", "sarcoma", "neoplastic", "abnormal"],

            # 4: Immune / Fluid
            4: ["killer", "helper", "tc1", "antibody", "blood", "marrow", "erythro", "lymph",
                "myeloid", "progenitor", "precursor", "reticulocyte", "plasmablast", "platelet",
                "spermat", "oocyte", "blast", "innate", "pbmc", "t cell", "b cell", "be cell",
                "dendritic", "macrophage", "neutrophil", "monocyte", "leukocyte", "lymphocyte",
                "mast cell", "langerhans", "plasma", "nk cell", "thymocyte", "ilc", "microglia",
                "basophil", "eosinophil", "megakaryocyte", "mononuclear", "phagocyte", "granulocyte",
                "hematopoietic", "centroblast", "centrocyte", "promyelocyte", "myelocyte",
                "hofbauer", "kupffer", "natural killer", "t regulatory", "t-regulatory", "pre-b",
                "antigen presenting", "inflammatory"],

            # 3: Endothelial
            3: ["endothelial", "capillary", "vascular", "vein", "artery", "lymphatic", "endocardial", "tip cell"],

            # 1: Epithelial (Barrier/Sheet)
            1: ["epithelial", "keratinocyte", "basal", "squamous", "goblet", "enterocyte",
                "hepatocyte", "luminal", "ciliated", "club cell", "pneumocyte", "colonocyte",
                "acinar", "urothelial", "lens", "podocyte", "paneth", "parietal", "alveolar",
                "ionocyte", "trophoblast", "merkel", "foveolar", "peptic", "chief", "ductal",
                "sebocyte", "sebaceous", "keratocyte", "m cell", "pancreatic", "epsilon", "endocrine",
                "secretory", "absorptive", "barrier", "hillock", "serous", "mucus", "mucous",
                "cholangiocyte", "ependymal", "transit amplifying", "umbrella", "deuterosomal",
                "gip cell", "p/d1", "lactocyte", "brush cell", "intercalated", "peg cell",
                "gland", "glandular", "epiderm", "urothelium", "mesothelial", "principal cell",
                "papillary", "periderm", "receptor", "dark cell", "follicular", "chromaffin",
                "endoderm", "pp cell", "kidney", "microfold"],

            # 2: Mesenchymal / Structural (Scaffold)
            2: ["fibro", "stromal", "muscle", "adipocyte", "chondro", "osteo", "pericyte", "melanocyte",
                "mesenchymal", "myo", "schwann", "neuron", "glial", "astrocyte", "oligodendrocyte",
                "amacrine", "bipolar", "cone", "rod", "photoreceptor", "ganglion", "stellate",
                "mesangial", "myoid", "tendon", "odontoblast", "perineurial", "leptomeningeal",
                "paraxial", "decidua", "mural", "adventitial", "mesenchyme", "cajal", "purkinje",
                "horizontal", "leydig", "sertoli", "theca", "granulosa", "supporting", "pineal",
                "chandelier", "neuroblast", "glioblast", "granule", "interstitial", "connective",
                "contractile", "mueller", "offx", "trabecular meshwork", "cortical cell of adrenal",
                "collagen", "extracellular matrix", "reticular", "endosteal", "eurydendroid",
                "neural", "neurectoderm", "neuroplacodal", "noradrenergic", "mesoderm",
                "hypothalamus", "retinal"]
        }

        mechanical_mapping = {}
        unmapped = []
        # Find the maximum type_id to size our array
        max_id = max(int(k) for k in class_mapping.keys())
        mechanical_array = [0] * (max_id + 1)

        # Mapping Loop
        for type_id, cell_name in class_mapping.items():
            name_lower = cell_name.lower()
            assigned_mech_id = 0

            # We run the check in a specific order to prioritize function
            # 5 -> 4 -> 3 -> 1 -> 2
            for mech_id in [5, 4, 3, 1, 2]:
                keywords = mechanical_rules[mech_id]
                if any(keyword in name_lower for keyword in keywords):
                    assigned_mech_id = mech_id
                    break

            mechanical_mapping[type_id] = assigned_mech_id

            # Don't track "unknown" or generic "cell" as an unmapped error
            if assigned_mech_id == 0 and name_lower not in ["unknown", "cell", "cell in vitro"]:
                unmapped.append(cell_name)

            # Insert directly into the array at the correct index
            mechanical_array[int(type_id)] = assigned_mech_id

        # Save as a flat list: [4, 4, 1, 0, ...]
        with open(output_json_path, 'w') as oFile:
            json.dump(mechanical_array, oFile)

        print(f"✅ Success! Mapped the majority of {len(class_mapping)} cells.")
        print(f"Left {len(unmapped)} biological 'edge cases' as Unknown (0).")
        if unmapped:
            print("Final Unmapped Samples:", unmapped)

generate_mechanical_mapping("../Network/models/class_mapping.json", "../Network/models/mechanical_mapping.json")