def init_blackboard():
    return {
        "status": "initialized",
        "aligned_data": None,
        "reconstructed_data": None,
        "raw_features": None,
        "validated_features": None
    }

def agent_parsing_and_alignment(image_tensor, hrnet_weights):
    keypoints = hrnet_weights.process(image_tensor)
    foreground_mask = keypoints.generate_mask()
    aligned_tensor = image_tensor * foreground_mask
    return aligned_tensor

def agent_stripe_reconstruction(aligned_tensor, tps_grid):
    warped_tensor = tps_grid.apply_transform(aligned_tensor)
    binarized_texture = warped_tensor.threshold()
    return binarized_texture

def agent_feature_extraction(binarized_texture):
    skeletons = binarized_texture.zhang_suen_thinning()
    nodes = skeletons.find_topological_nodes()
    features = {
        "y_shapes": nodes.get_connectivity(3),
        "terminations": nodes.get_connectivity(1),
        "o_shapes": nodes.get_closed_loops()
    }
    return features

def agent_review_and_correction(raw_features, gemini_client, system_instruction):
    payload = {
        "instruction": system_instruction,
        "vision_features": raw_features
    }
    vlm_response = gemini_client.invoke(payload)
    return vlm_response.extract_validated_nodes()

def run_openstripe_mas(image_tensor, hrnet_weights, tps_grid, gemini_client, system_instruction):
    bb = init_blackboard()
    
    bb["aligned_data"] = agent_parsing_and_alignment(image_tensor, hrnet_weights)
    bb["status"] = "alignment_complete"
    
    bb["reconstructed_data"] = agent_stripe_reconstruction(bb["aligned_data"], tps_grid)
    bb["status"] = "reconstruction_complete"
    
    bb["raw_features"] = agent_feature_extraction(bb["reconstructed_data"])
    bb["status"] = "extraction_complete"
    
    bb["validated_features"] = agent_review_and_correction(bb["raw_features"], gemini_client, system_instruction)
    bb["status"] = "correction_complete"
    
    return bb["validated_features"]
