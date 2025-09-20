#!/usr/bin/env python3
"""
Analyze exact position correspondence between BB and BT logical Z operators.
This script finds all exact matches between blocks and reports the specific 
qubit positions that correspond between the two codes.
"""

import numpy as np
from pathlib import Path
from bivariate_bicycle_codes import get_BB_Hx_Hz
from bivariate_tricycle_codes import get_BT_Hx_Hz
from bposd.css import css_code


def generate_logical_z_operators(code_type):
    """Generate logical Z operators from parity check matrices using bposd css_code."""
    print(f"Generating {code_type} logical Z operators...")
    
    if code_type == 'BB':
        # BB code parameters: l=6, m=6, polynomials from standard construction  
        l, m = 6, 6
        a_poly = [[3, 0], [0, 1], [0, 2]]  # x^3 + y + y^2
        b_poly = [[0, 3], [1, 0], [2, 0]]  # y^3 + x + x^2
        
        Hx, Hz = get_BB_Hx_Hz(a_poly, b_poly, l, m)
        print(f"BB code: Hx shape {Hx.shape}, Hz shape {Hz.shape}")
        
        # Create CSS code and extract logical Z operators
        code = css_code(Hx, Hz)
        code.test()  # Initialize the code
        print(f"BB Code: N={code.N}, K={code.K}")
        
        # Extract logical Z operators (lz attribute)
        logical_z = code.lz
        
    elif code_type == 'BT':
        # BT code parameters: l=6, m=6, polynomials from standard construction
        l, m = 6, 6
        a_poly = [[3, 0], [0, 1], [0, 2]]  # x^3 + y + y^2 (same as BB)
        b_poly = [[0, 3], [1, 0], [2, 0]]  # y^3 + x + x^2 (same as BB)
        c_poly = [[1, 0], [0, 2]]          # x + y^2 (from working config)
        
        Hx, Hz = get_BT_Hx_Hz(a_poly, b_poly, c_poly, l, m)
        print(f"BT code: Hx shape {Hx.shape}, Hz shape {Hz.shape}")
        
        # Create CSS code and extract logical Z operators
        code = css_code(Hx, Hz)
        code.test()  # Initialize the code
        print(f"BT Code: N={code.N}, K={code.K}")
        
        # Extract logical Z operators (lz attribute)
        logical_z = code.lz
        
    else:
        raise ValueError("code_type must be 'BB' or 'BT'")
    
    print(f"Generated {logical_z.shape[0]} logical Z operators of length {logical_z.shape[1]}")
    return logical_z

def load_logical_z_operators(code_type):
    """Load logical Z operators for BB or BT code, generating if needed."""
    if code_type == 'BB':
        file_path = Path('BB_logical_z_operators.npy')
        expected_shape = (12, 72)  # 12 operators, 72 qubits each
        block_size = 36
        num_blocks = 2
    elif code_type == 'BT':
        file_path = Path('BT_logical_z_operators.npy')
        expected_shape = (6, 108)  # 6 operators, 108 qubits each  
        block_size = 36
        num_blocks = 3
    else:
        raise ValueError("code_type must be 'BB' or 'BT'")
    
    if not file_path.exists():
        print(f"{file_path} not found. Generating logical Z operators...")
        operators = generate_logical_z_operators(code_type)
        # Save as dense array to avoid pickle issues
        if hasattr(operators, 'toarray'):
            operators = operators.toarray()
        operators = np.asarray(operators, dtype=np.uint8)
        np.save(file_path, operators)
        print(f"Saved to {file_path}")
    else:
        try:
            operators = np.load(file_path, allow_pickle=False)
            print(f"Loaded {code_type} logical Z operators: shape {operators.shape}")
        except:
            print(f"Error loading {file_path}, regenerating...")
            operators = generate_logical_z_operators(code_type)
            if hasattr(operators, 'toarray'):
                operators = operators.toarray()
            operators = np.asarray(operators, dtype=np.uint8)
            np.save(file_path, operators)
            print(f"Regenerated and saved to {file_path}")
    
    if operators.shape != expected_shape:
        print(f"Warning: Expected shape {expected_shape}, got {operators.shape}")
    
    return operators, block_size, num_blocks

def split_into_blocks(operators, block_size, num_blocks):
    """Split operators into blocks."""
    num_ops = operators.shape[0]
    blocks = []
    
    for i in range(num_ops):
        op_blocks = []
        for j in range(num_blocks):
            start_idx = j * block_size
            end_idx = (j + 1) * block_size
            block = operators[i, start_idx:end_idx]
            op_blocks.append(block)
        blocks.append(op_blocks)
    
    return blocks

def find_nonzero_positions(block):
    """Find positions of nonzero elements in a block."""
    block_array = np.asarray(block).flatten()
    return set(np.where(block_array != 0)[0])

def analyze_block_correspondence(bb_blocks, bt_blocks):
    """Find all exact block correspondences between BB and BT operators."""
    correspondences = []
    
    print("\n=== EXACT BLOCK CORRESPONDENCES ===")
    
    for bb_idx in range(len(bb_blocks)):
        for bt_idx in range(len(bt_blocks)):
            matches = []
            
            # Check each BB block against each BT block
            for bb_block_idx in range(len(bb_blocks[bb_idx])):
                bb_block = bb_blocks[bb_idx][bb_block_idx]
                bb_positions = find_nonzero_positions(bb_block)
                
                for bt_block_idx in range(len(bt_blocks[bt_idx])):
                    bt_block = bt_blocks[bt_idx][bt_block_idx]
                    bt_positions = find_nonzero_positions(bt_block)
                    
                    # Check for exact match
                    try:
                        match_condition = (bb_positions == bt_positions and len(bb_positions) > 0)
                    except Exception as e:
                        print(f"Error comparing positions: {e}")
                        print(f"bb_positions type: {type(bb_positions)}, bt_positions type: {type(bt_positions)}")
                        continue
                    
                    if match_condition:
                        matches.append({
                            'bb_block': bb_block_idx,
                            'bt_block': bt_block_idx,
                            'positions': [int(pos) for pos in sorted(bb_positions)],
                            'weight': len(bb_positions)
                        })
            
            if matches:
                correspondences.append({
                    'bb_op': bb_idx,
                    'bt_op': bt_idx,
                    'matches': matches
                })
    
    return correspondences

def print_detailed_correspondences(correspondences):
    """Print detailed correspondence information."""
    
    print(f"\nFound {len(correspondences)} operator pairs with exact block matches:")
    print("=" * 80)
    
    # Group by correspondence type
    perfect_matches = []  # Both blocks match
    partial_matches = []  # Some blocks match
    zero_matches = []     # Only zero blocks match
    
    for corr in correspondences:
        bb_idx = corr['bb_op']
        bt_idx = corr['bt_op']
        matches = corr['matches']
        
        # Check if this is a perfect match (multiple blocks match)
        non_zero_matches = [m for m in matches if m['weight'] > 0]
        zero_matches_only = [m for m in matches if m['weight'] == 0]
        
        if len(non_zero_matches) >= 2:
            perfect_matches.append(corr)
        elif len(non_zero_matches) == 1:
            partial_matches.append(corr)
        else:
            zero_matches.append(corr)
    
    # Print perfect matches first
    if perfect_matches:
        print("\n🎯 PERFECT MATCHES (Multiple blocks match):")
        print("-" * 50)
        for corr in perfect_matches:
            bb_idx = corr['bb_op']
            bt_idx = corr['bt_op']
            matches = corr['matches']
            
            print(f"\nBB_{bb_idx} ↔ BT_{bt_idx}:")
            for match in matches:
                if match['weight'] > 0:
                    bb_block = match['bb_block']
                    bt_block = match['bt_block'] 
                    positions = match['positions']
                    weight = match['weight']
                    
                    print(f"  Block {bb_block} (BB) = Block {bt_block} (BT)")
                    print(f"    Weight: {weight}")
                    print(f"    Positions: {positions}")
    
    # Print partial matches
    if partial_matches:
        print(f"\n📍 PARTIAL MATCHES (Single block matches):")
        print("-" * 50)
        for corr in partial_matches:
            bb_idx = corr['bb_op']
            bt_idx = corr['bt_op']
            matches = [m for m in corr['matches'] if m['weight'] > 0]
            
            print(f"\nBB_{bb_idx} ↔ BT_{bt_idx}:")
            for match in matches:
                bb_block = match['bb_block']
                bt_block = match['bt_block']
                positions = match['positions']
                weight = match['weight']
                
                print(f"  Block {bb_block} (BB) = Block {bt_block} (BT)")
                print(f"    Weight: {weight}")
                print(f"    Positions: {positions}")

def analyze_weight_patterns(bb_blocks, bt_blocks):
    """Analyze weight patterns of logical operators."""
    
    print("\n=== WEIGHT PATTERN ANALYSIS ===")
    
    print("\nBB Logical Z Operators:")
    for i, blocks in enumerate(bb_blocks):
        weights = [int(np.sum(block != 0)) for block in blocks]
        total_weight = sum(weights)
        print(f"BB_{i}: blocks={weights}, total={total_weight}")
    
    print("\nBT Logical Z Operators:")
    for i, blocks in enumerate(bt_blocks):
        weights = [int(np.sum(block != 0)) for block in blocks]
        total_weight = sum(weights)
        print(f"BT_{i}: blocks={weights}, total={total_weight}")

def position_to_grid_coord(pos, l=6, m=6):
    """Convert linear position to (row, col) grid coordinate."""
    return (pos // m, pos % m)

def print_logical_operators_grid_coordinates():
    """Print all logical Z operators with their 6x6 grid coordinates."""
    print("\n" + "="*80)
    print("COMPLETE LOGICAL Z OPERATORS IN 6×6 GRID COORDINATES")
    print("="*80)
    
    # Load operators
    bb_ops, bb_block_size, bb_num_blocks = load_logical_z_operators('BB')
    bt_ops, bt_block_size, bt_num_blocks = load_logical_z_operators('BT')
    
    # Split into blocks
    bb_blocks = split_into_blocks(bb_ops, bb_block_size, bb_num_blocks)
    bt_blocks = split_into_blocks(bt_ops, bt_block_size, bt_num_blocks)
    
    print("\n📋 BB CODE LOGICAL Z OPERATORS (2 blocks × 6×6 each):")
    print("-" * 70)
    
    for i, blocks in enumerate(bb_blocks):
        print(f"\nBB_{i}:")
        for block_idx, block in enumerate(blocks):
            block_array = np.asarray(block).flatten()
            nonzero_positions = np.where(block_array != 0)[0]
            
            if len(nonzero_positions) > 0:
                grid_coords = [(int(position_to_grid_coord(pos)[0]), int(position_to_grid_coord(pos)[1])) for pos in nonzero_positions]
                print(f"  Block {block_idx}: {grid_coords}")
            else:
                print(f"  Block {block_idx}: [] (all zeros)")
    
    print(f"\n📋 BT CODE LOGICAL Z OPERATORS (3 blocks × 6×6 each):")
    print("-" * 70)
    
    for i, blocks in enumerate(bt_blocks):
        print(f"\nBT_{i}:")
        for block_idx, block in enumerate(blocks):
            block_array = np.asarray(block).flatten()
            nonzero_positions = np.where(block_array != 0)[0]
            
            if len(nonzero_positions) > 0:
                grid_coords = [(int(position_to_grid_coord(pos)[0]), int(position_to_grid_coord(pos)[1])) for pos in nonzero_positions]
                print(f"  Block {block_idx}: {grid_coords}")
            else:
                print(f"  Block {block_idx}: [] (all zeros)")

def gf2_rref(matrix):
    """Compute reduced row echelon form over GF(2)."""
    A = matrix.copy().astype(np.uint8)
    rows, cols = A.shape
    
    lead = 0
    for r in range(rows):
        if lead >= cols:
            break
            
        # Find pivot
        i = r
        while A[i, lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if lead == cols:
                    return A % 2  # Ensure GF(2) on return
        
        # Swap rows
        if i != r:
            A[[i, r]] = A[[r, i]]
        
        # Eliminate column using GF(2) arithmetic
        for i in range(rows):
            if i != r and A[i, lead] == 1:
                A[i] = (A[i] ^ A[r]) % 2  # Critical: mod 2 for GF(2)
        
        lead += 1
    
    return A % 2  # Ensure final result is in GF(2)

def embed_bb_in_bt_space(bb_logical, block_positions, block_size=36):
    """Embed BB logical operator (2 blocks) into BT space (3 blocks) at specified positions.
    
    Args:
        bb_logical: BB logical operator (72 qubits = 2 blocks of 36)
        block_positions: tuple of 2 block indices (e.g., (0,1) for blocks 1&2, (0,2) for blocks 1&3, (1,2) for blocks 2&3)
        block_size: size of each block (36 for 6x6 grid)
    
    Returns:
        BT-space logical operator (108 qubits = 3 blocks of 36)
    """
    # Split BB logical into 2 blocks
    bb_block1 = bb_logical[:block_size]
    bb_block2 = bb_logical[block_size:]
    
    # Create 3-block BT space (all zeros initially)
    bt_space = np.zeros(3 * block_size, dtype=np.uint8)
    
    # Place BB blocks at specified positions
    pos1, pos2 = block_positions
    bt_space[pos1*block_size:(pos1+1)*block_size] = bb_block1
    bt_space[pos2*block_size:(pos2+1)*block_size] = bb_block2
    
    return bt_space

def check_logical_equivalence_modulo_stabilizers(bb_logical, bt_logical, bb_hz, bt_hz, embedding_blocks=(0,1)):
    """Check if BB and BT logical operators are equivalent modulo their Z stabilizers.
    
    Args:
        embedding_blocks: which blocks to embed BB operator in BT space (0,1)=blocks 1&2, (0,2)=blocks 1&3, (1,2)=blocks 2&3
    """
    # Embed BB operator in BT space at specified block positions
    bb_in_bt_space = embed_bb_in_bt_space(bb_logical, embedding_blocks)
    
    # Now both operators are in the same space (108 qubits)
    bt_padded = bt_logical.copy()
    
    # Compute difference
    logical_diff = (bb_in_bt_space ^ bt_padded).astype(np.uint8)
    
    # Check if difference is zero (identical operators)
    if np.all(logical_diff == 0):
        return True, "identical", 0, embedding_blocks
    
    # Pad BB Hz to BT space (108 qubits)
    bb_hz_in_bt_space = np.zeros((bb_hz.shape[0], 108), dtype=np.uint8)
    pos1, pos2 = embedding_blocks
    bb_hz_in_bt_space[:, pos1*36:(pos1+1)*36] = bb_hz[:, :36]  # First block of BB Hz
    bb_hz_in_bt_space[:, pos2*36:(pos2+1)*36] = bb_hz[:, 36:]  # Second block of BB Hz
    
    # Combine all Z stabilizers in BT space
    combined_hz = np.concatenate([bb_hz_in_bt_space, bt_hz], axis=0)
    
    # Remove zero rows
    combined_hz = combined_hz[~np.all(combined_hz == 0, axis=1)]
    
    if combined_hz.shape[0] == 0:
        return False, "no_stabilizers", 0, embedding_blocks
    
    # Check if logical_diff is in span of combined_hz
    original_rank = np.linalg.matrix_rank(gf2_rref(combined_hz))
    
    # Augment with logical difference
    augmented = np.concatenate([combined_hz, logical_diff.reshape(1, -1)], axis=0)
    augmented_rank = np.linalg.matrix_rank(gf2_rref(augmented))
    
    # If ranks are equal, logical_diff is in the span (equivalent modulo stabilizers)
    equivalent = (original_rank == augmented_rank)
    
    return equivalent, "computed", augmented_rank - original_rank, embedding_blocks

def verify_logical_equivalence_all_pairs():
    """Verify logical equivalence for all BB-BT pairs modulo stabilizers with multiple block embeddings."""
    print("\n" + "="*80)
    print("LOGICAL EQUIVALENCE VERIFICATION (MODULO Z STABILIZERS)")
    print("="*80)
    print("Testing BB operators embedded in ALL possible 2-block positions within BT 3-block space")
    print("Block positions: (0,1)=blocks 1&2, (0,2)=blocks 1&3, (1,2)=blocks 2&3")
    
    # Load operators and codes
    bb_ops, _, _ = load_logical_z_operators('BB')  
    bt_ops, _, _ = load_logical_z_operators('BT')
    
    # Get stabilizer matrices
    bb_a_poly = [[3, 0], [0, 1], [0, 2]]
    bb_b_poly = [[0, 3], [1, 0], [2, 0]]
    bb_hx, bb_hz = get_BB_Hx_Hz(bb_a_poly, bb_b_poly, 6, 6)
    
    bt_a_poly = [[3, 0], [0, 1], [0, 2]]
    bt_b_poly = [[0, 3], [1, 0], [2, 0]]  
    bt_c_poly = [[1, 0], [0, 2]]
    bt_hx, bt_hz = get_BT_Hx_Hz(bt_a_poly, bt_b_poly, bt_c_poly, 6, 6)
    
    print(f"\nBB stabilizers Hz: {bb_hz.shape}")
    print(f"BT stabilizers Hz: {bt_hz.shape}")
    
    # Track results for all embedding positions
    all_embeddings = [(0,1), (0,2), (1,2)]
    embedding_names = ["blocks 1&2", "blocks 1&3", "blocks 2&3"]
    
    results_by_embedding = {}
    
    for embed_idx, embedding in enumerate(all_embeddings):
        print(f"\n" + "="*60)
        print(f"TESTING EMBEDDING: BB → BT {embedding_names[embed_idx]} {embedding}")
        print("="*60)
        
        equivalent_pairs = []
        identical_pairs = []
        
        for bb_idx in range(bb_ops.shape[0]):
            bb_logical = bb_ops[bb_idx].astype(np.uint8)
            
            for bt_idx in range(bt_ops.shape[0]):
                bt_logical = bt_ops[bt_idx].astype(np.uint8)
                
                # Check equivalence with this embedding
                is_equiv, reason, rank_diff, used_embedding = check_logical_equivalence_modulo_stabilizers(
                    bb_logical, bt_logical, bb_hz, bt_hz, embedding
                )
                
                if reason == "identical":
                    identical_pairs.append((bb_idx, bt_idx, embedding))
                    status = "≡ (identical)"
                elif is_equiv:
                    equivalent_pairs.append((bb_idx, bt_idx, embedding))
                    status = "≡ (equiv mod Hz)"
                else:
                    status = f"≢ (rank diff: +{rank_diff})"
                
                print(f"BB_{bb_idx} ↔ BT_{bt_idx}: {status}")
        
        results_by_embedding[embedding] = {
            'identical': identical_pairs,
            'equivalent': equivalent_pairs,
            'name': embedding_names[embed_idx]
        }
    
    # Print comprehensive summary
    print(f"\n" + "="*80)
    print("COMPREHENSIVE SUMMARY: ALL EMBEDDING POSITIONS")
    print("="*80)
    
    total_identical = 0
    total_equivalent = 0
    
    for embedding, results in results_by_embedding.items():
        print(f"\n🎯 EMBEDDING: BB → BT {results['name']} {embedding}")
        print("-" * 50)
        
        identical_pairs = results['identical']
        equivalent_pairs = results['equivalent']
        
        print(f"  Identical operators: {len(identical_pairs)}")
        for bb_idx, bt_idx, emb in identical_pairs:
            bb_weight = int(np.sum(bb_ops[bb_idx] != 0))
            bt_weight = int(np.sum(bt_ops[bt_idx] != 0))
            print(f"    BB_{bb_idx} ≡ BT_{bt_idx} (weights: {bb_weight}, {bt_weight})")
        
        print(f"  Equivalent modulo Hz: {len(equivalent_pairs)}")
        for bb_idx, bt_idx, emb in equivalent_pairs:
            bb_weight = int(np.sum(bb_ops[bb_idx] != 0))
            bt_weight = int(np.sum(bt_ops[bt_idx] != 0))
            print(f"    BB_{bb_idx} ≡ BT_{bt_idx} mod Hz (weights: {bb_weight}, {bt_weight})")
        
        embedding_total = len(identical_pairs) + len(equivalent_pairs)
        total_pairs = bb_ops.shape[0] * bt_ops.shape[0]
        print(f"  Subtotal: {embedding_total}/{total_pairs} ({100*embedding_total/total_pairs:.1f}%)")
        
        total_identical += len(identical_pairs)
        total_equivalent += len(equivalent_pairs)
    
    # Overall summary
    print(f"\n" + "="*60)
    print("OVERALL SUMMARY (ALL EMBEDDINGS)")
    print("="*60)
    total_all_embeddings = total_identical + total_equivalent
    total_possible = bb_ops.shape[0] * bt_ops.shape[0] * len(all_embeddings)
    
    print(f"Total equivalences found: {total_all_embeddings}/{total_possible} ({100*total_all_embeddings/total_possible:.1f}%)")
    print(f"  - Identical operators: {total_identical}")
    print(f"  - Equivalent modulo Hz: {total_equivalent}")
    
    # Check for new equivalences in blocks 1&3 or 2&3
    blocks_12 = set((bb, bt) for bb, bt, emb in results_by_embedding[(0,1)]['identical'] + results_by_embedding[(0,1)]['equivalent'])
    blocks_13 = set((bb, bt) for bb, bt, emb in results_by_embedding[(0,2)]['identical'] + results_by_embedding[(0,2)]['equivalent'])
    blocks_23 = set((bb, bt) for bb, bt, emb in results_by_embedding[(1,2)]['identical'] + results_by_embedding[(1,2)]['equivalent'])
    
    new_in_13 = blocks_13 - blocks_12
    new_in_23 = blocks_23 - blocks_12
    
    if new_in_13 or new_in_23:
        print(f"\n🆕 NEW EQUIVALENCES discovered in alternative embeddings:")
        if new_in_13:
            print(f"  Blocks 1&3 found {len(new_in_13)} new pairs: {list(new_in_13)}")
        if new_in_23:
            print(f"  Blocks 2&3 found {len(new_in_23)} new pairs: {list(new_in_23)}")
    else:
        print(f"\n❌ No new equivalences found in alternative block positions")
        print("   All equivalences occur only in the standard blocks 1&2 embedding")
    
    return results_by_embedding

def print_perfect_matches_detailed():
    """Print detailed comparison of perfect matches with grid coordinates."""
    print("\n" + "="*80)
    print("PERFECT MATCHES: DETAILED GRID COORDINATE COMPARISON")
    print("="*80)
    
    # Load and process operators
    bb_ops, bb_block_size, bb_num_blocks = load_logical_z_operators('BB')
    bt_ops, bt_block_size, bt_num_blocks = load_logical_z_operators('BT')
    bb_blocks = split_into_blocks(bb_ops, bb_block_size, bb_num_blocks)
    bt_blocks = split_into_blocks(bt_ops, bt_block_size, bt_num_blocks)
    
    # Perfect matches: BB_8↔BT_2, BB_9↔BT_3
    perfect_matches = [(8, 2), (9, 3)]
    
    for bb_idx, bt_idx in perfect_matches:
        print(f"\n🎯 PERFECT MATCH: BB_{bb_idx} ↔ BT_{bt_idx}")
        print("=" * 50)
        
        bb_operator = bb_blocks[bb_idx]
        bt_operator = bt_blocks[bt_idx]
        
        # Compare each block
        for block_idx in range(2):  # BB has 2 blocks
            bb_block = np.asarray(bb_operator[block_idx]).flatten()
            bt_block = np.asarray(bt_operator[block_idx]).flatten()
            
            bb_positions = np.where(bb_block != 0)[0]
            bt_positions = np.where(bt_block != 0)[0]
            
            bb_coords = [(int(position_to_grid_coord(pos)[0]), int(position_to_grid_coord(pos)[1])) for pos in bb_positions]
            bt_coords = [(int(position_to_grid_coord(pos)[0]), int(position_to_grid_coord(pos)[1])) for pos in bt_positions]
            
            print(f"\nBlock {block_idx}:")
            print(f"  BB_{bb_idx} block {block_idx}: {bb_coords}")
            print(f"  BT_{bt_idx} block {block_idx}: {bt_coords}")
            
            if bb_coords == bt_coords:
                print(f"  ✓ Perfect match! ({len(bb_coords)} positions)")
            else:
                print(f"  ✗ Mismatch")
        
        # Show BT's third block (should be all zeros for perfect matches)
        bt_block2 = np.asarray(bt_operator[2]).flatten()
        bt_positions2 = np.where(bt_block2 != 0)[0]
        bt_coords2 = [(int(position_to_grid_coord(pos)[0]), int(position_to_grid_coord(pos)[1])) for pos in bt_positions2]
        print(f"\nBT Block 2 (should be empty): {bt_coords2}")

def main():
    try:
        # Load operators
        print("Loading BB and BT logical Z operators...")
        bb_ops, bb_block_size, bb_num_blocks = load_logical_z_operators('BB')
        bt_ops, bt_block_size, bt_num_blocks = load_logical_z_operators('BT')
        
        # Split into blocks
        print("Splitting operators into blocks...")
        bb_blocks = split_into_blocks(bb_ops, bb_block_size, bb_num_blocks)
        bt_blocks = split_into_blocks(bt_ops, bt_block_size, bt_num_blocks)
        
        # Analyze weight patterns
        analyze_weight_patterns(bb_blocks, bt_blocks)
        
        # Find correspondences
        print("\nAnalyzing block correspondences...")
        correspondences = analyze_block_correspondence(bb_blocks, bt_blocks)
        
        # Print detailed results
        print_detailed_correspondences(correspondences)
        
        # Summary statistics
        total_pairs = len(bb_blocks) * len(bt_blocks)
        matched_pairs = len(correspondences)
        
        print(f"\n=== SUMMARY ===")
        print(f"Total possible BB-BT pairs: {total_pairs}")
        print(f"Pairs with exact block matches: {matched_pairs}")
        print(f"Match percentage: {matched_pairs/total_pairs*100:.1f}%")
        
        # Key correspondences summary
        print(f"\n=== KEY CORRESPONDENCES ===")
        print("Based on the analysis, the main structural relationships are:")
        print("• BB_0-5 (light): Only block 1 active → BT_0,1: Only block 1 active")  
        print("• BB_8: (11,5) → BT_2: (11,5,0) - Perfect 2-block match")
        print("• BB_9: (13,5) → BT_3: (13,5,0) - Perfect 2-block match") 
        print("• BB_10,11 (heavy): Different structure from BT_4,5")
        
        # Print complete grid coordinate listings
        print_logical_operators_grid_coordinates()
        
        # Print detailed perfect match analysis
        print_perfect_matches_detailed()
        
        # Verify logical equivalence modulo stabilizers using Gaussian elimination
        verify_logical_equivalence_all_pairs()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the logical Z operator files are generated first.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()