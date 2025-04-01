import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from dowhy import CausalModel
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import gsq
from causallearn.graph.Endpoint import Endpoint
import warnings
import os
import gc
from itertools import combinations
from collections import Counter
import json
import pickle

warnings.filterwarnings('ignore')

def detect_perfect_relationships_and_constants(data_array, var_names, output_dir, chunk_idx=None):
    """
    Detect constant variables and perfect relationships between variables.
    Save the results to separate files for reference.
    
    Parameters:
    -----------
    data_array : numpy.ndarray
        The data array to analyze
    var_names : list
        List of variable names corresponding to columns in data_array
    output_dir : str
        Directory to save output files
    chunk_idx : int, optional
        If provided, identifies which chunk this analysis belongs to
    
    Returns:
    --------
    dict
        Information about detected constant variables and perfect relationships
    """
    # Create output directory for special relationships
    special_dir = os.path.join(output_dir, "special_relationships")
    os.makedirs(special_dir, exist_ok=True)
    
    # Determine file suffix based on chunk_idx
    suffix = f"_chunk_{chunk_idx}" if chunk_idx is not None else ""
    
    results = {
        "constant_variables": [],
        "perfect_relationships": []
    }
    
    # Detect constant variables
    variances = np.var(data_array, axis=0)
    constant_vars = np.isclose(variances, 0)
    
    if np.any(constant_vars):
        constant_indices = np.where(constant_vars)[0]
        for idx in constant_indices:
            if idx < len(var_names):
                var_name = var_names[idx]
                var_value = data_array[0, idx]  # Value of the constant
                results["constant_variables"].append({
                    "variable": var_name,
                    "value": float(var_value)
                })
    
    # Detect perfect relationships
    # Compute correlation matrix
    try:
        corr_matrix = np.corrcoef(data_array, rowvar=False)
        np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlations
        
        # Find perfect correlations (close to 1 or -1)
        perfect_pos = np.isclose(corr_matrix, 1.0, atol=1e-5)
        perfect_neg = np.isclose(corr_matrix, -1.0, atol=1e-5)
        perfect_corr = np.logical_or(perfect_pos, perfect_neg)
        
        # Log perfect relationships
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):  # Only check upper triangle
                if perfect_corr[i, j]:
                    if i < len(var_names) and j < len(var_names):
                        sign = "+" if perfect_pos[i, j] else "-"
                        corr_value = corr_matrix[i, j]
                        results["perfect_relationships"].append({
                            "var1": var_names[i],
                            "var2": var_names[j],
                            "correlation": float(corr_value),
                            "relationship": f"{var_names[i]} {'=' if sign == '+' else '= -'} {var_names[j]}"
                        })
    except Exception as e:
        print(f"Warning: Could not compute correlations - {e}")
    
    # Save results to files
    constants_file = os.path.join(special_dir, f"constant_variables{suffix}.json")
    with open(constants_file, 'w') as f:
        json.dump(results["constant_variables"], f, indent=2)
    
    perfect_file = os.path.join(special_dir, f"perfect_relationships{suffix}.json")
    with open(perfect_file, 'w') as f:
        json.dump(results["perfect_relationships"], f, indent=2)
    
    # Also create a human-readable text file
    summary_file = os.path.join(special_dir, f"special_relationships_summary{suffix}.txt")
    with open(summary_file, 'w') as f:
        f.write("SPECIAL RELATIONSHIPS DETECTED\n")
        f.write("=============================\n\n")
        
        f.write("CONSTANT VARIABLES:\n")
        if results["constant_variables"]:
            for item in results["constant_variables"]:
                f.write(f"  {item['variable']} = {item['value']}\n")
        else:
            f.write("  None detected\n")
        
        f.write("\nPERFECT RELATIONSHIPS:\n")
        if results["perfect_relationships"]:
            for item in results["perfect_relationships"]:
                f.write(f"  {item['relationship']} (r = {item['correlation']})\n")
        else:
            f.write("  None detected\n")
    
    print(f"Detected {len(results['constant_variables'])} constant variables")
    print(f"Detected {len(results['perfect_relationships'])} perfect relationships")
    print(f"Saved special relationships to {special_dir}")
    
    return results

def perform_memory_efficient_causal_analysis(data_file, chunk_size=1000, overlap=200, output_dir="output", p_value_threshold=0.05):
    """
    Perform FCI causal discovery in chunks of rows, merge the results,
    then run unified causal effect estimation using DoWhy - optimized for memory usage.
    
    Parameters:
    -----------
    data_file : str
        Path to the CSV file
    chunk_size : int
        Number of rows to analyze in each chunk
    overlap : int
        Number of overlapping rows between chunks
    output_dir : str
        Directory to save output files
    p_value_threshold : float
        P-value threshold for the FCI algorithm
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "chunk_results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "special_relationships"), exist_ok=True)
    
    print("Preparing for chunk-based analysis...")
    
    # Get column information and row count without loading the entire dataset
    print("Reading column information...")
    df_sample = pd.read_csv(data_file, nrows=5)
    
    # Identify numeric columns
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    excluded_cols = ['geo']  # Add columns to exclude here
    analysis_vars = [col for col in numeric_cols if not any(col.lower() == ex.lower() for ex in excluded_cols)]
    
    print(f"Total variables for analysis: {len(analysis_vars)}")
    
    # Get total row count using chunked reading
    print("Counting total rows...")
    total_rows = 0
    for chunk in pd.read_csv(data_file, chunksize=10000, usecols=['geo']):
        total_rows += len(chunk)
    
    print(f"Total rows in dataset: {total_rows}")
    
    # Create chunks of rows with overlap
    chunks = []
    i = 0
    
    while i < total_rows:
        end_idx = min(i + chunk_size, total_rows)
        chunks.append((i, end_idx))
        i += (chunk_size - overlap)
    
    print(f"Created {len(chunks)} chunks of rows (size={chunk_size}, overlap={overlap})")
    
    # Initialize dictionaries to aggregate special relationships across chunks
    all_constant_vars = {}
    all_perfect_relationships = set()
    
    # Process each chunk with minimal memory footprint
    for chunk_idx, (start_idx, end_idx) in enumerate(chunks):
        print(f"\n--- Processing Chunk {chunk_idx+1}/{len(chunks)} ---")
        print(f"Rows in this chunk: {start_idx} to {end_idx-1}")
        
        # Create a unique file identifier for this chunk
        chunk_file = os.path.join(output_dir, "chunk_results", f"chunk_{chunk_idx+1}.pkl")
        
        # Skip if already processed
        if os.path.exists(chunk_file):
            print(f"Chunk {chunk_idx+1} already processed. Skipping...")
            continue
        
        # Read only the specific chunk of rows and only the columns we need
        print(f"Reading chunk {chunk_idx+1}...")
        
        # Calculate skip rows: skip all rows except header + chunk rows
        skip_rows = list(range(1, start_idx + 1))  # +1 because we want to keep header row
        if end_idx < total_rows:
            skip_rows.extend(range(end_idx + 1, total_rows + 1))
        
        # Read only the required subset of the data
        chunk_data = pd.read_csv(data_file, 
                                skiprows=skip_rows if skip_rows else None,
                                nrows=(end_idx - start_idx) if not skip_rows else None,
                                usecols=analysis_vars)
        
        print(f"Loaded chunk with shape: {chunk_data.shape}")
        
        # Convert to numpy array
        data_array = chunk_data.to_numpy()
        
        # Detect and save special relationships in this chunk
        print("Detecting constant variables and perfect relationships...")
        special_relationships = detect_perfect_relationships_and_constants(
            data_array, 
            analysis_vars, 
            output_dir,
            chunk_idx+1
        )
        
        # Aggregate special relationships
        for const_var in special_relationships["constant_variables"]:
            var_name = const_var["variable"]
            if var_name in all_constant_vars:
                all_constant_vars[var_name]["count"] += 1
            else:
                all_constant_vars[var_name] = {
                    "value": const_var["value"],
                    "count": 1
                }
        
        for rel in special_relationships["perfect_relationships"]:
            rel_key = f"{rel['var1']}|{rel['var2']}|{rel['correlation']}"
            all_perfect_relationships.add(rel_key)
        
        # Run FCI algorithm on this chunk
        try:
            print(f"Running FCI algorithm on this chunk with p-value threshold {p_value_threshold}...")
            G, edges = fci(data_array, gsq, p_value_threshold, verbose=False)
            
            # Update node map for this chunk
            node_map = {f"X{i}": name for i, name in enumerate(analysis_vars)}
            
            # Process edges from this chunk
            edge_list = G.get_graph_edges()
            
            # Store only the necessary edge information
            chunk_edges = []
            
            for edge in edge_list:
                try:
                    from_node = node_map[edge.get_node1().get_name()]
                    to_node = node_map[edge.get_node2().get_name()]
                    
                    # Store raw FCI edge type
                    endpoint1 = str(edge.get_endpoint1())  # Convert Endpoint to string for serialization
                    endpoint2 = str(edge.get_endpoint2())
                    
                    chunk_edges.append({
                        'from_node': from_node,
                        'to_node': to_node,
                        'endpoint1': endpoint1,
                        'endpoint2': endpoint2
                    })
                    
                except KeyError as e:
                    print(f"Warning: Node name not found in mapping: {e}")
            
            print(f"Found {len(chunk_edges)} edges in this chunk")
            
            # Save this chunk's results to disk
            with open(chunk_file, 'wb') as f:
                pickle.dump({
                    'edges': chunk_edges,
                    'vars': analysis_vars
                }, f)
            
            print(f"Saved chunk {chunk_idx+1} results to {chunk_file}")
            
        except Exception as e:
            print(f"Error processing chunk {chunk_idx+1}: {e}")
        
        # Clear all variables to free memory
        del chunk_data, data_array, G, edges
        gc.collect()
    
    # Save aggregated special relationships
    special_dir = os.path.join(output_dir, "special_relationships")
    
    # Save aggregated constant variables
    constants_file = os.path.join(special_dir, "aggregated_constant_variables.json")
    with open(constants_file, 'w') as f:
        json.dump(all_constant_vars, f, indent=2)
    
    # Save aggregated perfect relationships
    perfect_rels = []
    for rel_key in all_perfect_relationships:
        var1, var2, corr = rel_key.split('|')
        perfect_rels.append({
            "var1": var1,
            "var2": var2,
            "correlation": float(corr),
            "relationship": f"{var1} {'=' if float(corr) > 0 else '= -'} {var2}"
        })
    
    perfect_file = os.path.join(special_dir, "aggregated_perfect_relationships.json")
    with open(perfect_file, 'w') as f:
        json.dump(perfect_rels, f, indent=2)
    
    # Create a human-readable aggregated summary
    summary_file = os.path.join(special_dir, "aggregated_special_relationships_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("AGGREGATED SPECIAL RELATIONSHIPS ACROSS ALL CHUNKS\n")
        f.write("===============================================\n\n")
        
        f.write("CONSTANT VARIABLES:\n")
        if all_constant_vars:
            for var_name, info in sorted(all_constant_vars.items(), key=lambda x: x[1]["count"], reverse=True):
                f.write(f"  {var_name} = {info['value']} (found in {info['count']} chunks)\n")
        else:
            f.write("  None detected\n")
        
        f.write("\nPERFECT RELATIONSHIPS:\n")
        if perfect_rels:
            for rel in sorted(perfect_rels, key=lambda x: x["var1"]):
                f.write(f"  {rel['relationship']} (r = {rel['correlation']})\n")
        else:
            f.write("  None detected\n")
    
    print("\nAll chunks processed. Merging results...")
    
    # Initialize counters for edge voting
    edge_votes = Counter()
    fci_edges = {}  # Format: (node1, node2) -> (endpoint1, endpoint2, vote_count)
    
    # Function to convert endpoint string back to comparable value
    def endpoint_value(endpoint_str):
        if "TAIL" in endpoint_str:
            return "TAIL"
        elif "ARROW" in endpoint_str:
            return "ARROW"
        elif "CIRCLE" in endpoint_str:
            return "CIRCLE"
        else:
            return endpoint_str
    
    # Incrementally load and merge all chunk results
    for chunk_idx in range(len(chunks)):
        chunk_file = os.path.join(output_dir, "chunk_results", f"chunk_{chunk_idx+1}.pkl")
        
        if not os.path.exists(chunk_file):
            print(f"Warning: Results for chunk {chunk_idx+1} not found!")
            continue
        
        try:
            # Load this chunk's results
            with open(chunk_file, 'rb') as f:
                chunk_result = pickle.load(f)
            
            # Process edges from this chunk
            for edge in chunk_result['edges']:
                from_node = edge['from_node']
                to_node = edge['to_node']
                endpoint1 = endpoint_value(edge['endpoint1'])
                endpoint2 = endpoint_value(edge['endpoint2'])
                
                # Count votes for directed edges
                if endpoint2 == "ARROW" and endpoint1 == "TAIL":
                    edge_votes[(from_node, to_node)] += 1
                elif endpoint1 == "ARROW" and endpoint2 == "TAIL":
                    edge_votes[(to_node, from_node)] += 1
                
                # Store edge type information
                edge_pair = tuple(sorted([from_node, to_node]))
                
                if edge_pair in fci_edges:
                    # Update the vote count for this edge type
                    prev_from, prev_to, prev_endpoint1, prev_endpoint2, prev_count = fci_edges[edge_pair]
                    if prev_endpoint1 == endpoint1 and prev_endpoint2 == endpoint2:
                        fci_edges[edge_pair] = (prev_from, prev_to, prev_endpoint1, prev_endpoint2, prev_count + 1)
                    else:
                        # Add as new edge type
                        edge_key = (edge_pair[0], edge_pair[1], endpoint1, endpoint2)
                        if edge_key not in fci_edges:
                            fci_edges[edge_key] = (from_node, to_node, endpoint1, endpoint2, 1)
                else:
                    fci_edges[edge_pair] = (from_node, to_node, endpoint1, endpoint2, 1)
            
            print(f"Merged {len(chunk_result['edges'])} edges from chunk {chunk_idx+1}")
            
            # Keep variable list from first chunk
            if chunk_idx == 0:
                analysis_vars = chunk_result['vars']
            
        except Exception as e:
            print(f"Error merging results from chunk {chunk_idx+1}: {e}")
        
        # Free memory
        gc.collect()
    
    # Save merged edge votes to disk
    edge_votes_file = os.path.join(output_dir, "edge_votes.json")
    with open(edge_votes_file, 'w') as f:
        # Convert tuples to lists for JSON serialization
        json_votes = {f"{from_node}|{to_node}": votes for (from_node, to_node), votes in edge_votes.items()}
        json.dump(json_votes, f, indent=2)
    
    # Create a combined graph based on voting
    combined_graph = nx.DiGraph()
    for var in analysis_vars:
        combined_graph.add_node(var)
    
    print("\nAdding all unique edges found across chunks...")

    for edge_str, votes in json_votes.items():
        from_node, to_node = edge_str.split('|')
        combined_graph.add_edge(from_node, to_node, weight=votes)
        print(f"Added edge: {from_node} -> {to_node} (votes: {votes})")
    
    # Save the graph structure
    nx.write_gexf(combined_graph, os.path.join(output_dir, "causal_graph.gexf"))
    
    # Helper function to convert Endpoint object to string symbol
    def endpoint_to_symbol(endpoint):
        if endpoint == "TAIL":
            return "-"
        elif endpoint == "ARROW":
            return ">"
        elif endpoint == "CIRCLE":
            return "o"
        else:
            return "?"
    
    # Save the FCI edge types
    edge_file = os.path.join(output_dir, "fci_edges.txt")
    with open(edge_file, 'w') as f:
        f.write("Causal edges discovered by FCI across all chunks:\n")
        for edge_data in fci_edges.values():
            if len(edge_data) == 5:  # Make sure we have the right format
                from_node, to_node, endpoint1, endpoint2, votes = edge_data
                endpoint1_symbol = endpoint_to_symbol(endpoint1)
                endpoint2_symbol = endpoint_to_symbol(endpoint2)
                edge_type = f"{endpoint1_symbol}{endpoint2_symbol}"
                
                edge_str = f"{from_node} {edge_type} {to_node} (votes: {votes})"
                f.write(edge_str + "\n")
    
    print(f"\nSaved edge list to {edge_file}")
    
    # Plot the combined graph
    plt.figure(figsize=(20, 16))
    
    # Use force layout for better placement
    pos = nx.kamada_kawai_layout(combined_graph)
    
    # Get edge weights for line thickness
    edge_weights = [combined_graph[u][v]['weight'] for u, v in combined_graph.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    normalized_weights = [1 + 4 * (w / max_weight) for w in edge_weights]
    
    # Draw the graph
    nx.draw(combined_graph, pos, with_labels=True, node_color='lightblue', 
            node_size=2500, arrowsize=15, font_size=8, font_weight='bold',
            connectionstyle='arc3, rad=0.1', arrows=True, 
            width=normalized_weights)  # Line width based on vote count
    
    # Save the figure
    graph_file = os.path.join(output_dir, "fci_causal_graph.png")
    plt.title("Combined Causal Graph from Row-Chunked FCI Algorithm", fontsize=16)
    plt.tight_layout()
    plt.savefig(graph_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved causal graph to {graph_file}")
    
    svg_file = os.path.join(output_dir, "fci_causal_graph.svg")
    plt.savefig(svg_file, format='svg', bbox_inches='tight')
    plt.close()
    
    # Identify all directed edges from the combined graph
    directed_edges = list(combined_graph.edges())
    print(f"\nFound {len(directed_edges)} directed causal relationships in the combined graph.")
    
    # Check for special relationships in identified edges
    print("\nAnalyzing causal relationships with respect to special relationships...")
    
    # Load aggregated special relationships 
    special_edges_file = os.path.join(output_dir, "special_relationships_in_causal_graph.txt")
    with open(special_edges_file, 'w') as f:
        f.write("SPECIAL RELATIONSHIPS IN CAUSAL GRAPH\n")
        f.write("===================================\n\n")
        
        # Check for constant variables in causal relationships
        f.write("CONSTANT VARIABLES IN CAUSAL RELATIONSHIPS:\n")
        const_in_graph = []
        for var_name in all_constant_vars:
            edges_with_const = []
            for u, v in directed_edges:
                if u == var_name or v == var_name:
                    edges_with_const.append((u, v))
            
            if edges_with_const:
                const_in_graph.append(var_name)
                f.write(f"  {var_name} = {all_constant_vars[var_name]['value']}:\n")
                for u, v in edges_with_const:
                    f.write(f"    {u} -> {v}\n")
        
        if not const_in_graph:
            f.write("  None found\n")
            
        # Check for perfect relationships in causal graph
        f.write("\nPERFECT RELATIONSHIPS IN CAUSAL GRAPH:\n")
        perfect_in_graph = []
        for rel in perfect_rels:
            var1, var2 = rel["var1"], rel["var2"]
            edges_with_perfect = []
            
            for u, v in directed_edges:
                if (u == var1 and v == var2) or (u == var2 and v == var1):
                    edges_with_perfect.append((u, v))
            
            if edges_with_perfect:
                perfect_in_graph.append((var1, var2))
                f.write(f"  {rel['relationship']} (r = {rel['correlation']}):\n")
                for u, v in edges_with_perfect:
                    f.write(f"    {u} -> {v}\n")
        
        if not perfect_in_graph:
            f.write("  None found\n")
    
    print(f"Saved analysis of special relationships in causal graph to {special_edges_file}")
    
    # Now load the full dataset for DoWhy causal effect estimation
    print("\nLoading full dataset for causal effect estimation...")
    
    # Determine which columns we need for the DoWhy analysis
    required_cols = set()
    for u, v in directed_edges:
        required_cols.add(u)
        required_cols.add(v)
    
    # Add potential confounders (any node with outgoing edges to treatments or outcomes)
    for node in combined_graph.nodes():
        for u, v in directed_edges:
            if combined_graph.has_edge(node, u) or combined_graph.has_edge(node, v):
                required_cols.add(node)
    
    required_cols = list(required_cols)
    print(f"Loading {len(required_cols)} columns for DoWhy analysis")
    
    # Load only the required columns
    df = pd.read_csv(data_file, usecols=required_cols)
    
    # Perform a unified causal effect estimation using DoWhy
    # Limit the number of relationships to analyze if there are too many
    max_relationships = min(50, len(directed_edges))
    print(f"\nEstimating causal effects for up to {max_relationships} relationships...")
    
    # Sort relationships by vote count (edge weight)
    edge_importance = []
    for u, v in directed_edges:
        importance = combined_graph[u][v]['weight']
        edge_importance.append((u, v, importance))
    
    # Select the relationships with the most votes
    selected_edges = [edge[:2] for edge in sorted(edge_importance, key=lambda x: x[2], reverse=True)[:max_relationships]]
    
    # Estimate causal effects
    results = []
    
    for treatment, outcome in selected_edges:
        print(f"\nEstimating effect of {treatment} on {outcome}...")
        
        # Check if this involves a constant variable or perfect relationship
        is_special = False
        special_note = ""
        
        # Check for constant variable
        if treatment in all_constant_vars:
            is_special = True
            special_note += f"WARNING: Treatment '{treatment}' is a constant variable. "
        
        if outcome in all_constant_vars:
            is_special = True
            special_note += f"WARNING: Outcome '{outcome}' is a constant variable. "
        
        # Check for perfect relationship
        for rel in perfect_rels:
            if (treatment == rel["var1"] and outcome == rel["var2"]) or (treatment == rel["var2"] and outcome == rel["var1"]):
                is_special = True
                special_note += f"WARNING: Treatment and outcome have a perfect correlation (r = {rel['correlation']}). "
        
        if is_special:
            print(special_note)
        
        # Find potential confounders using the graph structure
        confounders = []
        for node in combined_graph.nodes():
            if node != treatment and node != outcome:
                # Check if node is a potential confounder (affects both treatment and outcome)
                if combined_graph.has_edge(node, treatment) and combined_graph.has_edge(node, outcome):
                    confounders.append(node)
        
        # Limit number of confounders to manage memory usage
        if len(confounders) > 10:
            print(f"Limiting confounders from {len(confounders)} to 10 for memory efficiency")
            confounders = confounders[:10]
        
        print(f"Controlling for confounders: {confounders}")
        
        try:
            # Create a causal model
            model = CausalModel(
                data=df,
                treatment=treatment,
                outcome=outcome,
                common_causes=confounders
            )
            
            # Identify the causal effect
            identified_estimand = model.identify_effect()
            
            # Estimate the causal effect
            estimate = model.estimate_effect(identified_estimand,
                                           method_name="backdoor.linear_regression")
            
            effect = estimate.value
            p_value = estimate.p_value if hasattr(estimate, 'p_value') else None
            stderr = estimate.stderr if hasattr(estimate, 'stderr') else None
            
            results.append({
                'treatment': treatment,
                'outcome': outcome,
                'effect': effect,
                'p_value': p_value,
                'stderr': stderr,
                'confounders': confounders,
                'vote_count': combined_graph[treatment][outcome]['weight'],
                'special_relationship': special_note if is_special else ""
            })
            
            print(f"Causal effect: {effect:.4f}")
            
        except Exception as e:
            print(f"Error estimating effect of {treatment} on {outcome}: {e}")
            results.append({
                'treatment': treatment,
                'outcome': outcome,
                'effect': None,
                'p_value': None,
                'stderr': None,
                'error': str(e),
                'confounders': confounders,
                'vote_count': combined_graph[treatment][outcome]['weight'],
                'special_relationship': special_note if is_special else ""
            })
    
    # Save estimation results to CSV
    if results:
        # Convert confounders list to string for CSV storage
        for r in results:
            if 'confounders' in r:
                r['confounders'] = ', '.join(r['confounders'])
        
        results_df = pd.DataFrame(results)
        results_file = os.path.join(output_dir, "causal_effects.csv")
        results_df.to_csv(results_file, index=False)
        print(f"\nSaved causal effect estimates to {results_file}")
        
        # Print top effects
        valid_results = [r for r in results if r['effect'] is not None]
        if valid_results:
            sorted_results = sorted(valid_results, key=lambda x: abs(x['effect']), reverse=True)
            
            print("\nTop 5 strongest causal effects:")
            for i, r in enumerate(sorted_results[:5]):
                print(f"{i+1}. {r['treatment']} → {r['outcome']}: {r['effect']:.4f} (votes: {r['vote_count']})")
                if r['special_relationship']:
                    print(f"   Note: {r['special_relationship']}")
    else:
        print("\nNo valid causal effects were found.")
    
    # Clean up memory one last time
    del df
    gc.collect()
    
    print("\nMemory-efficient unified causal analysis complete.")

if __name__ == "__main__":
    data_file = "data/by_country/afg.csv"
    perform_memory_efficient_causal_analysis(data_file, chunk_size=50, overlap=10, p_value_threshold=0.5)
    