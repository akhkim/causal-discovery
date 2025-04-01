import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from dowhy import CausalModel
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import mv_fisherz
from causallearn.graph.Endpoint import Endpoint
import warnings
import os
warnings.filterwarnings('ignore')

def perform_causal_analysis(data_file, output_dir="output"):
    """
    Perform causal discovery using causallearn's FCI algorithm on all variables
    and estimate causal effects between all pairs of variables.
    
    Parameters:
    -----------
    data_file : str
        Path to the merged CSV file
    output_dir : str
        Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(data_file)
    
    # Check for missing values
    missing_values = df.isna().sum()
    print(f"\nMissing values in each column:")
    print(missing_values[missing_values > 0])
    
    # Prepare data for causal discovery
    print("\nPreparing data for causal discovery...")
    
    # Keep only numeric columns, excluding identifier columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude common identifier columns that shouldn't be part of causal analysis
    excluded_cols = []  # Add columns to exclude here
    analysis_vars = [col for col in numeric_cols if not any(col.lower() == ex.lower() for ex in excluded_cols)]
    
    print(f"\nIncluding {len(analysis_vars)} variables in causal analysis:")
    for i, var in enumerate(analysis_vars):
        print(f"{i+1}. {var}")
    
    # Create a dataset with only the selected variables
    data_for_analysis = df[analysis_vars].copy()
    
    # Check if there are too few complete cases - this can cause the math domain error
    n_complete = data_for_analysis.dropna().shape[0]
    min_required = len(analysis_vars) + 3  # Minimum sample size needed for Fisher's Z test
    
    if n_complete < min_required:
        print(f"\nWARNING: Only {n_complete} complete cases available, but at least {min_required} needed.")
        print("Filling missing values with mean to avoid numerical errors.")
        data_for_analysis = data_for_analysis.fillna(data_for_analysis.mean())
    
    # Handle specific columns with too many missing values
    missing_pct = data_for_analysis.isna().mean()
    high_missing_cols = missing_pct[missing_pct > 0.3].index.tolist()
    
    if high_missing_cols:
        print(f"\nWARNING: The following columns have >30% missing values and may cause problems:")
        for col in high_missing_cols:
            print(f"  - {col}: {missing_pct[col]*100:.1f}% missing")
        
        # Fill these columns with mean to avoid numerical errors
        data_for_analysis[high_missing_cols] = data_for_analysis[high_missing_cols].fillna(
            data_for_analysis[high_missing_cols].mean())
        print("These columns have been imputed with mean values to enable analysis.")
    
    # Convert to numpy array
    print("\nRunning FCI algorithm from causallearn...")
    data_array = data_for_analysis.to_numpy()
    
    try:
        # Run FCI algorithm with error handling
        G, edges = fci(data_array, mv_fisherz, 0.05, verbose=True)
    except ValueError as e:
        if "math domain error" in str(e):
            print("\nERROR: Math domain error encountered during FCI execution.")
            print("This typically happens when there are too few samples relative to variables.")
            print("Imputing remaining missing values and trying again with a smaller dataset...")
            
            # Impute all missing values
            data_for_analysis = data_for_analysis.fillna(data_for_analysis.mean())
            
            # Select variables with lower missing rates if dataset is large
            if len(analysis_vars) > 10:
                # Sort columns by missing rate
                cols_by_missing = missing_pct.sort_values().index.tolist()
                # Take the 10 columns with least missing values
                reduced_cols = cols_by_missing[:10]
                print(f"\nReducing analysis to 10 variables with least missing data:")
                for col in reduced_cols:
                    print(f"  - {col}")
                data_for_analysis = data_for_analysis[reduced_cols].copy()
                analysis_vars = reduced_cols
            
            # Try FCI again with the cleaned data
            data_array = data_for_analysis.to_numpy()
            G, edges = fci(data_array, mv_fisherz, 0.1, verbose=True)  # Using higher alpha for more permissive tests
        else:
            raise e
    
    # Update the node map to use "X#" format that causallearn uses internally
    node_map = {f"X{i}": name for i, name in enumerate(analysis_vars)}
    
    # Helper function to convert Endpoint object to string symbol
    def endpoint_to_symbol(endpoint):
        if endpoint == Endpoint.TAIL:
            return "-"
        elif endpoint == Endpoint.ARROW:
            return ">"
        elif endpoint == Endpoint.CIRCLE:
            return "o"
        else:
            return "?"
    
    # Save edges to file with variable names
    edge_file = os.path.join(output_dir, "fci_edges.txt")
    with open(edge_file, 'w') as f:
        f.write("Causal edges discovered by FCI:\n")
        # Get edges from the graph object
        edge_list = G.get_graph_edges()
        for edge in edge_list:
            try:
                from_node = node_map[edge.get_node1().get_name()]
                to_node = node_map[edge.get_node2().get_name()]
                
                # Convert endpoints to string symbols
                endpoint1_symbol = endpoint_to_symbol(edge.get_endpoint1())
                endpoint2_symbol = endpoint_to_symbol(edge.get_endpoint2())
                edge_type = f"{endpoint1_symbol}{endpoint2_symbol}"
                
                edge_str = f"{from_node} {edge_type} {to_node}"
                f.write(edge_str + "\n")
                print(edge_str)
            except KeyError as e:
                print(f"Warning: Node name not found in mapping: {e}")
    
    print(f"\nSaved edge list to {edge_file}")
    
    # Convert to NetworkX graph for visualization
    nx_graph = nx.DiGraph()
    
    # Add nodes
    for name in analysis_vars:
        nx_graph.add_node(name)
    
    # Add edges with appropriate direction
    edge_list = G.get_graph_edges()
    for edge in edge_list:
        try:
            from_node = node_map[edge.get_node1().get_name()]
            to_node = node_map[edge.get_node2().get_name()]
            
            # Check endpoints
            endpoint1 = edge.get_endpoint1()
            endpoint2 = edge.get_endpoint2()
            
            # Handle edge directions based on endpoints
            if endpoint2 == Endpoint.ARROW:
                if endpoint1 == Endpoint.TAIL or endpoint1 == Endpoint.CIRCLE:
                    nx_graph.add_edge(from_node, to_node)
            elif endpoint1 == Endpoint.ARROW:
                if endpoint2 == Endpoint.TAIL or endpoint2 == Endpoint.CIRCLE:
                    nx_graph.add_edge(to_node, from_node)
            elif endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.TAIL:
                # Undirected edge
                nx_graph.add_edge(from_node, to_node)
                nx_graph.add_edge(to_node, from_node)
            elif endpoint1 == Endpoint.CIRCLE and endpoint2 == Endpoint.CIRCLE:
                # Undetermined edge
                nx_graph.add_edge(from_node, to_node)
                nx_graph.add_edge(to_node, from_node)
        except KeyError as e:
            print(f"Warning: Skipping edge due to unknown node: {e}")
    
    # Plot the graph
    plt.figure(figsize=(20, 16))
    
    pos = nx.kamada_kawai_layout(nx_graph)
    
    # Draw the graph
    nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', 
            node_size=2500, arrowsize=15, font_size=8, font_weight='bold',
            connectionstyle='arc3, rad=0.1', arrows=True)
    
    # Save the figure with high resolution
    graph_file = os.path.join(output_dir, "fci_causal_graph.png")
    plt.title("Causal Graph from FCI Algorithm", fontsize=16)
    plt.tight_layout()
    plt.savefig(graph_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved causal graph to {graph_file}")
    
    svg_file = os.path.join(output_dir, "fci_causal_graph.svg")
    plt.savefig(svg_file, format='svg', bbox_inches='tight')
    print(f"Saved vector graphics version to {svg_file}")
    
    print("\nAnalyzing key causal relationships...")
    
    # Function to estimate causal effect
    def estimate_effect(treatment, outcome, confounders):
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
            
            return {
                'treatment': treatment,
                'outcome': outcome, 
                'effect': estimate.value,
                'stderr': estimate.stderr if hasattr(estimate, 'stderr') else None,
                'p_value': estimate.p_value if hasattr(estimate, 'p_value') else None
            }
        except Exception as e:
            print(f"Error estimating effect of {treatment} on {outcome}: {e}")
            return {
                'treatment': treatment,
                'outcome': outcome,
                'effect': None,
                'stderr': None,
                'p_value': None,
                'error': str(e)
            }
    
    # Identify all directed edges
    directed_edges = []
    edge_list = G.get_graph_edges()
    for edge in edge_list:
        try:
            from_node = node_map[edge.get_node1().get_name()]
            to_node = node_map[edge.get_node2().get_name()]
            endpoint1 = edge.get_endpoint1()
            endpoint2 = edge.get_endpoint2()
            
            # Check for directed edges
            if endpoint2 == Endpoint.ARROW and endpoint1 in [Endpoint.TAIL, Endpoint.CIRCLE]:
                directed_edges.append((from_node, to_node))
            elif endpoint1 == Endpoint.ARROW and endpoint2 in [Endpoint.TAIL, Endpoint.CIRCLE]:
                directed_edges.append((to_node, from_node))
        except KeyError:
            continue
    
    print(f"\nFound {len(directed_edges)} directed causal relationships.")
    
    # Estimate causal effects for directed edges
    results = []
    
    for edge in directed_edges:
        treatment, outcome = edge
        
        # Find confounders
        confounders = []
        for node in nx_graph.nodes():
            if node != treatment and node != outcome:
                if (nx_graph.has_edge(node, treatment) or nx_graph.has_edge(treatment, node)) and \
                   (nx_graph.has_edge(node, outcome) or nx_graph.has_edge(outcome, node)):
                    confounders.append(node)
        
        print(f"\nEstimating effect of {treatment} on {outcome}...")
        print(f"Controlling for confounders: {confounders}")
        
        result = estimate_effect(treatment, outcome, confounders)
        results.append(result)
        
        effect_str = f"{result['effect']:.4f}" if result['effect'] is not None else "Error"
        print(f"Causal effect: {effect_str}")
    
    # Save estimation results to CSV
    if results:
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
                print(f"{i+1}. {r['treatment']} → {r['outcome']}: {r['effect']:.4f}")
    else:
        print("\nNo valid causal effects were found.")
    
    print("\nCausal analysis complete.")

if __name__ == "__main__":
    data_file = "cleaned_dataset.csv"
    perform_causal_analysis(data_file)