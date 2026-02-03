import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import powerlaw 
import igraph as ig 
import leidenalg as la
from collections import defaultdict

from Qommunity.samplers.regular.louvain_sampler import LouvainSampler
from Qommunity.iterative_searcher import IterativeSearcher

def rewrite_subjectID(subject_ID):
    subject_4digits = subject_ID.split("-")
    subject = "-".join(subject_4digits[:-1])
    digits = str(subject_4digits[-1][1:])
    return subject + "-" + digits

def extract_giant_component(
    matrix, 
    threshold
):
    """
    Extract giant component.
    """
    # Binarize the connectome
    binary_matrix = (matrix > threshold).astype(int)
    # Safety check --> No self-loops
    np.fill_diagonal(binary_matrix, 0)  
    # Create graph
    G = nx.from_numpy_array(binary_matrix, create_using=nx.Graph)
    # If empty graph return None for later Warning
    if G.number_of_edges() == 0:
        return None
    # Extract the giant component and return the networkx subgraph
    return G.subgraph(
        max(nx.connected_components(G), key=len)
    ).copy()

def fit_power_law_to_degrees(
    degrees_array,
    min_samples=50,
    min_unique=10,
    discrete=True,
    alpha_sig=0.05,
):
    """
    Fit a power-law to degree data (assumed to come from the giant component).
    Prints a warning if the power-law is not statistically distinguishable
    from a lognormal distribution.
    """
    degrees = np.asarray(degrees_array)
    degrees = degrees[degrees > 0]

    if len(degrees) < min_samples or len(np.unique(degrees)) < min_unique:
        return np.nan, np.nan, np.nan

    try:
        fit = powerlaw.Fit(degrees, discrete=discrete, verbose=False)

        # Compare power-law vs lognormal
        R, p = fit.distribution_compare("power_law", "lognormal")

        if p > alpha_sig:
            degree_distribution = "Indistinguishable"
        elif R > 0:
            degree_distribution = "Power-law"
        else:
            degree_distribution = "Lognormal"

        return fit.power_law.alpha, fit.power_law.xmin, R, degree_distribution

    except:
        return np.nan, np.nan, np.nan

def newman_modularity(G, runs=100, resolution=1):
    """ TODO """
    # Louvain algorithm 
    CL, Louvain = IterativeSearcher(LouvainSampler(G, use_weights=False, resolution=resolution)).run(
        num_runs=runs,
        save_results=False,
        saving_path=None,
        elapse_times=False,
        iterative_verbosity=0
    )
    MLouvain = Louvain.max()
    CLouvain = CL[Louvain.argmax()]

    # Leiden algorithm
    IG = ig.Graph.from_networkx(G)
    nx_nodes = dict(zip(IG.vs.indices,IG.vs["_nx_name"]))
    Leiden = np.zeros((runs,))
    CLe = np.empty(shape=(runs, ), dtype=object)
    for run in range(runs):
        CLe_ = list(la.find_partition(
            IG,
            partition_type=la.RBConfigurationVertexPartition,
            weights=None,
            resolution_parameter=1,
        ))
        CLe[run] = [[nx_nodes[n] for n in CLe_[i]] for i in range(len(CLe_))]
        Leiden[run] = nx.community.modularity(G, CLe[run], resolution=resolution)
    MLeiden = Leiden.max()
    CLeiden = CLe[Leiden.argmax()]

    # We return the maximum modularity
    if MLeiden>=MLouvain:
        M = MLeiden
        C = CLeiden
    else:
        M = MLouvain
        C = CLouvain

    return M, C


def participation_coefficient(G, communities):
    """
    G: NetworkX graph
    partition: dict {node: community} or list aligned with G.nodes()
    Returns: dict {node: participation coefficient}
    """

    partition = {}
    for comm_id, nodes in enumerate(communities):
        for node in nodes:
            partition[node] = comm_id

    if isinstance(partition, list):
        partition = dict(zip(G.nodes(), partition))
    
    # Build community-wise neighbor counts
    pc = {}
    for node in G.nodes():
        k_i = G.degree(node, weight=None)
        if k_i == 0:
            pc[node] = 0.0
            continue
        
        comms = defaultdict(float)
        for neighbor in G.neighbors(node):
            comms[partition[neighbor]] += G[node][neighbor].get("weight", 1.0)
        
        sum_sq = sum((comms[c]/k_i)**2 for c in comms)
        pc[node] = 1.0 - sum_sq
    
    return pc

def norm_rich_club_coef(G, rich_club, Q=100, m=100):
    """TODO"""
    # Normalize the Rich club coefficient
    phi_rand_list = {k: [] for k in rich_club.keys()}
    E = G.number_of_edges()
    nswap = m * E
    max_tries = 50 * nswap

    for kkk in range(Q):
        print(kkk)
        Grand = G.copy()
        nx.double_edge_swap(Grand, nswap=nswap, max_tries=max_tries)
        phi_r = nx.rich_club_coefficient(Grand, normalized=False)
        for k in rich_club.keys():
            phi_rand_list[k].append(
                phi_r.get(k, 0) # get k or 0 if k does not exist
            )                   # (the randomized version might not have that particular degree)

    phi_norm = {k: rich_club[k] / np.mean(phi_rand_list[k]) for k in rich_club.keys()}
    return phi_norm

if __name__ == '__main__':
    # Get the subject to process
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Main directory to store the results")
    parser.add_argument("subject", type=str, help="Full ID of the subject (e.g., UPENN-GBM-XXXXX_11)")
    parser.add_argument("connectome", type=str, help="Path of the connectome to process")
    parser.add_argument("output", type=str, help="Path of the output file with the graph metrics")
    parser.add_argument("demographics", type=str, help="Path to the CSV file with clinical data")
    parser.add_argument("--atlas", type=str, help="Path to the file with the labels")
    parser.add_argument("--min_streamlines", type=int, default=0, help="Minimum number of streamlines per voxel to consider")
    args = parser.parse_args()
    
    # Loading the metadata that is available in the demographics
    DIR = args.dir[:-1] if args.dir[-1]=="/" else args.dir # We delete the last "/" if present
    SUBJECT = rewrite_subjectID(args.subject)
    LESION_CONNECTOME = args.connectome
    LESION_CONNECTOME_METRICS = args.output
    ATLAS = args.atlas
    DEMOGRAPHICS = pd.read_csv(args.demographics)

    # We preselect only the current working subject
    ID_column = DEMOGRAPHICS.columns[0] 
    row = pd.DataFrame(DEMOGRAPHICS.loc[DEMOGRAPHICS[ID_column]==SUBJECT])
    if row.empty:
        raise Warning(f"______ No entry for {SUBJECT} was found in the clinical data: \n\t{args.demographics} ______")
    else:
        CONNECTOME = pd.read_csv(LESION_CONNECTOME, header=None).values.astype(np.float64)
        GIANT_COMPONENT = extract_giant_component(CONNECTOME, threshold=args.min_streamlines)
        DEGREES = dict(GIANT_COMPONENT.degree())

        row["size"] = len(GIANT_COMPONENT)
        row["density"] = nx.density(GIANT_COMPONENT)
        row["avg_degree"] =  2 * GIANT_COMPONENT.number_of_edges() / GIANT_COMPONENT.number_of_nodes() # Identical to: np.mean(list(dict(DEGREES).values())))
        
        local_clustering = nx.clustering(GIANT_COMPONENT, weight=None)
        row["avg_clustering"] = np.mean(np.array(list(local_clustering.values()))) # Identical to nx.average_clustering

        modularity, communities = newman_modularity(GIANT_COMPONENT, runs=100, resolution=1)
        row["modularity"] = modularity

        row["global_efficiency"] = nx.global_efficiency(GIANT_COMPONENT)

        row["avg_local_efficiency"] = nx.local_efficiency(GIANT_COMPONENT)

        row["avg_shortest_path_length"] = nx.average_shortest_path_length(GIANT_COMPONENT, weight=None, method='dijkstra')

        evc = nx.eigenvector_centrality(GIANT_COMPONENT)
        row["avg_eigenvector_centrality"] = np.mean(list(evc.values()))

        bnc = nx.betweenness_centrality(GIANT_COMPONENT)
        row["avg_betweenness_centrality"] = np.mean(list(bnc.values()))

        cnc = nx.closeness_centrality(GIANT_COMPONENT)
        row["avg_closeness_centrality"] = np.mean(list(cnc.values()))

        ebc = np.array(list(nx.edge_betweenness_centrality(GIANT_COMPONENT, weight=None).values()))
        row["avg_edge_betwenness_centrality"] = np.mean(ebc)
        row["median_edge_betwenness_centrality"] = np.median(ebc)
        row["min_edge_betwenness_centrality"] = np.min(ebc)
        row["max_edge_betwenness_centrality"] = np.max(ebc)

        pc = nx.percolation_centrality(GIANT_COMPONENT, weight=None)
        pc_ = np.array(list(pc.values()))
        row["avg_percolation_centrality"] = np.mean(pc_)
        row["median_percolation_centrality"] = np.median(pc_)
        row["min_percolation_centrality"] = np.min(pc_)
        row["max_percolation_centrality"] = np.max(pc_)

        participation = participation_coefficient(GIANT_COMPONENT, communities)
        participation_ = np.array(list(participation.values()))
        row["avg_participation_coef"] = np.mean(participation_)
        row["median_participation_coef"] = np.median(participation_)
        row["min_participation_coef"] = np.min(participation_)
        row["max_participation_coef"] = np.max(participation_)

        row["deg_assortativity"] = nx.degree_assortativity_coefficient(GIANT_COMPONENT)

        alpha, xmin, R, distribution = fit_power_law_to_degrees(list(DEGREES.values()))
        row["powerlaw_exponent"] = alpha
        row["LLR_distribution"] = R
        row["distribution"] = distribution

        row["s_metric_SF"] = nx.s_metric(GIANT_COMPONENT)

        rich_club =  nx.rich_club_coefficient(GIANT_COMPONENT, normalized=False)
        rich_club_ = np.array(list(rich_club.values()))
        row["avg_rich_club_coef"] = np.mean(rich_club_)
        row["median_rich_club_coef"] = np.median(rich_club_)
        row["min_rich_club_coef"] = np.min(rich_club_)
        row["max_rich_club_coef"] = np.max(rich_club_)

        ### NOT NORMALIZING THE RICH CLUB COEFFICIENT FOR NOW (WE ARE NOT CLAIMING THAT IT IS, THIS MIGHT BE DONE AFTERWARDS)
        #rich_club_norm = norm_rich_club_coef(GIANT_COMPONENT, rich_club, Q=100, m=50)
        #rich_club_norm_ = np.array(list(rich_club_norm.values()))
        ##row["avg_rich_club_coef_norm"] = np.mean(rich_club_)
        #row["median_rich_club_coef_norm"] = np.median(rich_club_)
        #row["min_rich_club_coef_norm"] = np.min(rich_club_)
        #row["max_rich_club_coef_norm"] = np.max(rich_club_)

        string = ""
        for i, (k,v) in enumerate(zip(row.columns, row.values[0])):
            string+=f"{v},"
        print(string[:-1])

        # Subgraph info
        labels = pd.read_csv(ATLAS, sep=" ", header=None)
        label_to_name = {}
        if "HCPEX" in ATLAS.upper():
            for node, hemisphere, ROI in zip(labels[0], labels[1], labels[2]):
                label_to_name[int(node)-1] = f"{ROI} {hemisphere}" # Nodes start from 0 and labels from 1
        elif "AAL3" in ATLAS.upper():
            for node, ROI in zip(labels[0], labels[1]):
                label_to_name[int(node)-1] = ROI # Nodes start from 0 and labels from 1
        else:
            raise Warning(f"______ Parcellation {ATLAS.split("/")[-1]} not implemented! ______")

        # Node statistics
        rows = []
        for node in GIANT_COMPONENT.nodes():
            row = {
                "Node ID (0-indexed)": node,
                "ROI": label_to_name[node],
                "Node degree": DEGREES[node],
                "Node clustering": local_clustering[node],
                "Node eigenvector centrality": evc[node],
                "Node betweenness centrality": bnc[node],
                "Node closeness centrality": cnc[node],
                "Node percolation centrality": pc[node],
                "Node participation coefficient": participation[node]
            }
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            f"{LESION_CONNECTOME_METRICS}_node-stats.csv", 
            sep=",", 
            index=False
        )
        
        # Rich club coefficient
        pd.DataFrame(
            [{"Degree k": k, "Rich club phi(k)": v} for k,v in rich_club.items()]
        ).to_csv(
            f"{LESION_CONNECTOME_METRICS}_rich-club.csv", 
            sep=",", 
            index=False
        )