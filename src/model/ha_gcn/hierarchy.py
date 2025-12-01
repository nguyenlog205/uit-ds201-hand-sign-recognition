"""
Hierarchical Graph Construction for HA-GCN
Hand-aware graph structures: SH (Structured Hand Graph) and PH (Parameterized Hand Graph)
"""

import numpy as np
import torch
from typing import Tuple, List


class HandGraphConstructor:
    """
    Construct hand-aware graphs for HA-GCN
    
    Creates:
    - SH (Structured Hand Graph): Fixed hand topology based on physical connections
    - PH (Parameterized Hand Graph): Learnable hand graph
    """
    
    def __init__(self, num_nodes: int = 27, hand_node_indices: List[int] = None):
        """
        Initialize hand graph constructor
        
        Args:
            num_nodes: Total number of nodes (default: 27 for sign language)
            hand_node_indices: List of node indices that belong to hands
                If None, assumes indices 7-26 are hand nodes (11 left + 11 right)
        """
        self.num_nodes = num_nodes
        
        if hand_node_indices is None:
            # Default: 7-16 for left hand, 17-26 for right hand
            self.hand_node_indices = list(range(7, 27))
        else:
            self.hand_node_indices = hand_node_indices
        
        # Left hand: 7-16 (11 nodes), Right hand: 17-26 (11 nodes)
        self.left_hand_indices = list(range(7, 18))  # 11 nodes
        self.right_hand_indices = list(range(17, 27))  # 11 nodes
    
    def build_structured_hand_graph(self) -> np.ndarray:
        """
        Build Structured Hand Graph (SH)
        
        SH is a fixed graph based on physical hand connections.
        Only hand nodes are connected, body nodes have no connections.
        
        Returns:
            SH: Adjacency matrix (num_nodes, num_nodes) with hand connections
        """
        SH = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        # Define hand connections based on physical structure
        # Based on sign_27_A_hands.py from reference
        # Left hand: indices 7-16 (11 nodes, wrist at 7)
        # Right hand: indices 17-26 (11 nodes, wrist at 17)
        
        # Left hand connections (wrist 7 -> hand joints 8-16)
        left_hand_edges = [
            # Wrist to finger bases
            (7, 8), (7, 9), (7, 10), (7, 11), (7, 12),
            # Finger connections (based on physical structure)
            (8, 9), (10, 11), (12, 13), (13, 14), (14, 15), (15, 16),
        ]
        
        # Right hand connections (wrist 17 -> hand joints 18-26)
        right_hand_edges = [
            # Wrist to finger bases
            (17, 18), (17, 19), (17, 20), (17, 21), (17, 22),
            # Finger connections
            (18, 19), (20, 21), (22, 23), (23, 24), (24, 25), (25, 26),
        ]
        
        # Add all hand edges (bidirectional)
        all_hand_edges = left_hand_edges + right_hand_edges
        for i, j in all_hand_edges:
            if i < self.num_nodes and j < self.num_nodes:
                SH[i, j] = 1.0
                SH[j, i] = 1.0
        
        # Add self-connections for hand nodes
        for idx in self.hand_node_indices:
            if idx < self.num_nodes:
                SH[idx, idx] = 1.0
        
        return SH
    
    def build_parameterized_hand_graph(self) -> np.ndarray:
        """
        Build Parameterized Hand Graph (PH)
        
        PH is a learnable graph initialized with hand connections.
        All elements are learnable parameters.
        
        Returns:
            PH: Initial adjacency matrix (num_nodes, num_nodes) for learnable hand graph
        """
        # Initialize PH with SH structure, but values will be learnable
        PH = self.build_structured_hand_graph()
        
        # Normalize to get better initialization
        # Add small random noise for learnability
        noise = np.random.normal(0, 0.01, PH.shape).astype(np.float32)
        PH = PH + noise
        
        # Ensure non-negative (will be learned)
        PH = np.maximum(PH, 0.0)
        
        return PH
    
    def get_hand_graphs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get both hand graphs
        
        Returns:
            SH: Structured Hand Graph (fixed)
            PH: Parameterized Hand Graph (learnable initialization)
        """
        SH = self.build_structured_hand_graph()
        PH = self.build_parameterized_hand_graph()
        return SH, PH


def create_hand_graphs_for_spatial_partitioning(
    num_nodes: int = 27,
    num_partitions: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create hand graphs compatible with spatial partitioning strategy
    
    Args:
        num_nodes: Number of nodes
        num_partitions: Number of spatial partitions (default: 3 for ST-GCN)
    
    Returns:
        SH: Structured Hand Graph stacked by partitions (num_partitions, num_nodes, num_nodes)
        PH: Parameterized Hand Graph stacked by partitions (num_partitions, num_nodes, num_nodes)
    """
    constructor = HandGraphConstructor(num_nodes=num_nodes)
    SH_base = constructor.build_structured_hand_graph()
    PH_base = constructor.build_parameterized_hand_graph()
    
    # Stack for each partition (same graph for all partitions)
    SH = np.stack([SH_base] * num_partitions, axis=0)
    PH = np.stack([PH_base] * num_partitions, axis=0)
    
    return SH, PH