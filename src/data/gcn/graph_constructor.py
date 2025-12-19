"""
Graph Constructor Module
Convert pose sequences into graph representations for GCN models
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json


class SkeletonGraph:
    """
    Skeleton graph structure definition
    Defines the connectivity (edges) between body joints (nodes)
    """
    
    def __init__(self, layout: str = "openpose_body25"):
        """
        Initialize skeleton graph layout
        
        Args:
            layout: "openpose_body25", "openpose_coco", "openpose_mpi", or "custom"
        """
        self.layout = layout
        self.num_nodes, self.edges, self.center = self._get_layout(layout)
        self.A = self._build_adjacency_matrix()
        
    def _get_layout(self, layout: str) -> Tuple[int, List[Tuple], int]:
        """
        Get skeleton layout configuration
        
        Returns:
            num_nodes: Number of joints
            edges: List of (parent, child) joint connections
            center: Index of center joint
        """
        if layout == "mediapipe_27":
            # 27-point layout: 5 body + 11 left hand + 11 right hand
            # 0: Neck (computed from shoulders), 1-2: Shoulders, 3-4: Elbows
            # 5-15: Left hand (11 points), 16-26: Right hand (11 points)
            num_nodes = 27
            center = 0  # Neck as center
            
            edges = [
                # Body connectivity
                (0, 1), (0, 2),    # Neck -> Shoulders
                (1, 3), (2, 4),    # Shoulders -> Elbows
                
                # Arm-hand connection
                (3, 5),   # Left Elbow -> Left Wrist
                (4, 16),  # Right Elbow -> Right Wrist
                
                # Left Hand (11 points: wrist + 2 per finger)
                (5, 6), (6, 7),    # Thumb (MCP -> Tip)
                (5, 8), (8, 9),    # Index
                (5, 10), (10, 11), # Middle
                (5, 12), (12, 13), # Ring
                (5, 14), (14, 15), # Pinky
                
                # Right Hand (11 points)
                (16, 17), (17, 18), # Thumb
                (16, 19), (19, 20), # Index
                (16, 21), (21, 22), # Middle
                (16, 23), (23, 24), # Ring
                (16, 25), (25, 26)  # Pinky
            ]
            
        return num_nodes, edges, center
    
    def _build_adjacency_matrix(self) -> np.ndarray:
        """
        Build adjacency matrix from edge list
        
        Returns:
            A: Adjacency matrix of shape (num_nodes, num_nodes)
        """
        A = np.zeros((self.num_nodes, self.num_nodes))
        
        # Add edges (bidirectional)
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1
            
        # Add self-connections
        A = A + np.eye(self.num_nodes)
        
        return A
    
    def get_adjacency_matrix(self, strategy="spatial"):
        # Input: self.edges 
        # Output: A shape (3, num_nodes, num_nodes)
        
        num_node = self.num_nodes
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i, j) for i, j in self.edges]
        # Add reverse edges for undirected graph
        neighbor_link_sym = neighbor_link + [(j, i) for i, j in neighbor_link] 
        
        # 1. Build basic adjacency matrix
        A_binary = np.zeros((num_node, num_node))
        for i, j in neighbor_link_sym + self_link:
            A_binary[i, j] = 1
            
        if strategy == "uniform":
            D = np.sum(A_binary, axis=1)
            D[D == 0] = 1
            D_inv = np.diag(1.0 / D)
            A_norm = D_inv @ A_binary
            return np.stack([A_norm], axis=0) # (1, V, V)

        elif strategy == "spatial":
            # --- Standard ST-GCN (3 PARTITIONS) ---
            
            # Bước A: Calculate distance from each node to center (Center)
            # Mặc định dùng self.center (thường là 0 - Neck hoặc 23 - MidHip)
            # Khoảng cách ở đây tính theo "số bước nhảy" trên đồ thị (Hop distance)
            # Tuy nhiên, để đơn giản và nhanh, ta dùng định nghĩa topology cứng:
            # Node nào gần trung tâm (Neck/Hip) hơn thì là Centripetal.
            
            # Calculate shortest path distance (Hop distance) from center
            dist_center = np.zeros(num_node)
            for i in range(num_node):
                # Use simple BFS to calculate hop from node i to self.center
                dist_center[i] = self._get_hop_distance(i, self.center)
            
            # Bước B: Create 3 empty matrices
            # 0: Root (Chính nó)
            # 1: Centripetal (Hàng xóm gần tâm hơn)
            # 2: Centrifugal (Hàng xóm xa tâm hơn)
            A = np.zeros((3, num_node, num_node))
            
            for i, j in neighbor_link_sym + self_link:
                if i == j:
                    # Group 0: Root (Self-loop)
                    A[0, i, j] = 1
                elif dist_center[j] < dist_center[i]:
                    # Group 1: Neighbor j is closer to center than node i -> Centripetal
                    A[1, i, j] = 1
                else:
                    # Group 2: Neighbor j is farther from center than or equal to node i -> Centrifugal
                    A[2, i, j] = 1
            
            # Bước C: Normalize each partition
            # Each row of each partition must sum to 1 (or approximately)
            for p in range(3):
                # Count degree of node i in the entire graph 
                D = np.sum(A_binary, axis=1) 
                D[D == 0] = 1
                D_inv = np.diag(1.0 / D)
                A[p] = D_inv @ A[p]
                
            return A # Output: (3, 27, 27)

    def _get_hop_distance(self, start_node, end_node):
        # BFS to find shortest distance between 2 nodes
        if start_node == end_node: return 0
        visited = {start_node}
        queue = [(start_node, 0)]
        
        # Build adjacency list temporarily for BFS
        adj = {i: [] for i in range(self.num_nodes)}
        for u, v in self.edges:
            adj[u].append(v)
            adj[v].append(u)
            
        while queue:
            curr, dist = queue.pop(0)
            if curr == end_node: return dist
            
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')


class GraphConstructor:
    """
    Constructs graph data from pose sequences
    """
    
    def __init__(self, skeleton_layout: str = "mediapipe_27", normalize: bool = True):
        self.skeleton_layout = skeleton_layout
        self.skeleton = SkeletonGraph(skeleton_layout)
        self.normalize = normalize
        
        # --- Indices Mapping MediaPipe -> MediaPipe 27 subset ---
        # MediaPipe Hand (21 points): 
        # 0:Wrist, 1-4:Thumb, 5-8:Index, 9-12:Middle, 13-16:Ring, 17-20:Pinky
        self.mp_hand_indices = [0, 2, 4, 5, 8, 9, 12, 13, 16, 17, 20]
        
    def construct_graph_from_poses(
        self,
        poses_dict: Dict,
        strategy: str = "spatial",
        masks: Optional[np.ndarray] = None,
    ):
        # Input: poses_dict từ PoseExtractor
        # Output: Graph dictionary với 'x' shape (T, 27, 3)
        
        body_raw = poses_dict['body']       # (T, 33, 3)
        l_hand_raw = poses_dict['left_hand']  # (T, 21, 3)
        r_hand_raw = poses_dict['right_hand'] # (T, 21, 3)
        
        T = body_raw.shape[0]
        
        # --- 1. PREPARE DATA CONTAINERS ---
        # Chúng ta sẽ tạo ra node features (T, 27, 3)
        final_poses = np.zeros((T, 27, 3))
        
        # --- 2. EXTRACT BODY POINTS (5 points) ---
        # Node 0: Neck (Tính trung điểm 2 vai: MP 11 & 12)
        neck = (body_raw[:, 11] + body_raw[:, 12]) / 2.0
        final_poses[:, 0] = neck
        
        # Node 1, 2: Shoulders (MP 11, 12)
        final_poses[:, 1] = body_raw[:, 11] # Left Shoulder
        final_poses[:, 2] = body_raw[:, 12] # Right Shoulder
        
        # Node 3, 4: Elbows (MP 13, 14)
        final_poses[:, 3] = body_raw[:, 13] # Left Elbow
        final_poses[:, 4] = body_raw[:, 14] # Right Elbow
        
        # --- 3. EXTRACT HAND POINTS (11 points each) ---
        # Left Hand -> Nodes 5-15
        final_poses[:, 5:16] = l_hand_raw[:, self.mp_hand_indices]
        
        # Right Hand -> Nodes 16-26
        final_poses[:, 16:27] = r_hand_raw[:, self.mp_hand_indices]
        
        # --- 4. NORMALIZE ---
        if self.normalize:
            final_poses = self._normalize_poses(final_poses, body_raw)
            
        # --- 5. BUILD GRAPH ---
        A = self.skeleton.get_adjacency_matrix(strategy)
        edge_index = self._adjacency_to_edge_index(A)
        edge_attr = self._compute_edge_attributes(final_poses, edge_index, masks)
        
        return {
            'x': torch.from_numpy(final_poses).float(),
            'adj_matrix': torch.from_numpy(A).float(),
            'edge_index': torch.from_numpy(edge_index).long(),
            'edge_attr': torch.from_numpy(edge_attr).float(),
        }
    
    def _convert_mediapipe_to_mediapipe27(self, poses_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert MediaPipe format to mediapipe_27 format
        
        Args:
            poses_dict: Dictionary from MediaPipePoseExtractor
                - 'body': (T, 33, 3)
                - 'left_hand': (T, 21, 3)
                - 'right_hand': (T, 21, 3)
        
        Returns:
            poses: (T, 27, 3) array in mediapipe_27 format
        """
        body_raw = poses_dict['body']       # (T, 33, 3)
        l_hand_raw = poses_dict['left_hand']  # (T, 21, 3)
        r_hand_raw = poses_dict['right_hand'] # (T, 21, 3)
        
        T = body_raw.shape[0]
        final_poses = np.zeros((T, 27, 3))
        
        # 1. Extract Body Points (5 points)
        # Node 0: Neck (average of shoulders: MP 11 & 12)
        neck = (body_raw[:, 11] + body_raw[:, 12]) / 2.0
        final_poses[:, 0] = neck
        
        # Node 1, 2: Shoulders (MP 11, 12)
        final_poses[:, 1] = body_raw[:, 11]  # Left Shoulder
        final_poses[:, 2] = body_raw[:, 12]  # Right Shoulder
        
        # Node 3, 4: Elbows (MP 13, 14)
        final_poses[:, 3] = body_raw[:, 13]  # Left Elbow
        final_poses[:, 4] = body_raw[:, 14]  # Right Elbow
        
        # 2. Extract Hand Points (11 points each)
        # Left Hand -> Nodes 5-15
        final_poses[:, 5:16] = l_hand_raw[:, self.mp_hand_indices]
        
        # Right Hand -> Nodes 16-26
        final_poses[:, 16:27] = r_hand_raw[:, self.mp_hand_indices]
        
        return final_poses
    
    def _normalize_poses(self, poses, body_raw):
        """
        Normalize poses based on the 27-keypoint structure (not body_raw)
        
        Args:
            poses: (T, 27, 3) - already converted to 27 keypoints
            body_raw: (T, 33, 3) - only used to get hip reference for center calculation
        
        Returns:
            Normalized poses (T, 27, 3)
        """
        poses = poses.copy()
        coords = poses[..., :2]  # (T, 27, 2)
        
        # 1. Calculate center from poses itself (use neck node 0 as center)
        # Node 0 is neck, which is more stable than hip for sign language
        center = coords[:, 0:1, :]  # (T, 1, 2) - use neck as center
        
        # Subtract center (normalize to neck-centered coordinates)
        coords = coords - center
        
        # 2. Scale based on shoulder width from poses (nodes 1 and 2 are shoulders)
        left_sh = coords[:, 1, :]   # (T, 2) - left shoulder
        right_sh = coords[:, 2, :]  # (T, 2) - right shoulder
        shoulder_width = np.linalg.norm(left_sh - right_sh, axis=-1)  # (T,)
        
        # Use median shoulder width for stable scaling
        scale = np.median(shoulder_width[shoulder_width > 0]) if np.any(shoulder_width > 0) else 1.0
        
        # Normalize with padding factor
        if scale > 0:
            poses[..., :2] = coords / (scale * 1.2)  # 1.2 padding factor
        else:
            # Fallback: use body_raw hip if shoulder width is invalid
            left_hip = body_raw[:, 23, :2] if body_raw.shape[1] > 23 else coords[:, 0, :]
            right_hip = body_raw[:, 24, :2] if body_raw.shape[1] > 24 else coords[:, 0, :]
            hip_center = (left_hip + right_hip) / 2.0
            coords = coords - hip_center.reshape(-1, 1, 2)
            hip_width = np.linalg.norm(left_hip - right_hip, axis=-1)
            scale = np.median(hip_width[hip_width > 0]) if np.any(hip_width > 0) else 1.0
            if scale > 0:
                poses[..., :2] = coords / (scale * 1.2)
        
        return poses
    
    def _adjacency_to_edge_index(self, A: np.ndarray) -> np.ndarray:
        """
        Convert adjacency matrix to edge index format.
        Supports either a single adjacency (V, V) or ST-GCN partitions (K, V, V).
        """
        if A.ndim == 3:
            merged = np.sum(A, axis=0)
        else:
            merged = A

        edges = np.array(np.where(merged > 0))  # (2, E)
        return edges
    
    def _compute_edge_attributes(
        self,
        poses: np.ndarray,
        edge_index: np.ndarray,
        masks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute edge attributes (e.g., distance between joints)
        
        Args:
            poses: (T, V, C) pose sequence
            edge_index: (2, E) edge connections
            masks: Optional (T, V) boolean mask indicating valid keypoints
            
        Returns:
            edge_attr: (T, E, 1) edge attributes
        """
        T, V, C = poses.shape
        E = edge_index.shape[1]
        
        # Get source and target node coordinates
        src_coords = poses[:, edge_index[0], :2]  # (T, E, 2) - only x, y
        tgt_coords = poses[:, edge_index[1], :2]  # (T, E, 2)
        
        # Compute Euclidean distance
        distances = np.linalg.norm(src_coords - tgt_coords, axis=-1, keepdims=True)  # (T, E, 1)
        
        # Apply mask if provided (set distance to 0 for invalid edges)
        if masks is not None:
            src_mask = masks[:, edge_index[0]]  # (T, E)
            tgt_mask = masks[:, edge_index[1]]  # (T, E)
            valid_edge = src_mask & tgt_mask  # (T, E)
            distances[~valid_edge] = 0.0
        
        return distances