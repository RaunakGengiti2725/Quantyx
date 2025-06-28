"""
Interactive 3D visualizations for quantum holographic geometry.
This module provides state-of-the-art interactive visualizations using Plotly.
"""

import os
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import deque


class InteractiveQuantumVisualizer:
    """
    Advanced interactive visualizer for quantum holographic geometry.
    Provides 3D interactive plots, animations, and real-time correlation analysis.
    """
    
    def __init__(self, theme: str = "plotly_white"):
        """Initialize with professional theme."""
        self.theme = theme
        self.config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'quantum_holographic_plot',
                'height': 1080,
                'width': 1920,
                'scale': 2
            }
        }
    
    def create_interactive_3d_bulk_tree(
        self, 
        tree: nx.Graph, 
        weights: np.ndarray, 
        outdir: str,
        title: str = "Interactive 3D Quantum Bulk Geometry"
    ) -> str:
        """
        Create an interactive 3D bulk tree visualization with rotation, zoom, and hover effects.
        """
        # Calculate optimized 3D layout
        pos3d = self._calculate_3d_layout(tree, weights)
        
        # Separate nodes by type
        leaves = [n for n in tree.nodes if tree.degree[n] == 1]
        internal_nodes = [n for n in tree.nodes if tree.degree[n] > 1]
        
        # Create figure
        fig = go.Figure()
        
        # Add edges with varying thickness based on weights
        edge_traces = self._create_edge_traces(tree, pos3d, weights)
        for trace in edge_traces:
            fig.add_trace(trace)
        
        # Add boundary qubits (leaves)
        if leaves:
            leaf_coords = np.array([pos3d[n] for n in leaves])
            fig.add_trace(go.Scatter3d(
                x=leaf_coords[:, 0],
                y=leaf_coords[:, 1],
                z=leaf_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=25,  # Increased from 15 to 25 for much larger nodes
                    color='#2E86AB',
                    opacity=0.9,
                    line=dict(width=6, color='white')  # Thicker borders
                ),
                text=[f"Boundary Qubit {n}" for n in leaves],
                hovertemplate="<b>%{text}</b><br>" +
                            "Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>" +
                            "<extra></extra>",
                name="Boundary Qubits"
            ))
        
        # Add bulk vertices (internal nodes)
        if internal_nodes:
            bulk_coords = np.array([pos3d[n] for n in internal_nodes])
            fig.add_trace(go.Scatter3d(
                x=bulk_coords[:, 0],
                y=bulk_coords[:, 1],
                z=bulk_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=30,  # Increased from 18 to 30 for much larger bulk nodes
                    color='#A23B72',
                    opacity=0.9,
                    symbol='square',
                    line=dict(width=6, color='white')  # Thicker borders
                ),
                text=[f"Bulk Vertex {n}" for n in internal_nodes],
                hovertemplate="<b>%{text}</b><br>" +
                            "Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>" +
                            "Connections: %{customdata}<br>" +
                            "<extra></extra>",
                customdata=[tree.degree[n] for n in internal_nodes],
                name="Bulk Vertices"
            ))
        
        # Ultra-modern layout
        fig.update_layout(
            template=self.theme,
            title=dict(
                text=f"<b>{title}</b><br><sub>Interactive Holographic Entanglement Structure</sub>",
                x=0.5,
                font=dict(size=24, family="Arial Black")
            ),
            scene=dict(
                xaxis_title="Radial Depth (AdS Coordinate)",
                yaxis_title="Boundary Coordinate Y",
                zaxis_title="Boundary Coordinate Z",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor='rgba(245,245,245,0.8)',
                xaxis=dict(gridcolor='rgba(200,200,200,0.5)', showbackground=True),
                yaxis=dict(gridcolor='rgba(200,200,200,0.5)', showbackground=True),
                zaxis=dict(gridcolor='rgba(200,200,200,0.5)', showbackground=True),
                aspectmode='cube'
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, b=0, t=60),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            )
        )
        
        # Add annotations
        fig.add_annotation(
            text="Einstein Equation: G<sub>μν</sub> = 8πGT<sub>μν</sub>",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            font=dict(size=14, color="rgba(0,0,0,0.7)"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        )
        
        # Save interactive plot
        os.makedirs(outdir, exist_ok=True)
        filename = os.path.join(outdir, "interactive_3d_bulk_tree.html")
        fig.write_html(filename, config=self.config)
        
        return filename
    
    def create_animated_correlation_evolution(
        self,
        times: np.ndarray,
        correlations: List[float],
        outdir: str,
        title: str = "Einstein Correlation Evolution"
    ) -> str:
        """
        Create animated correlation evolution showing time-dependent physics.
        """
        # Create frames for animation
        frames = []
        for i, (t, corr) in enumerate(zip(times, correlations)):
            frame_data = [
                go.Scatter(
                    x=times[:i+1],
                    y=correlations[:i+1],
                    mode='lines+markers',
                    line=dict(color='#2E86AB', width=4),
                    marker=dict(size=8, color='#A23B72', line=dict(width=2, color='white')),
                    name='Einstein Correlation'
                ),
                go.Scatter(
                    x=[t],
                    y=[corr],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Current Time'
                )
            ]
            frames.append(go.Frame(data=frame_data, name=f"t={t:.2f}"))
        
        # Initial plot
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=[times[0]],
                    y=[correlations[0]],
                    mode='lines+markers',
                    line=dict(color='#2E86AB', width=4),
                    marker=dict(size=8, color='#A23B72'),
                    name='Einstein Correlation'
                )
            ],
            frames=frames
        )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
        fig.add_hline(y=0.7, line_dash="dot", line_color="green", opacity=0.7)
        fig.add_hline(y=-0.7, line_dash="dot", line_color="green", opacity=0.7)
        
        # Animation controls
        fig.update_layout(
            template=self.theme,
            title=dict(
                text=f"<b>{title}</b><br><sub>Time Evolution of Curvature-Energy Correspondence</sub>",
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title="Evolution Time (ħ/J)",
            yaxis_title="Correlation Coefficient r",
            xaxis=dict(range=[times.min(), times.max()]),
            yaxis=dict(range=[-1.1, 1.1]),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'x': 0.1,
                'y': 1.02,
                'buttons': [
                    {
                        'label': '▶ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 800, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300}
                        }]
                    },
                    {
                        'label': '⏸ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[f"t={t:.2f}"], {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }],
                        'label': f"t={t:.2f}",
                        'method': 'animate'
                    } for t in times
                ],
                'active': 0,
                'x': 0.1,
                'len': 0.9,
                'y': 0,
                'currentvalue': {
                    'visible': True,
                    'prefix': "Time: ",
                    'xanchor': "right",
                    'font': {'size': 16}
                }
            }],
            width=1000,
            height=600
        )
        
        # Save animated plot
        filename = os.path.join(outdir, "animated_correlation_evolution.html")
        fig.write_html(filename, config=self.config)
        
        return filename
    
    def create_interactive_dashboard(
        self,
        times: np.ndarray,
        correlations: List[float],
        entropies: np.ndarray,
        weights_history: List[np.ndarray],
        outdir: str
    ) -> str:
        """
        Create a comprehensive interactive dashboard with multiple linked plots.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Einstein Correlation Evolution", "Entanglement Entropy Dynamics",
                          "Weight Distribution", "Correlation vs Entropy"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Correlation evolution
        fig.add_trace(
            go.Scatter(
                x=times,
                y=correlations,
                mode='lines+markers',
                name='Einstein Correlation',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=6, color='#A23B72')
            ),
            row=1, col=1
        )
        
        # 2. Entropy dynamics
        for i in range(min(3, entropies.shape[1])):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=entropies[:, i],
                    mode='lines',
                    name=f'Region {i+1}',
                    line=dict(width=2)
                ),
                row=1, col=2
            )
        
        # 3. Weight distribution (latest)
        if weights_history:
            latest_weights = weights_history[-1]
            fig.add_trace(
                go.Histogram(
                    x=latest_weights,
                    nbinsx=20,
                    name='Weight Distribution',
                    marker_color='#F18F01',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # 4. Correlation vs Entropy scatter
        avg_entropy = np.mean(entropies, axis=1)
        fig.add_trace(
            go.Scatter(
                x=avg_entropy,
                y=correlations,
                mode='markers',
                name='Correlation vs Entropy',
                marker=dict(
                    size=10,
                    color=times,
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="Time")
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            template=self.theme,
            title=dict(
                text="<b>Quantum Holographic Geometry Dashboard</b>",
                x=0.5,
                font=dict(size=24)
            ),
            height=800,
            showlegend=True
        )
        
        # Save dashboard
        filename = os.path.join(outdir, "interactive_dashboard.html")
        fig.write_html(filename, config=self.config)
        
        return filename
    
    def _calculate_3d_layout(self, tree: nx.Graph, weights: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        """Calculate optimized 3D layout for tree visualization."""
        # Find root for hierarchical layout
        leaves = [n for n in tree.nodes if tree.degree[n] == 1]
        total_leaves = len(leaves)
        
        # Find best root (node with most leaf descendants)
        root = None
        max_descendants = 0
        for node in tree.nodes:
            if tree.degree[node] > 1:
                # Count reachable leaves
                visited = {node}
                queue = deque([node])
                leaf_count = 0
                while queue:
                    curr = queue.popleft()
                    for nbr in tree.neighbors(curr):
                        if nbr in visited:
                            continue
                        visited.add(nbr)
                        if nbr in leaves:
                            leaf_count += 1
                        else:
                            queue.append(nbr)
                if leaf_count > max_descendants:
                    max_descendants = leaf_count
                    root = node
        
        if root is None:
            root = list(tree.nodes)[0]
        
        # Create hierarchical layout
        depth_map = nx.single_source_shortest_path_length(tree, source=root)
        max_depth = max(depth_map.values())
        
        # Group nodes by depth
        coords_by_depth = {}
        for node, depth in depth_map.items():
            coords_by_depth.setdefault(depth, []).append(node)
        
        pos3d = {}
        layer_spread = 80  # MASSIVE increase from 40 to 80 for ultra-wide interactive 3D
        
        for depth, nodes_in_depth in coords_by_depth.items():
            n_nodes = len(nodes_in_depth)
            if n_nodes == 1:
                pos3d[nodes_in_depth[0]] = (depth * layer_spread, 0, 0)
            else:
                if n_nodes <= 8:
                    # ULTRA-MASSIVE circular arrangement
                    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
                    radius = 50 + depth * 15  # HUGE radius increase for ultra-wide interactive circles
                    for i, node in enumerate(nodes_in_depth):
                        y = radius * np.cos(angles[i])
                        z = radius * np.sin(angles[i])
                        pos3d[node] = (depth * layer_spread, y, z)
                else:
                    # ULTRA-MASSIVE grid arrangement
                    grid_size = int(np.ceil(np.sqrt(n_nodes)))
                    spacing = 40  # HUGE spacing increase from 18 to 40 for ultra-wide grids
                    start_offset = -(grid_size - 1) * spacing / 2
                    for i, node in enumerate(nodes_in_depth):
                        row = i // grid_size
                        col = i % grid_size
                        y = start_offset + col * spacing
                        z = start_offset + row * spacing
                        pos3d[node] = (depth * layer_spread, y, z)
        
        # Center around origin
        if pos3d:
            xs, ys, zs = zip(*pos3d.values())
            x_center, y_center, z_center = np.mean(xs), np.mean(ys), np.mean(zs)
            pos3d = {node: (x - x_center, y - y_center, z - z_center) 
                    for node, (x, y, z) in pos3d.items()}
        
        return pos3d
    
    def _create_edge_traces(self, tree: nx.Graph, pos3d: Dict, weights: np.ndarray) -> List[go.Scatter3d]:
        """Create edge traces with varying thickness and color based on weights."""
        traces = []
        
        # Normalize weights for coloring
        if len(weights) > 0:
            norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        else:
            norm_weights = np.ones(len(tree.edges))
        
        # Create edge traces
        for i, (u, v) in enumerate(tree.edges):
            if u in pos3d and v in pos3d:
                x0, y0, z0 = pos3d[u]
                x1, y1, z1 = pos3d[v]
                
                # ULTRA-THICK edge thickness based on weight
                if i < len(weights):
                    thickness = 8 + 20 * norm_weights[i]  # Much thicker: 8-28 range instead of 3-10
                    color_intensity = norm_weights[i]
                else:
                    thickness = 15  # Tripled default thickness from 5 to 15
                    color_intensity = 0.5
                
                # Color based on weight
                color = f'rgba({int(255 * color_intensity)}, {int(100 + 155 * (1-color_intensity))}, {int(200 * color_intensity)}, 0.8)'
                
                trace = go.Scatter3d(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[z0, z1, None],
                    mode='lines',
                    line=dict(color=color, width=thickness),
                    hoverinfo='skip',
                    showlegend=False
                )
                traces.append(trace)
        
        return traces 