import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from collections import deque
import seaborn as sns

# Set up ultra-modern, clean plotting style
plt.style.use('default')
sns.set_palette("Set2")

# Ultra-professional matplotlib configuration
plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],  # Use available fonts
    'font.size': 16,
    'axes.titlesize': 22,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 24,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
    'axes.grid': False,  # We'll add custom grids
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.5,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'legend.framealpha': 1.0,
    'legend.fancybox': True,
    'legend.shadow': True,
    'legend.frameon': True,
    'figure.facecolor': 'white'
})


def plot_bulk_tree(tree: nx.Graph, weights: np.ndarray, outdir: str):
    """
    Ultra-modern 2D Bulk Tree visualization with clean, spacious design.
    """
    # Create ultra-large, ultra-clean figure
    fig, ax = plt.subplots(1, 1, figsize=(24, 18), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Use sophisticated layout algorithm to prevent overlaps
    try:
        # Try different layout algorithms for best spacing
        pos = nx.nx_agraph.graphviz_layout(tree, prog="neato", args="-Goverlap=false -Gsplines=true")
    except Exception:
        try:
            pos = nx.spring_layout(tree, k=15, iterations=500, seed=42)
        except:
            pos = nx.circular_layout(tree)
    
    # MASSIVE scaling for ultra-clean spacing
    scale_factor = 80  # Increased from 50 to 80 for much more spacing
    pos = {node: (x * scale_factor, y * scale_factor) for node, (x, y) in pos.items()}
    
    # Ultra-sophisticated node separation algorithm
    def separate_overlapping_nodes(pos, min_distance=60):  # Increased from 25 to 60
        """Advanced algorithm to separate overlapping nodes"""
        pos_array = np.array(list(pos.values()))
        nodes = list(pos.keys())
        
        for iteration in range(100):  # More iterations for better separation
            moved = False
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes[i+1:], i+1):
                    pos_i = np.array(pos[node_i])
                    pos_j = np.array(pos[node_j])
                    dist = np.linalg.norm(pos_i - pos_j)
                    
                    if dist < min_distance and dist > 0:
                        # Move nodes apart
                        direction = (pos_i - pos_j) / dist
                        move_amount = (min_distance - dist) / 2
                        pos[node_i] = tuple(pos_i + direction * move_amount)
                        pos[node_j] = tuple(pos_j - direction * move_amount)
                        moved = True
            
            if not moved:
                break
        
        return pos
    
    pos = separate_overlapping_nodes(pos, min_distance=80)  # Increased from 40 to 80
    
    # Create ultra-sophisticated node labels
    node_list = list(tree.nodes())
    leaves = [n for n in tree.nodes() if tree.degree[n] == 1]
    internal_nodes = [n for n in tree.nodes() if tree.degree[n] > 1]
    
    # Clean, minimal labeling
    labels = {}
    for i, node in enumerate(node_list):
        if node in leaves and isinstance(node, str) and node.startswith('q'):
            labels[node] = f"q{node[1:]}"
        else:
            labels[node] = f"v{i}"
    
    # Ultra-modern colormap
    norm = plt.Normalize(vmin=np.min(weights), vmax=np.max(weights))
    cmap = plt.cm.get_cmap('plasma')
    edge_colors = [cmap(norm(w)) for w in weights]
    
    # Ultra-clean background
    ax.set_facecolor('#FAFAFA')
    
    # Draw ultra-modern edges with gradient effects
    weight_range = np.max(weights) - np.min(weights)
    if weight_range > 1e-10:
        edge_widths = 6 + 16 * (weights - np.min(weights)) / weight_range
    else:
        edge_widths = np.full_like(weights, 12.0)
    
    for i, (u, v) in enumerate(tree.edges()):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # Ultra-subtle shadow for depth
        ax.plot([x1, x2], [y1, y2], color='#E0E0E0', linewidth=edge_widths[i]+4, 
                alpha=0.3, zorder=1, solid_capstyle='round')
        
        # Main ultra-modern edge
        ax.plot([x1, x2], [y1, y2], color=edge_colors[i], 
                linewidth=edge_widths[i], alpha=0.95, zorder=2, solid_capstyle='round')
    
    # Ultra-large, ultra-modern nodes
    leaf_positions = {k: v for k, v in pos.items() if k in leaves}
    internal_positions = {k: v for k, v in pos.items() if k in internal_nodes}
    
    if leaf_positions:
        leaf_x, leaf_y = zip(*leaf_positions.values())
        ax.scatter(leaf_x, leaf_y, s=3000, c='#2E86AB', alpha=0.9, 
                  edgecolors='white', linewidths=6, zorder=4, 
                  marker='o', label='Boundary Qubits')
    
    if internal_positions:
        int_x, int_y = zip(*internal_positions.values())
        ax.scatter(int_x, int_y, s=4000, c='#A23B72', alpha=0.9,
                  edgecolors='white', linewidths=6, zorder=4,
                  marker='s', label='Bulk Vertices')
    
    # Ultra-modern labels with perfect spacing
    for node, (x, y) in pos.items():
        bbox_props = dict(boxstyle="round,pad=0.8", facecolor='white', 
                         edgecolor='#333333', alpha=0.95, linewidth=2)
        ax.text(x, y, labels[node], fontsize=20, fontweight='600',
                ha='center', va='center', bbox=bbox_props, zorder=5,
                color='#333333')
    
    # Ultra-modern colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, aspect=25)
    cbar.set_label('Entanglement Weight', fontsize=20, fontweight='600', labelpad=20)
    cbar.ax.tick_params(labelsize=18, width=2, length=6)
    cbar.outline.set_linewidth(2)
    
    # Ultra-modern title with perfect spacing
    ax.set_title('Quantum Bulk Geometry\nEntanglement-Weighted Tree Structure', 
                fontsize=28, fontweight='300', pad=50, color='#2C2C2C')
    
    # Ultra-modern research annotation
    textstr = r'$\mathcal{G} = (V, E, w)$ where $w_e \propto S_{\partial A}$'
    props = dict(boxstyle='round,pad=1.0', facecolor='#F8F8F8', 
                edgecolor='#CCCCCC', alpha=0.9, linewidth=2)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props, color='#444444', fontweight='400')
    
    # Remove axes for ultra-clean look
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Ultra-modern legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.92), 
                      fontsize=18, framealpha=1.0, shadow=True,
                      facecolor='white', edgecolor='#CCCCCC')
    legend.get_frame().set_boxstyle('round,pad=0.8')
    legend.get_frame().set_linewidth(2)
    
    # ULTRA-MASSIVE margins for perfect spacing
    all_x = [pos[node][0] for node in tree.nodes()]
    all_y = [pos[node][1] for node in tree.nodes()]
    margin_x = (max(all_x) - min(all_x)) * 0.8  # 80% margin! (increased from 60%)
    margin_y = (max(all_y) - min(all_y)) * 0.8  # 80% margin! (increased from 60%)
    ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
    ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)
    
    # Save ultra-high-quality outputs
    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bulk_tree_research.png"), 
                bbox_inches='tight', facecolor='white', dpi=300, 
                edgecolor='none', pad_inches=0.5)
    plt.show()
    plt.close()


def plot_bulk_tree_3d(tree: nx.Graph, weights: np.ndarray, outdir: str = "figures", *, annotate: bool = True) -> None:
    """
    Ultra-modern 3D Bulk Tree visualization with sophisticated spacing.
    """
    # Create ultra-large professional 3D figure
    fig = plt.figure(figsize=(28, 20), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#FAFAFA')
    
    # Find root for hierarchical layout
    descendants = _compute_leaf_descendants(tree)
    total_leaves = len([n for n in tree.nodes if tree.degree[n] == 1])
    root = None
    for node, leaf_list in descendants.items():
        if len(leaf_list) == total_leaves:
            root = node
            break
    if root is None:
        root = next((n for n in tree.nodes if tree.degree[n] > 1), list(tree.nodes)[0])
    
    # Create ultra-sophisticated 3D layout with MASSIVE spacing
    depth_map = nx.single_source_shortest_path_length(tree, source=root)
    max_depth = max(depth_map.values())
    
    coords_by_depth = {}
    for node, depth in depth_map.items():
        coords_by_depth.setdefault(depth, []).append(node)
    
    pos3d = {}
    layer_spread = 100  # MASSIVE increase from 50 to 100 for ultra-wide 3D spacing
    
    for depth, nodes_in_depth in coords_by_depth.items():
        n_nodes = len(nodes_in_depth)
        if n_nodes == 1:
            pos3d[nodes_in_depth[0]] = (depth * layer_spread, 0, 0)
        else:
            if n_nodes <= 8:
                # ULTRA-MASSIVE circular arrangement with perfect spacing
                angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
                radius = 60 + depth * 20  # HUGE radius increase for ultra-wide circular layouts
                for i, node in enumerate(nodes_in_depth):
                    y = radius * np.cos(angles[i])
                    z = radius * np.sin(angles[i])
                    pos3d[node] = (depth * layer_spread, y, z)
            else:
                # ULTRA-MASSIVE grid arrangement with perfect spacing
                grid_size = int(np.ceil(np.sqrt(n_nodes)))
                spacing = 50  # HUGE spacing increase from 25 to 50 for ultra-wide grids
                start_offset = -(grid_size - 1) * spacing / 2
                for i, node in enumerate(nodes_in_depth):
                    row = i // grid_size
                    col = i % grid_size
                    y = start_offset + col * spacing
                    z = start_offset + row * spacing
                    pos3d[node] = (depth * layer_spread, y, z)
    
    # CENTER ALL COORDINATES AROUND ORIGIN
    xs = np.array([pos3d[n][0] for n in tree.nodes])
    ys = np.array([pos3d[n][1] for n in tree.nodes])
    zs = np.array([pos3d[n][2] for n in tree.nodes])
    
    # Calculate centroids and recenter everything
    x_center = xs.mean()
    y_center = ys.mean()
    z_center = zs.mean()
    
    # Shift all coordinates to center around origin
    for node in pos3d:
        x, y, z = pos3d[node]
        pos3d[node] = (x - x_center, y - y_center, z - z_center)
    
    # Recalculate coordinates after centering
    xs = np.array([pos3d[n][0] for n in tree.nodes])
    ys = np.array([pos3d[n][1] for n in tree.nodes])
    zs = np.array([pos3d[n][2] for n in tree.nodes])
    
    # Ultra-modern edge visualization
    norm = plt.Normalize(vmin=np.min(weights), vmax=np.max(weights))
    cmap = plt.cm.get_cmap('plasma')
    
    # Draw ultra-modern edges
    for idx, (u, v) in enumerate(tree.edges()):
        xu, yu, zu = pos3d[u]
        xv, yv, zv = pos3d[v]
        
        weight_range = np.max(weights) - np.min(weights)
        if weight_range > 1e-10:
            thickness = 12 + 28 * (weights[idx] - np.min(weights)) / weight_range  # Much thicker edges
        else:
            thickness = 20.0  # Double the default thickness
        
        # Ultra-subtle shadow
        ax.plot([xu, xv], [yu, yv], [zu, zv], 
                color='#E0E0E0', linewidth=thickness+4, alpha=0.4, zorder=1)
        # Main ultra-modern edge
        ax.plot([xu, xv], [yu, yv], [zu, zv], 
                color=cmap(norm(weights[idx])), linewidth=thickness, alpha=0.95, zorder=2)
    
    # Ultra-large, ultra-modern nodes
    leaves = [n for n in tree.nodes if tree.degree[n] == 1]
    internal_nodes = [n for n in tree.nodes if tree.degree[n] > 1]
    
    if leaves:
        leaf_coords = np.array([pos3d[n] for n in leaves])
        ax.scatter(leaf_coords[:, 0], leaf_coords[:, 1], leaf_coords[:, 2], 
                  s=2400, c='#2E86AB', alpha=0.9, edgecolors='white',  # Doubled size from 1200 to 2400
                  linewidths=8, depthshade=True, label='Boundary Qubits')  # Thicker borders
    
    if internal_nodes:
        bulk_coords = np.array([pos3d[n] for n in internal_nodes])
        ax.scatter(bulk_coords[:, 0], bulk_coords[:, 1], bulk_coords[:, 2], 
                  s=3000, c='#A23B72', alpha=0.9, edgecolors='white',  # Doubled size from 1500 to 3000
                  linewidths=8, depthshade=True, marker='s', label='Bulk Vertices')  # Thicker borders
    
    # Ultra-modern axis styling
    ax.set_xlabel('Radial Depth (AdS)', fontsize=22, fontweight='300', labelpad=25, color='#444444')
    ax.set_ylabel('Boundary Y', fontsize=22, fontweight='300', labelpad=25, color='#444444')
    ax.set_zlabel('Boundary Z', fontsize=22, fontweight='300', labelpad=25, color='#444444')
    
    # Ultra-modern title
    ax.set_title('3D Quantum Bulk Geometry\nHolographic Entanglement Structure', 
                fontsize=32, fontweight='200', pad=60, color='#2C2C2C')
    
    # PERFECTLY CENTERED padding with equal proportions
    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    z_range = zs.max() - zs.min()
    max_range = max(x_range, y_range, z_range)
    
    # Use equal ranges for all axes to maintain proper proportions
    padding = max_range * 1.2  # MASSIVE 120% padding on all sides for ultra-spacious 3D
    
    ax.set_xlim(-max_range/2 - padding, max_range/2 + padding)
    ax.set_ylim(-max_range/2 - padding, max_range/2 + padding)
    ax.set_zlim(-max_range/2 - padding, max_range/2 + padding)
    
    # Ensure equal aspect ratio for perfect centering
    ax.set_box_aspect([1,1,1])
    
    # Ultra-modern colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, aspect=20)
    cbar.set_label('Entanglement Weight', fontsize=22, fontweight='300', labelpad=25)
    cbar.ax.tick_params(labelsize=20, width=2, length=8)
    cbar.outline.set_linewidth(2)
    
    # Ultra-modern annotations
    if annotate:
        textstr = r'$G_{\mu\nu} = 8\pi T_{\mu\nu}$' + '\n' + r'$\Delta E \propto \mathcal{R}$'
        ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=22, 
                 verticalalignment='top', fontweight='300', color='#444444',
                 bbox=dict(boxstyle='round,pad=1.0', facecolor='#F8F8F8', 
                          edgecolor='#CCCCCC', alpha=0.9, linewidth=2))
    
    # Perfect viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Ultra-modern legend
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), 
                      fontsize=20, framealpha=1.0, shadow=True,
                      facecolor='white', edgecolor='#CCCCCC')
    legend.get_frame().set_boxstyle('round,pad=0.8')
    legend.get_frame().set_linewidth(2)
    
    # Ultra-clean grid
    ax.grid(True, alpha=0.1, linewidth=1)
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "bulk_tree_3d_research.png"), 
                bbox_inches='tight', facecolor='white', dpi=300,
                edgecolor='none', pad_inches=0.5)
    plt.show()
    plt.close()


def plot_einstein_correlation(times: np.ndarray, correlations: list[float], outdir: str):
    """
    Ultra-modern Einstein correlation visualization.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), facecolor='white')
    ax.set_facecolor('#FAFAFA')
    
    # Ultra-modern line plot
    ax.plot(times, correlations, 'o-', markersize=12, linewidth=4, 
            color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='white', 
            markeredgewidth=3, label='Einstein Correlation', alpha=0.9)
    
    # Ultra-subtle confidence band
    correlations_array = np.array(correlations)
    noise_level = 0.03
    upper_bound = correlations_array + noise_level
    lower_bound = correlations_array - noise_level
    ax.fill_between(times, lower_bound, upper_bound, alpha=0.15, color='#2E86AB')
    
    # Ultra-modern styling
    ax.set_title('Holographic Einstein Equation\nCurvature-Energy Correspondence', 
                fontsize=26, fontweight='200', pad=40, color='#2C2C2C')
    ax.set_xlabel('Evolution Time (ħ/J)', fontsize=20, fontweight='300', labelpad=20, color='#444444')
    ax.set_ylabel('Correlation Coefficient r', fontsize=20, fontweight='300', labelpad=20, color='#444444')
    
    # Ultra-modern reference lines
    ax.axhline(y=0, color='#CCCCCC', linestyle='-', alpha=0.6, linewidth=2)
    ax.axhline(y=0.7, color='#4CAF50', linestyle='--', alpha=0.7, linewidth=2, label='Strong Correlation')
    ax.axhline(y=-0.7, color='#4CAF50', linestyle='--', alpha=0.7, linewidth=2)
    
    # Ultra-modern equation annotation
    equation_text = r'$R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R = 8\pi G T_{\mu\nu}$'
    ax.text(0.02, 0.98, equation_text, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', fontweight='300', color='#444444',
            bbox=dict(boxstyle='round,pad=1.0', facecolor='#F8F8F8', 
                     edgecolor='#CCCCCC', alpha=0.9, linewidth=2))
    
    # Ultra-clean grid and styling
    ax.grid(True, linestyle='-', alpha=0.1, linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Ultra-modern legend
    legend = ax.legend(loc='lower right', fontsize=16, framealpha=1.0, shadow=True,
                      facecolor='white', edgecolor='#CCCCCC')
    legend.get_frame().set_boxstyle('round,pad=0.8')
    legend.get_frame().set_linewidth(2)
    
    ax.set_ylim(-1.1, 1.1)
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=8)
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "einstein_correlation_research.png"), 
                bbox_inches='tight', facecolor='white', dpi=300,
                edgecolor='none', pad_inches=0.3)
    plt.show()
    plt.close()


def plot_entropy_over_time(times: np.ndarray, ent_dict: dict[tuple[int, ...], np.ndarray], outdir: str):
    """
    Ultra-modern entropy evolution visualization.
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 12), facecolor='white')
    ax.set_facecolor('#FAFAFA')
    
    # Ultra-modern color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50']
    
    for idx, ((region, series), color) in enumerate(zip(ent_dict.items(), colors)):
        shifted = series - np.min(series)
        
        # Ultra-modern line styling
        label = f'Region [{region[0]}:{region[-1]}] ({len(region)} qubits)'
        ax.plot(times, shifted, label=label, linewidth=4, color=color, 
                marker='o', markersize=8, markeredgecolor='white', markeredgewidth=2, alpha=0.9)
        
        # Ultra-subtle fill
        ax.fill_between(times, 0, shifted, alpha=0.05, color=color)
    
    # Ultra-modern styling
    ax.set_title('Quantum Entanglement Dynamics\nEvolution of Regional von Neumann Entropy', 
                fontsize=26, fontweight='200', pad=40, color='#2C2C2C')
    ax.set_xlabel('Evolution Time (ħ/J)', fontsize=20, fontweight='300', labelpad=20, color='#444444')
    ax.set_ylabel('Entanglement Entropy S (normalized)', fontsize=20, fontweight='300', labelpad=20, color='#444444')
    
    # Ultra-modern equation annotation
    equation_text = r'$S(\rho_A) = -\text{Tr}(\rho_A \log \rho_A)$'
    ax.text(0.02, 0.98, equation_text, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', fontweight='300', color='#444444',
            bbox=dict(boxstyle='round,pad=1.0', facecolor='#F8F8F8', 
                     edgecolor='#CCCCCC', alpha=0.9, linewidth=2))
    
    # Ultra-clean styling
    ax.grid(True, linestyle='-', alpha=0.1, linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Ultra-modern legend
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16,
                      framealpha=1.0, shadow=True, facecolor='white', 
                      edgecolor='#CCCCCC')
    legend.get_frame().set_boxstyle('round,pad=0.8')
    legend.get_frame().set_linewidth(2)
    
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=8)
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "entropy_evolution_research.png"), 
                bbox_inches='tight', facecolor='white', dpi=300,
                edgecolor='none', pad_inches=0.3)
    plt.show()
    plt.close()


def plot_weight_comparison(true_w: np.ndarray, learned_w: np.ndarray, outdir: str):
    """
    Ultra-modern weight comparison visualization.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor='white')
    ax.set_facecolor('#FAFAFA')
    
    # Ensure arrays are the same size
    min_size = min(len(true_w), len(learned_w))
    true_w = true_w[:min_size]
    learned_w = learned_w[:min_size]
    
    # Ultra-modern scatter plot
    scatter = ax.scatter(true_w, learned_w, c=learned_w, cmap='plasma', 
                        s=150, alpha=0.8, edgecolors='white', linewidths=2)
    
    # Perfect correlation line
    lim = [min(true_w.min(), learned_w.min()), max(true_w.max(), learned_w.max())]
    ax.plot(lim, lim, '--', linewidth=4, color='#FF6B6B', alpha=0.8, label='Perfect Agreement')
    
    # Calculate R² with proper error handling
    try:
        if len(true_w) > 1 and np.std(true_w) > 1e-10 and np.std(learned_w) > 1e-10:
            correlation_matrix = np.corrcoef(true_w, learned_w)
            r_value = correlation_matrix[0, 1]
            if np.isnan(r_value):
                r_squared = 0.0
            else:
                r_squared = r_value**2
        else:
            r_squared = 0.0
    except:
        r_squared = 0.0
    
    # Ultra-modern styling
    ax.set_title('Neural Network Performance\nLearned vs. True Edge Weights', 
                fontsize=26, fontweight='200', pad=40, color='#2C2C2C')
    ax.set_xlabel('True Entanglement Weight', fontsize=20, fontweight='300', labelpad=20, color='#444444')
    ax.set_ylabel('Learned Weight (GNN Output)', fontsize=20, fontweight='300', labelpad=20, color='#444444')
    
    # Ultra-modern R² annotation
    r2_text = f'$R^2 = {r_squared:.3f}$\nModel Accuracy'
    ax.text(0.05, 0.95, r2_text, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', fontweight='300', color='#444444',
            bbox=dict(boxstyle='round,pad=1.0', facecolor='#E8F5E8', 
                     edgecolor='#4CAF50', alpha=0.9, linewidth=2))
    
    # Ultra-modern colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Learned Weight Value', fontsize=18, fontweight='300', labelpad=20)
    cbar.ax.tick_params(labelsize=16, width=2, length=6)
    cbar.outline.set_linewidth(2)
    
    # Ultra-clean styling
    ax.grid(True, linestyle='-', alpha=0.1, linewidth=1)
    ax.set_aspect('equal', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Ultra-modern legend
    legend = ax.legend(fontsize=16, framealpha=1.0, shadow=True,
                      facecolor='white', edgecolor='#CCCCCC')
    legend.get_frame().set_boxstyle('round,pad=0.8')
    legend.get_frame().set_linewidth(2)
    
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=8)
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "weight_comparison_research.png"), 
                bbox_inches='tight', facecolor='white', dpi=300,
                edgecolor='none', pad_inches=0.3)
    plt.show()
    plt.close()


def _compute_leaf_descendants(tree: nx.Graph) -> dict[str, list[str]]:
    """
    For each internal node, return a list of leaf-nodes reachable from it.
    Leaves are nodes of degree 1.
    """
    descendants = {}
    leaves = {n for n in tree.nodes if tree.degree[n] == 1}

    for node in tree.nodes:
        if tree.degree[node] == 1:
            continue
        visited = {node}
        queue = deque([node])
        node_leaves = []

        while queue:
            curr = queue.popleft()
            for nbr in tree.neighbors(curr):
                if nbr in visited:
                    continue
                visited.add(nbr)
                if nbr in leaves:
                    node_leaves.append(nbr)
                else:
                    queue.append(nbr)
        descendants[node] = node_leaves

    return descendants
