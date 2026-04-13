#!/usr/bin/env python3
"""Generate GC-ACT architecture diagram for the paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_rounded_box(ax, xy, width, height, text, color='#4A90D9',
                     text_color='white', fontsize=9, alpha=0.9, bold=False):
    """Draw a rounded rectangle with centered text."""
    x, y = xy
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#333333',
                          linewidth=1.2, alpha=alpha, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=3)
    return box


def draw_arrow(ax, start, end, color='#555555', style='->', lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0'))


def draw_dashed_arrow(ax, start, end, color='#888888', lw=1.2):
    """Draw a dashed arrow."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, linestyle='dashed'))


def generate_gcact_architecture():
    """Generate the main GC-ACT architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-1, 10.5)
    ax.axis('off')

    # Title
    ax.text(8, 10.1, 'GC-ACT: Gesture-Conditioned Action Chunking Transformer',
            ha='center', va='center', fontsize=15, fontweight='bold', color='#222222')

    # INPUT LAYER
    # Camera images
    cam_color = '#5BA55B'
    draw_rounded_box(ax, (0.5, 8.2), 2.2, 0.8, 'Left Endoscope\n(360x480)', cam_color, fontsize=8)
    draw_rounded_box(ax, (3.0, 8.2), 2.2, 0.8, 'PSM1 Wrist\n(360x480)', cam_color, fontsize=8)
    draw_rounded_box(ax, (5.5, 8.2), 2.2, 0.8, 'PSM2 Wrist\n(360x480)', cam_color, fontsize=8)

    # Gesture input (NEW - highlighted)
    draw_rounded_box(ax, (9.0, 8.2), 2.8, 0.8, 'Gesture Classifier\n(ResNet-18, 93.3% acc)',
                     '#E8544E', fontsize=8, bold=True)

    # Gesture one-hot
    draw_rounded_box(ax, (9.3, 7.0), 2.2, 0.6, 'One-Hot (10D)',
                     '#F4A460', text_color='#333333', fontsize=8)

    # BACKBONE
    backbone_color = '#4A90D9'
    draw_rounded_box(ax, (1.5, 6.8), 4.5, 0.8, 'EfficientNet-B3 Backbone (shared weights)',
                     backbone_color, fontsize=9)

    # Spatial features
    draw_rounded_box(ax, (1.5, 5.6), 4.5, 0.7, 'Spatial Features + Positional Encoding',
                     '#6BAED6', fontsize=9)

    # CVAE ENCODER (training only)
    draw_rounded_box(ax, (12.5, 7.0), 2.8, 0.6, 'CVAE Encoder\n(training only)',
                     '#BBBBBB', text_color='#444444', fontsize=8)
    draw_rounded_box(ax, (13.0, 6.0), 1.8, 0.6, 'Latent z\n(256D)',
                     '#9B59B6', fontsize=8)

    # TRANSFORMER ENCODER
    # Token assembly
    token_color = '#F39C12'
    draw_rounded_box(ax, (0.3, 4.2), 2.0, 0.7, 'Latent z\nToken',
                     '#9B59B6', fontsize=8)
    draw_rounded_box(ax, (2.6, 4.2), 2.0, 0.7, 'Proprio\nToken',
                     '#3498DB', fontsize=8)
    draw_rounded_box(ax, (4.9, 4.2), 2.0, 0.7, 'Gesture\nToken',
                     '#E8544E', fontsize=8, bold=True)
    draw_rounded_box(ax, (7.2, 4.2), 4.0, 0.7, 'Image Feature Tokens\n(flattened spatial)',
                     '#6BAED6', fontsize=8)

    # Positional embeddings label
    ax.text(3.45, 3.7, 'pos_embed[0]', ha='center', fontsize=6.5, color='#666666', style='italic')
    ax.text(1.3, 3.7, 'pos_embed[1]', ha='center', fontsize=6.5, color='#666666', style='italic')
    ax.text(5.9, 3.7, 'pos_embed[2]', ha='center', fontsize=6.5, color='#666666', style='italic')

    # NEW label
    ax.text(5.9, 5.1, 'NEW', ha='center', fontsize=8, color='#E8544E',
            fontweight='bold', style='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF0F0', edgecolor='#E8544E', linewidth=1))

    # Transformer encoder
    enc_color = '#2C3E50'
    draw_rounded_box(ax, (0.3, 2.5), 10.9, 0.9, 'Transformer Encoder (4 layers, 8 heads, dim=512)',
                     enc_color, fontsize=10, bold=True)

    # TRANSFORMER DECODER
    draw_rounded_box(ax, (0.3, 1.0), 10.9, 0.9, 'Transformer Decoder (7 layers, cross-attention to encoder)',
                     '#1A242F', fontsize=10, bold=True)

    # Query embeddings
    draw_rounded_box(ax, (12.5, 1.1), 2.8, 0.7, '60 Learned\nQuery Embeddings',
                     token_color, text_color='#333333', fontsize=8)

    # OUTPUT
    output_color = '#27AE60'
    draw_rounded_box(ax, (2.5, -0.5), 6.5, 0.8, 'Action Chunk: 60 steps x 20D\n[pos(3) + rot6d(6) + jaw(1)] x 2 arms',
                     output_color, fontsize=9, bold=True)

    # ARROWS
    # Cameras to backbone
    for x_start in [1.6, 4.1, 6.6]:
        draw_arrow(ax, (x_start, 8.2), (x_start, 7.6))

    # Backbone to spatial features
    draw_arrow(ax, (3.75, 6.8), (3.75, 6.3))

    # Spatial features to image tokens
    draw_arrow(ax, (3.75, 5.6), (9.2, 4.9))

    # Gesture classifier to one-hot
    draw_arrow(ax, (10.4, 8.2), (10.4, 7.6), color='#E8544E')

    # One-hot to gesture token (gesture projection)
    draw_arrow(ax, (10.4, 7.0), (5.9, 4.9), color='#E8544E')
    ax.text(8.0, 6.1, '$W_g$', ha='center', fontsize=9, color='#E8544E',
            fontweight='bold', style='italic')

    # CVAE to latent z
    draw_dashed_arrow(ax, (13.9, 7.0), (13.9, 6.6))

    # Latent z to token
    draw_arrow(ax, (13.9, 6.0), (1.3, 4.9), color='#9B59B6')

    # Tokens to encoder
    draw_arrow(ax, (5.75, 4.2), (5.75, 3.4))

    # Encoder to decoder
    draw_arrow(ax, (5.75, 2.5), (5.75, 1.9))

    # Query embeddings to decoder
    draw_arrow(ax, (12.5, 1.45), (11.2, 1.45))

    # Decoder to output
    draw_arrow(ax, (5.75, 1.0), (5.75, 0.3))

    # LEGEND
    legend_y = -0.3
    ax.add_patch(FancyBboxPatch((12.0, 0.5), 0.3, 0.3, boxstyle="round,pad=0.05",
                                 facecolor='#E8544E', edgecolor='#333333', linewidth=0.8))
    ax.text(12.5, 0.65, '= GC-ACT additions (new)', fontsize=8, va='center', color='#444444')

    ax.add_patch(FancyBboxPatch((12.0, 0.0), 0.3, 0.3, boxstyle="round,pad=0.05",
                                 facecolor='#4A90D9', edgecolor='#333333', linewidth=0.8))
    ax.text(12.5, 0.15, '= Existing ACT components', fontsize=8, va='center', color='#444444')

    plt.tight_layout()
    return fig


def generate_pipeline_diagram():
    """Generate the 3-stage pipeline + gesture conditioning overview."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')

    ax.text(8, 4.2, 'Inference Pipeline: 3-Stage Suturing with Gesture Conditioning',
            ha='center', va='center', fontsize=14, fontweight='bold', color='#222222')

    # Stage boxes
    draw_rounded_box(ax, (0.5, 2.2), 3.5, 1.2, 'Stage 1: Needle Pickup\n\nACT v2\n(no gesture)',
                     '#3498DB', fontsize=9, bold=True)
    draw_rounded_box(ax, (5.5, 2.2), 3.5, 1.2, 'Stage 2: Needle Throw\n\nGC-ACT\n(gesture-conditioned)',
                     '#E8544E', fontsize=9, bold=True)
    draw_rounded_box(ax, (10.5, 2.2), 3.5, 1.2, 'Stage 3: Knot Tying\n\nGC-ACT\n(gesture-conditioned)',
                     '#E8544E', fontsize=9, bold=True)

    # Arrows between stages
    draw_arrow(ax, (4.0, 2.8), (5.5, 2.8), lw=2.5, color='#333333')
    draw_arrow(ax, (9.0, 2.8), (10.5, 2.8), lw=2.5, color='#333333')

    # Gesture classifier feeding into stages 2 and 3
    draw_rounded_box(ax, (6.0, 0.3), 4.5, 0.8, 'Gesture Classifier (ResNet-18)\nPredicts current phase from endoscope image',
                     '#F39C12', text_color='#333333', fontsize=8)

    draw_arrow(ax, (7.5, 1.1), (7.25, 2.2), color='#E8544E', lw=1.8)
    draw_arrow(ax, (9.5, 1.1), (12.25, 2.2), color='#E8544E', lw=1.8)

    # Gesture sequences
    ax.text(7.25, 1.65, 'G2→G3→G16→G7', ha='center', fontsize=7,
            color='#E8544E', style='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFF5F5', edgecolor='#E8544E', linewidth=0.5))
    ax.text(12.25, 1.65, 'G13→G14→G15', ha='center', fontsize=7,
            color='#E8544E', style='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFF5F5', edgecolor='#E8544E', linewidth=0.5))

    # Camera input
    draw_rounded_box(ax, (0.5, 0.3), 3.5, 0.8, '3 Cameras: Left Endoscope,\nPSM1 Wrist, PSM2 Wrist',
                     '#5BA55B', fontsize=8)
    draw_arrow(ax, (2.25, 1.1), (2.25, 2.2), color='#5BA55B', lw=1.8)
    draw_arrow(ax, (4.0, 0.7), (6.0, 0.7), color='#5BA55B', lw=1.2)

    plt.tight_layout()
    return fig


def generate_token_detail():
    """Generate detailed view of the transformer encoder input tokens."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')

    ax.text(7, 3.2, 'Transformer Encoder Input: Token Assembly',
            ha='center', va='center', fontsize=13, fontweight='bold', color='#222222')

    # ACT (baseline) tokens
    ax.text(4.5, 2.5, 'ACT (baseline):', ha='center', fontsize=10, color='#555555')
    draw_rounded_box(ax, (0.5, 1.7), 2.0, 0.7, 'Latent z\n(256D)', '#9B59B6', fontsize=9)
    draw_rounded_box(ax, (3.0, 1.7), 2.2, 0.7, 'Proprio\n(zeros, 512D)', '#3498DB', fontsize=9)
    draw_rounded_box(ax, (5.7, 1.7), 3.0, 0.7, 'Image Features\n(N tokens, 512D)', '#6BAED6', fontsize=9)

    # pos_embed labels
    ax.text(1.5, 1.5, 'pos[0]', ha='center', fontsize=7, color='#888888', style='italic')
    ax.text(4.1, 1.5, 'pos[1]', ha='center', fontsize=7, color='#888888', style='italic')

    # GC-ACT tokens
    ax.text(4.5, 0.9, 'GC-ACT (ours):', ha='center', fontsize=10, color='#E8544E', fontweight='bold')
    draw_rounded_box(ax, (0.5, 0.1), 2.0, 0.7, 'Latent z\n(256D)', '#9B59B6', fontsize=9)
    draw_rounded_box(ax, (3.0, 0.1), 2.2, 0.7, 'Proprio\n(zeros, 512D)', '#3498DB', fontsize=9)
    draw_rounded_box(ax, (5.7, 0.1), 2.2, 0.7, 'Gesture\n(10D → 512D)', '#E8544E', fontsize=9, bold=True)
    draw_rounded_box(ax, (8.4, 0.1), 3.0, 0.7, 'Image Features\n(N tokens, 512D)', '#6BAED6', fontsize=9)

    ax.text(1.5, -0.1, 'pos[0]', ha='center', fontsize=7, color='#888888', style='italic')
    ax.text(4.1, -0.1, 'pos[1]', ha='center', fontsize=7, color='#888888', style='italic')
    ax.text(6.8, -0.1, 'pos[2] (new)', ha='center', fontsize=7, color='#E8544E', style='italic')

    # Bracket showing the addition
    ax.annotate('', xy=(5.6, 0.45), xytext=(8.0, 0.45),
                arrowprops=dict(arrowstyle='<->', color='#E8544E', lw=1.5))
    ax.text(10.5, 0.45, 'Added token:\ngesture one-hot\nprojected to 512D',
            ha='center', va='center', fontsize=8, color='#E8544E',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F0', edgecolor='#E8544E'))

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import os
    output_dir = os.path.expanduser('~/paper_materials')
    os.makedirs(output_dir, exist_ok=True)

    print("Generating architecture diagrams...")

    fig1 = generate_gcact_architecture()
    path1 = os.path.join(output_dir, 'gcact_architecture.png')
    fig1.savefig(path1, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Main architecture: {path1}")

    fig2 = generate_pipeline_diagram()
    path2 = os.path.join(output_dir, 'pipeline_overview.png')
    fig2.savefig(path2, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Pipeline overview: {path2}")

    fig3 = generate_token_detail()
    path3 = os.path.join(output_dir, 'token_assembly.png')
    fig3.savefig(path3, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Token assembly:   {path3}")

    # Also save as PDF for LaTeX
    fig1.savefig(os.path.join(output_dir, 'gcact_architecture.pdf'),
                 bbox_inches='tight', facecolor='white')
    fig2.savefig(os.path.join(output_dir, 'pipeline_overview.pdf'),
                 bbox_inches='tight', facecolor='white')
    fig3.savefig(os.path.join(output_dir, 'token_assembly.pdf'),
                 bbox_inches='tight', facecolor='white')
    print("  PDF versions saved too.")

    plt.close('all')
    print("Done.")
