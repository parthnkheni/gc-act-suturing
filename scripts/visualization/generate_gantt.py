#!/usr/bin/env python3
"""Generate milestone-oriented Gantt chart color-coded by Specific Aims.
v3: Added meeting-date annotations and more prominent date axis per professor feedback."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

# Color scheme by Specific Aim
AIM_COLORS = {
    'SA1': '#1565C0',  # Blue - Imitation Learning Algorithm
    'SA2': '#2E7D32',  # Green - Simulation Environment
    'SA3': '#E65100',  # Orange - Physical dVRK Validation
    'SA4': '#7B1FA2',  # Purple - Continuous Multi-Stitch
    'ALL': '#546E7A',  # Gray - Cross-cutting / Planning
}

AIM_COLORS_LIGHT = {
    'SA1': '#BBDEFB',
    'SA2': '#C8E6C9',
    'SA3': '#FFE0B2',
    'SA4': '#E1BEE7',
    'ALL': '#CFD8DC',
}

AIM_BAR_COLORS = {
    'SA1': '#42A5F5',
    'SA2': '#66BB6A',
    'SA3': '#FFA726',
    'SA4': '#AB47BC',
    'ALL': '#90A4AE',
}

# Meeting dates with short annotations (from meeting minutes)
meeting_annotations = [
    (datetime(2025,  9, 29), "Kickoff meeting;\nscope & Gantt assigned"),
    (datetime(2025, 10,  3), "WPI meeting w/\nJack & Kehan"),
    (datetime(2025, 10,  6), "3-tier scope\ndefined"),
    (datetime(2025, 10, 22), "JIGSAWS demo;\nsimulator on SCC"),
    (datetime(2025, 11,  3), "LSTM integration;\nORBIT explored"),
    (datetime(2025, 11, 10), "JHU meeting w/\nJack & Kazanzides"),
    (datetime(2025, 11, 17), "Data alignment;\nSTITCH 2.0 review"),
    (datetime(2025, 11, 24), "SAM segmentation;\nORBIT progress"),
    (datetime(2025, 12,  1), "SutureBot paper\ndiscovered (1,890 eps)"),
    (datetime(2025, 12,  8), "SutureBot deep-dive;\npivot to dataset"),
    (datetime(2025, 12, 22), "Labeling roadmap;\nKrieger meeting planned"),
    (datetime(2026,  1, 12), "800 labels target;\nZuskov met Krieger"),
    (datetime(2026,  2,  9), "4 models trained;\nIsaac Sim GUI ready"),
    (datetime(2026,  2, 23), "ACT vs LSTM decided;\npivot to ACT"),
    (datetime(2026,  2, 28), "ORBIT physics gap\nconfirmed; eval plan set"),
]

# Define items grouped by aim (order matters for display)
# (name, aim, start, end, is_milestone)
groups = [
    # Cross-cutting / Planning
    ("Project Formulation & Scoping",                    'ALL', "2025-09-01", "2025-09-25", False),
    ("JHU Collaboration Initiated",                      'ALL', "2025-10-20", "2025-11-05", False),
    ("Sem 1 Final Presentation",                         'ALL', "2025-12-10", "2025-12-10", True),
    ("Paper Writing & Results",                          'ALL', "2026-03-15", "2026-04-20", False),
    ("Final Demonstration & Defense",                    'ALL', "2026-04-25", "2026-04-25", True),

    # SA1: Imitation Learning Algorithm
    ("Literature Review & Algorithm Selection",          'SA1', "2025-09-15", "2025-10-20", False),
    ("JIGSAWS Dataset Acquisition",                      'SA1', "2025-10-01", "2025-10-15", False),
    ("LSTM Baseline Development",                        'SA1', "2025-10-15", "2025-11-20", False),
    ("SutureBot Dataset (1,890 eps, 10 tissues)",        'SA1', "2025-11-01", "2025-12-01", False),
    ("ACT Architecture Selection",                       'SA1', "2025-11-15", "2025-12-05", False),
    ("Baseline IL model trained",                        'SA1', "2025-12-05", "2025-12-05", True),
    ("Gesture Label Annotation (G1\u2013G16)",           'SA1', "2025-12-15", "2026-01-20", False),
    ("ACT v1 Training (ResNet18)",                       'SA1', "2026-01-05", "2026-01-20", False),
    ("ACT v2 Training (EfficientNet-B3, 10 tissues)",    'SA1', "2026-01-20", "2026-02-10", False),
    ("Gesture Classifier (93.3% acc)",                   'SA1', "2026-02-01", "2026-02-15", False),
    ("GC-ACT Training (gesture-conditioned)",            'SA1', "2026-02-10", "2026-02-25", False),
    ("GC-ACT achieves 0.707 mm (< 1 mm)",               'SA1', "2026-02-25", "2026-02-25", True),
    ("Augmented Fine-tuning",                            'SA1', "2026-03-01", "2026-03-08", False),
    ("50-Ep Offline Eval (7 configs, 650 evals)",        'SA1', "2026-03-01", "2026-03-08", False),
    ("GEARS Quality Monitor",                            'SA1', "2026-03-03", "2026-03-08", False),
    ("All offline evaluation complete",                  'SA1', "2026-03-08", "2026-03-08", True),

    # SA2: Simulation Environment
    ("Jetstream2 Cloud Setup (A100 + L40S)",             'SA2', "2025-09-20", "2025-10-10", False),
    ("dVRK-ROS-AMBF Exploration",                        'SA2', "2025-10-05", "2025-10-30", False),
    ("ORBIT-Surgical / Isaac Sim 4.1 Setup",             'SA2', "2025-11-10", "2025-12-15", False),
    ("ACT Deployment in ORBIT-Surgical",                 'SA2', "2026-02-15", "2026-03-01", False),
    ("Coordinate Calibration & IK Verify",               'SA2', "2026-02-20", "2026-03-01", False),
    ("Closed-loop sim control verified",                 'SA2', "2026-03-01", "2026-03-01", True),

    # SA3: Physical dVRK Validation
    ("Deployment Package Preparation",                   'SA3', "2026-03-05", "2026-03-15", False),
    ("ROS Integration & Camera Verification",            'SA3', "2026-03-15", "2026-03-25", False),
    ("Single-Subtask Testing (NP, NT, KT)",              'SA3', "2026-03-25", "2026-04-05", False),
    ("Parameter Tuning at JHU",                          'SA3', "2026-04-01", "2026-04-08", False),
    ("Single suture < 2 mm on 5 trials",                 'SA3', "2026-04-08", "2026-04-08", True),

    # SA4: Continuous Multi-Stitch Suturing
    ("Multi-Stitch Pipeline Development",                'SA4', "2026-03-08", "2026-03-20", False),
    ("Wound Detection Module",                           'SA4', "2026-03-10", "2026-03-25", False),
    ("Multi-Stitch Integration & Testing",               'SA4', "2026-04-05", "2026-04-15", False),
    ("Full Autonomous Suturing Demo",                    'SA4', "2026-04-12", "2026-04-20", False),
    ("5 consecutive sutures + closing knot",             'SA4', "2026-04-20", "2026-04-20", True),
]

# Parse dates
parsed = []
for name, aim, s, e, is_ms in groups:
    sd = datetime.strptime(s, "%Y-%m-%d")
    ed = datetime.strptime(e, "%Y-%m-%d")
    parsed.append((name, aim, sd, ed, is_ms))

n = len(parsed)

# Plot
fig, ax = plt.subplots(figsize=(26, 18))
fig.patch.set_facecolor('white')

x_min = datetime(2025, 8, 20)
x_max = datetime(2026, 5, 10)

# Reserve space at top for meeting annotation rows
ANNOT_ROWS = 5.5  # extra y-space above the chart for annotations
y_top = n + ANNOT_ROWS + 2.5

# Compute y ranges per aim group for background bands
aim_y_ranges = {}
for i, (name, aim, sd, ed, is_ms) in enumerate(parsed):
    y = n - i
    if aim not in aim_y_ranges:
        aim_y_ranges[aim] = [y, y]
    aim_y_ranges[aim][0] = min(aim_y_ranges[aim][0], y)
    aim_y_ranges[aim][1] = max(aim_y_ranges[aim][1], y)

# Draw background bands first
for aim, (y_lo, y_hi) in aim_y_ranges.items():
    rect = plt.Rectangle(
        (mdates.date2num(x_min), y_lo - 0.45),
        mdates.date2num(x_max) - mdates.date2num(x_min),
        (y_hi - y_lo) + 0.9,
        facecolor=AIM_COLORS_LIGHT[aim], alpha=0.3,
        edgecolor='none', zorder=0
    )
    ax.add_patch(rect)
    aim_short = {'SA1': 'SA 1', 'SA2': 'SA 2', 'SA3': 'SA 3', 'SA4': 'SA 4', 'ALL': ''}
    if aim != 'ALL':
        ax.text(mdates.date2num(x_min) + 2, (y_lo + y_hi) / 2,
                aim_short[aim], fontsize=10, fontweight='bold',
                color=AIM_COLORS[aim], alpha=0.6, va='center', ha='left',
                rotation=90, zorder=1)

# Draw items
y_labels = []
y_ticks = []

for i, (name, aim, sd, ed, is_ms) in enumerate(parsed):
    y = n - i
    color = AIM_BAR_COLORS[aim]
    dark_color = AIM_COLORS[aim]

    if is_ms:
        ax.plot(sd, y, marker='D', markersize=13, color=dark_color,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax.annotate(name,
                    xy=(sd, y), xytext=(12, 0),
                    textcoords='offset points', fontsize=8.5,
                    fontweight='bold', color=dark_color, va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=dark_color, alpha=0.85, linewidth=0.5))
        y_labels.append("")
    else:
        duration = max((ed - sd).days, 1)
        ax.barh(y, duration, left=sd, height=0.55,
                color=color, alpha=0.9, edgecolor='white', linewidth=0.8, zorder=3)
        y_labels.append(name)

    y_ticks.append(y)

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=8.5)

# Meeting date annotations at top of chart
annot_base_y = n + 1.5  # baseline for annotation markers

# 3-row stagger to avoid overlap
annot_y_offsets = [14, 42, 70]

for idx, (mdate, label) in enumerate(meeting_annotations):
    # Small triangle marker at chart top
    ax.plot(mdate, annot_base_y, marker='v', markersize=7, color='#37474F',
            markeredgecolor='white', markeredgewidth=0.5, zorder=6, clip_on=False)
    # Thin vertical dashed line from marker down through chart
    ax.axvline(x=mdate, color='#78909C', linestyle=':', linewidth=0.6, alpha=0.30, zorder=1,
               ymin=0, ymax=0.88)
    # 3-row stagger
    row = idx % 3
    offset_y = annot_y_offsets[row]
    # Date string + annotation
    date_str = mdate.strftime("%-m/%-d")
    ax.annotate(f"{date_str}\n{label}",
                xy=(mdate, annot_base_y), xytext=(0, offset_y),
                textcoords='offset points', fontsize=6.2,
                color='#263238', ha='center', va='bottom',
                fontweight='normal', linespacing=1.1,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#ECEFF1',
                          edgecolor='#90A4AE', alpha=0.92, linewidth=0.4),
                arrowprops=dict(arrowstyle='-', color='#90A4AE', lw=0.5, alpha=0.5),
                clip_on=False)

# Semester divider
spring_start = datetime(2026, 1, 12)
ax.axvline(x=spring_start, color='#C62828', linestyle='--', linewidth=1.8, alpha=0.6, zorder=2)

# Today marker
today = datetime(2026, 3, 9)
ax.axvline(x=today, color='#D32F2F', linestyle='-', linewidth=2.5, alpha=0.4, zorder=2)
ax.text(today, annot_base_y - 0.1, "TODAY\n(Mar 9)",
        fontsize=9, fontweight='bold', color='#D32F2F', alpha=0.9, ha='center', va='top')

# Horizontal separators between aim groups
prev_aim = None
for i, (name, aim, sd, ed, is_ms) in enumerate(parsed):
    y = n - i
    if prev_aim is not None and aim != prev_aim:
        sep_y = y + 0.5
        ax.axhline(y=sep_y, color='#BDBDBD', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)
    prev_aim = aim

# X-axis formatting: PROMINENT dates on top AND bottom
# Bottom axis  -- bold, larger, biweekly ticks
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
ax.set_xlim(x_min, x_max)
ax.tick_params(axis='x', which='major', labelsize=12, labelcolor='#212121',
               pad=8, length=6, width=1.5)
ax.tick_params(axis='x', which='minor', length=3, width=0.8)

# Make bottom x-axis tick labels bold
for label in ax.get_xticklabels():
    label.set_fontweight('bold')

# Add secondary x-axis on top
ax_top = ax.secondary_xaxis('top')
ax_top.xaxis.set_major_locator(mdates.MonthLocator())
ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax_top.tick_params(axis='x', which='major', labelsize=12, labelcolor='#212121',
                   pad=8, length=6, width=1.5)
for label in ax_top.get_xticklabels():
    label.set_fontweight('bold')

# Grid
ax.grid(axis='x', which='major', linestyle='-', alpha=0.15, zorder=0)
ax.grid(axis='x', which='minor', linestyle=':', alpha=0.08, zorder=0)
ax.set_axisbelow(True)

# Semester & period labels at very top
ax.text(mdates.date2num(spring_start - timedelta(days=55)), y_top - 0.3,
        "FALL SEMESTER 2025", fontsize=13, fontweight='bold', color='#546E7A',
        alpha=0.7, ha='center', clip_on=False)
ax.text(mdates.date2num(spring_start + timedelta(days=55)), y_top - 0.3,
        "SPRING SEMESTER 2026", fontsize=13, fontweight='bold', color='#C62828',
        alpha=0.7, ha='center', clip_on=False)

# Title
ax.set_title("Senior Design: Autonomous Wound Closure System (Team 23)\n"
             "Gantt Chart \u2014 September 2025 through April 2026\n"
             "Color-Coded by Specific Aim  |  Diamonds = Key Milestones  |  "
             "Triangles = Advisor Meetings",
             fontsize=15, fontweight='bold', pad=75, linespacing=1.4)

# Legend
aim_labels_legend = {
    'SA1': 'SA1: Imitation Learning Algorithm',
    'SA2': 'SA2: Simulation Environment',
    'SA3': 'SA3: Physical dVRK Validation',
    'SA4': 'SA4: Continuous Multi-Stitch Suturing',
    'ALL': 'Cross-Cutting / Planning',
}
legend_handles = []
for aim in ['SA1', 'SA2', 'SA3', 'SA4', 'ALL']:
    legend_handles.append(mpatches.Patch(color=AIM_BAR_COLORS[aim],
                                          label=aim_labels_legend[aim]))
legend_handles.append(plt.Line2D([0], [0], marker='D', color='#546E7A', linestyle='None',
                                  markersize=10, label='Key Milestone'))
legend_handles.append(plt.Line2D([0], [0], marker='v', color='#37474F', linestyle='None',
                                  markersize=8, label='Advisor Meeting'))

ax.legend(handles=legend_handles, loc='upper left', fontsize=9.5,
          framealpha=0.95, edgecolor='#ccc', ncol=4,
          bbox_to_anchor=(0.0, 1.0))

# Clean up
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)
ax.set_ylim(0.2, y_top)

plt.tight_layout()
plt.subplots_adjust(left=0.22, right=0.97, top=0.82, bottom=0.05)

# Save
out_png = '/home/exouser/paper_results/figures/gantt_chart_v2.png'
out_pdf = '/home/exouser/paper_results/figures/gantt_chart_v2.pdf'
plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
