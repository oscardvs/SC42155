import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

# ---------- Setup ----------
OUTDIR = Path(os.getenv("OUTDIR", "./figs")).expanduser().resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

# Enable constrained layout to avoid tight_layout warnings
matplotlib.rcParams['figure.constrained_layout.use'] = True

# ---------- Helper ----------
def save_fig(fig, filename):
    """Save figure to OUTDIR with proper bounding box."""
    fig.savefig(OUTDIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---------- Drawing functions ----------

def draw_ta_diagram(filename, is_residential=False):
    """Draw 2-state timed automaton for pump ON/OFF transitions."""
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.set_aspect('equal')
    ax.axis('off')

    # Node positions
    off_pos = (0.0, 0.0)
    on_pos  = (6.0, 0.0)
    r = 1.0

    # Nodes
    ax.add_patch(Circle(off_pos, r, fill=False, linewidth=2))
    ax.add_patch(Circle(on_pos,  r, fill=False, linewidth=2))
    ax.text(off_pos[0], off_pos[1], "OFF", ha='center', va='center', fontsize=16)
    ax.text(on_pos[0],  on_pos[1],  "ON",  ha='center', va='center', fontsize=16)

    # Arrows
    arrow1 = FancyArrowPatch(posA=(off_pos[0]+r, off_pos[1]+0.1),
                             posB=(on_pos[0]-r, on_pos[1]+0.1),
                             arrowstyle='->', mutation_scale=15,
                             connectionstyle='arc3,rad=0.25', linewidth=2)
    arrow2 = FancyArrowPatch(posA=(on_pos[0]-r, on_pos[1]-0.1),
                             posB=(off_pos[0]+r, off_pos[1]-0.1),
                             arrowstyle='->', mutation_scale=15,
                             connectionstyle='arc3,rad=-0.25', linewidth=2)
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)

    # Labels
    if is_residential:
        label_off_on = "p ∉ {P2,P3,P6} AND (n_j = 0)\nAND (T_j ≤ T_r)"
        label_on_off = "p ∈ {P2,P3,P6} OR (n_j = 1)\nOR (T_j > T_r)"
        footer = "Residential: forced-OFF P2,P3,P6; neighbor override n_j=1 (attached office ON)"
    else:
        label_off_on = "p ∉ {P5,P6} AND (T_i ≤ T_r)"
        label_on_off = "p ∈ {P5,P6} OR (T_i > T_r)"
        footer = "Office: forced-OFF P5,P6 (22–06). Decisions checked at 4 h ticks."

    ax.text(3.0, 1.6, label_off_on, ha='center', va='center', fontsize=12)
    ax.text(3.0,-1.6, label_on_off, ha='center', va='center', fontsize=12)
    ax.text(3.0,-3.0, footer, ha='center', va='center', fontsize=11)

    save_fig(fig, filename)


def draw_forced_windows(filename):
    """Stripe diagram for forced-off windows per building type."""
    fig, ax = plt.subplots(figsize=(12.5, 4.2), constrained_layout=True)

    periods = ["P1\n06–10","P2\n10–14","P3\n14–18","P4\n18–22","P5\n22–02","P6\n02–06"]
    x = np.arange(6)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 1.5)

    ax.text(-0.35, 1.0, "Office", va='center', fontsize=12)
    ax.text(-0.35, 0.0, "Residential", va='center', fontsize=12)

    for i in range(6):
        ax.add_patch(Rectangle((i-0.45, 0.7), 0.9, 0.5, fill=False))
        ax.add_patch(Rectangle((i-0.45,-0.25), 0.9, 0.5, fill=False))

    office_forced = {4,5}   # P5,P6
    res_forced    = {1,2,5} # P2,P3,P6

    for i in office_forced:
        ax.add_patch(Rectangle((i-0.45, 0.7), 0.9, 0.5, alpha=0.3))
    for i in res_forced:
        ax.add_patch(Rectangle((i-0.45,-0.25), 0.9, 0.5, alpha=0.3))

    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.set_yticks([])
    ax.set_title("Forced-OFF windows per building type (shaded)")

    save_fig(fig, filename)


def compute_pump_commands(T1, T2, Tref):
    """Compute pump commands (office y1, residential y2) for 6 periods."""
    y1, y2 = [], []
    for k in range(6):
        # Office rules
        if k in [4,5]:
            y1_k = 0
        else:
            y1_k = 1 if T1[k] <= Tref else 0
        y1.append(y1_k)

        # Residential rules
        if k in [1,2,5]:
            y2_k = 0
        else:
            n2 = y1_k  # neighbor effect
            if n2 == 1:
                y2_k = 0
            else:
                y2_k = 1 if T2[k] <= Tref else 0
        y2.append(y2_k)
    return np.array(y1), np.array(y2)


def draw_temperature_plot(filename, T1, T2, Tref):
    """Temperature evolution for 24 h with forced-off shading."""
    t = np.array([6,10,14,18,22,26])  # sample times
    t_rel = t - 6

    fig, ax = plt.subplots(figsize=(12.5, 4.2), constrained_layout=True)
    ax.plot(t_rel, T1, marker='o', label="Office 1")
    ax.plot(t_rel, T2, marker='o', label="Residential 2")
    ax.axhline(Tref, linestyle='--', linewidth=1, label="T_r")

    # Shade forced-off windows
    ax.axvspan(16, 24, alpha=0.08)
    ax.axvspan(4, 12, alpha=0.08)
    ax.axvspan(20, 24, alpha=0.08)

    ax.set_xlim(0, 24)
    ax.set_xlabel("Time since 06:00 [h]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("24 h temperature profiles (samples at 4 h grid)")
    ax.legend(loc="best")
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.set_xticks([0,4,8,12,16,20,24])
    ax.set_xticklabels(["06","10","14","18","22","02","06"])

    save_fig(fig, filename)


def draw_pump_states(filename, y1, y2):
    """Pump ON/OFF step plots over 24 h."""
    t_edges = np.array([0,4,8,12,16,20,24])

    fig, ax = plt.subplots(figsize=(12.5, 4.2), constrained_layout=True)
    ax.step(t_edges, np.r_[y1, y1[-1]], where='post', label="Office 1")
    ax.step(t_edges, np.r_[y2, y2[-1]], where='post', label="Residential 2")

    # Shaded forced-off windows
    ax.axvspan(16, 24, alpha=0.08)
    ax.axvspan(4, 12,  alpha=0.08)
    ax.axvspan(20, 24, alpha=0.08)

    ax.set_xlim(0, 24)
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0,1])
    ax.set_yticklabels(["OFF","ON"])
    ax.set_xlabel("Time since 06:00 [h]")
    ax.set_title("Pump commands (period-wise step functions)")
    ax.legend(loc="best")
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.set_xticks([0,4,8,12,16,20,24])
    ax.set_xticklabels(["06","10","14","18","22","02","06"])

    save_fig(fig, filename)


# ---------- Generate Q3.2 diagrams ----------
draw_ta_diagram("q3_2_office_ta.png", is_residential=False)
draw_ta_diagram("q3_2_res_ta.png", is_residential=True)
draw_forced_windows("q3_2_forced_windows.png")

# ---------- Generate Q3.4 plots ----------
Tref = 21.0
T1 = np.array([20.0, 22.0, 21.5, 20.0, 20.0, 20.0])
T2 = np.array([20.5, 20.0, 22.0, 20.0, 20.0, 20.0])
y1, y2 = compute_pump_commands(T1, T2, Tref)

draw_temperature_plot("q3_4_temperature.png", T1, T2, Tref)
draw_pump_states("q3_4_pump_states.png", y1, y2)

# ---------- Output summary ----------
print("Figures written to:", OUTDIR)
for p in sorted(OUTDIR.glob("*.png")):
    print(" -", p.name)

