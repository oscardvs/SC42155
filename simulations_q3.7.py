import os
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Utility & I/O
# -------------------------
FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

TICKS = 12        # 36 h at 3 h per tick
HOURS_PER_TICK = 3
TICK_AXIS = np.arange(TICKS)    # 0..11

# -------------------------
# Battery GLTS (Q3.5)
# -------------------------
class BatteryGLTS:
    """
    States:
      F        : full
      U1, U2   : 1st and 2nd supply tick (3 h each, max 6 h)
      R3_3..R3_1: 9 h recharge after a 3 h use
      R4_4..R4_1: 12 h recharge after a 6 h use
    Input:  d in {0,1}  (backup requested: L=1 and P_h=low)
    Output: allow in {0,1} (permission to engage/continue battery this tick)
    Mealy output on transitions:
      allow=1 only on F --d=1--> U1   and   U1 --d=1--> U2
    """
    def __init__(self):
        self.state = "F"

    def step(self, d: int) -> Tuple[int, str]:
        s = self.state
        allow = 0

        if s == "F":
            if d == 1:
                allow = 1
                self.state = "U1"
            else:
                self.state = "F"

        elif s == "U1":
            if d == 1:
                allow = 1
                self.state = "U2"
            else:
                self.state = "R3_3"

        elif s == "U2":
            # cap reached -> go to 12 h recharge irrespective of d
            self.state = "R4_4"

        elif s in ("R3_3", "R3_2"):
            self.state = {"R3_3":"R3_2", "R3_2":"R3_1"}[s]
        elif s == "R3_1":
            self.state = "F"

        elif s in ("R4_4", "R4_3", "R4_2"):
            self.state = {"R4_4":"R4_3", "R4_3":"R4_2", "R4_2":"R4_1"}[s]
        elif s == "R4_1":
            self.state = "F"

        else:
            raise ValueError(f"Unknown battery state {s}")

        return allow, self.state

# -------------------------
# Core supervisor (Q3.6)
# -------------------------
@dataclass
class SupervisorOut:
    Sp_closed: int      # 1=closed, 0=open
    Sbg_batt: int       # 1=b (battery), 0=g (generator)
    mode: str           # Off / GenOn / BattOn

class SupervisorCore:
    """
    Inputs per tick: L in {0,1}, Ph in {'high','low'}, allow in {0,1}
    Outputs per tick: S_p in {open,closed}, S_bg in {g,b}
    Modes: Off / GenOn / BattOn (for trace)
    """
    def __init__(self):
        self.mode = "Off"

    def step(self, L: int, Ph: str, allow: int) -> SupervisorOut:
        # Commands (Mealy map)
        if L == 1 and (Ph == "high" or allow == 1):
            Sp_closed = 1
        else:
            Sp_closed = 0

        if Sp_closed == 1 and Ph == "high":
            Sbg_batt = 0  # g
        elif Sp_closed == 1 and Ph == "low" and allow == 1:
            Sbg_batt = 1  # b
        else:
            Sbg_batt = 0  # default g

        # Mode (consistent with outputs)
        if Sp_closed == 0:
            self.mode = "Off"
        else:
            self.mode = "BattOn" if Sbg_batt == 1 else "GenOn"

        return SupervisorOut(Sp_closed=Sp_closed, Sbg_batt=Sbg_batt, mode=self.mode)

# -------------------------
# Scenarios (p_i and P_h)
# -------------------------
def scenario_A() -> Dict[str, np.ndarray]:
    """
    Weekday-like:
      - Office (1,3) peak in daytime ticks (2..5), Residential (2,4) evening ticks (6..9).
      - Hydro low at ticks 2,3 and 7; otherwise high.
    """
    p1 = np.zeros(TICKS, dtype=int)
    p3 = np.zeros(TICKS, dtype=int)
    p2 = np.zeros(TICKS, dtype=int)
    p4 = np.zeros(TICKS, dtype=int)

    p1[2:6] = 1
    p3[2:6] = 1
    p2[6:10] = 1
    p4[6:10] = 1

    Ph = np.array(["high"]*TICKS, dtype=object)
    Ph[[2,3,7]] = "low"

    return {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "Ph": Ph}

def scenario_B() -> Dict[str, np.ndarray]:
    """
    Cold snap:
      - Sustained building demand (L mostly 1).
      - Hydro low from ticks 1..5 (inclusive).
    """
    p1 = np.zeros(TICKS, dtype=int)
    p3 = np.zeros(TICKS, dtype=int)
    p2 = np.zeros(TICKS, dtype=int)
    p4 = np.zeros(TICKS, dtype=int)

    # Make L=1 for many ticks (e.g., 0..9)
    p1[2:8] = 1
    p3[2:8] = 1
    p2[0:6] = 1
    p4[0:6] = 1

    Ph = np.array(["high"]*TICKS, dtype=object)
    Ph[1:6] = "low"

    return {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "Ph": Ph}

# -------------------------
# Simulation driver
# -------------------------
def run_scenario(name: str, seq: Dict[str, np.ndarray]) -> pd.DataFrame:
    bat = BatteryGLTS()
    sup = SupervisorCore()

    p1, p2, p3, p4, Ph = seq["p1"], seq["p2"], seq["p3"], seq["p4"], seq["Ph"]
    L = (p1 + p2 + p3 + p4 >= 1).astype(int)
    b_req = (L == 1) & (Ph == "low")
    b_req = b_req.astype(int)

    rows = []
    for k in range(TICKS):
        d_k = int(b_req[k])
        allow_k, bat_state_next = bat.step(d_k)
        out = sup.step(int(L[k]), str(Ph[k]), int(allow_k))
        rows.append({
            "k": k,
            "p1": int(p1[k]), "p2": int(p2[k]), "p3": int(p3[k]), "p4": int(p4[k]),
            "L": int(L[k]),
            "Ph": Ph[k],
            "b_req": int(d_k),
            "allow": int(allow_k),
            "Sp_closed": int(out.Sp_closed),
            "Sbg_batt": int(out.Sbg_batt),            # 1=b, 0=g
            "mode": out.mode,
            "battery_state": bat_state_next
        })

    df = pd.DataFrame(rows)
    # CSV
    csv_path = os.path.join(FIG_DIR, f"q3_7_{name}_timeline.csv")
    df.to_csv(csv_path, index=False)
    print(f"[{name}] wrote {csv_path}")
    # Figures
    plot_inputs(df, name)
    plot_outputs(df, name)
    plot_battery(df, name)
    return df

# -------------------------
# Plotting
# -------------------------
def _step_axis(ax):
    ax.set_xticks(TICK_AXIS)
    ax.set_xlim(-0.5, TICKS-0.5)
    ax.grid(True, which="both", axis="x", alpha=0.25)

def plot_inputs(df: pd.DataFrame, name: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    # L
    ax.step(df["k"], df["L"], where="post", label="L (any building ON)")
    # Ph_low as 0/1
    Ph_low = (df["Ph"] == "low").astype(int)
    ax.step(df["k"], Ph_low, where="post", label="P_h = low")
    _step_axis(ax)
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlabel("Tick (3 h)")
    ax.set_title(f"Scenario {name}: inputs L and $P_h$")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"q3_7_{name}_inputs.png"), dpi=160)

def plot_outputs(df: pd.DataFrame, name: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.step(df["k"], df["Sp_closed"], where="post", label="$S_p$ (closed=1)")
    ax.step(df["k"], df["Sbg_batt"], where="post", label="$S_{bg}$ (b=1, g=0)")
    _step_axis(ax)
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlabel("Tick (3 h)")
    ax.set_title(f"Scenario {name}: supervisor outputs $S_p$ and $S_{{bg}}$")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"q3_7_{name}_outputs.png"), dpi=160)

def plot_battery(df: pd.DataFrame, name: str):
    # Map categorical states to integers for a stair plot
    states = df["battery_state"].tolist()
    uniq = ["F","U1","U2","R3_3","R3_2","R3_1","R4_4","R4_3","R4_2","R4_1"]
    idx_map = {s:i for i,s in enumerate(uniq)}
    y = [idx_map[s] for s in states]

    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    ax.step(df["k"], y, where="post")
    _step_axis(ax)
    ax.set_yticks(range(len(uniq)))
    ax.set_yticklabels(uniq)
    ax.set_xlabel("Tick (3 h)")
    ax.set_title(f"Scenario {name}: Battery GLTS state")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"q3_7_{name}_battery.png"), dpi=160)

# -------------------------
# Main
# -------------------------
def main():
    dfA = run_scenario("A", scenario_A())
    dfB = run_scenario("B", scenario_B())
    # Console summary
    for name, df in [("A", dfA), ("B", dfB)]:
        print(f"\nScenario {name} summary:")
        print(df[["k","L","Ph","b_req","allow","Sp_closed","Sbg_batt","mode","battery_state"]])

if __name__ == "__main__":
    main()

