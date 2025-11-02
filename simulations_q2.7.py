import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

OUT_DIR = "sim_outputs_q7"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# Physical constants and geometry
# ---------------------------
rho = 1000.0          # kg/m^3
cp = 4180.0           # J/(kg K)

# Pipe
Ldh = 1500.0          # m
Di = 0.15             # m
Acs = np.pi*(Di**2)/4
G_dh_per_m = 0.40     # W/(K m)  (per instructor clarifications: conductance per meter)
k_dh = 0.60           # W/(m K)  (axial conductivity, small but non-zero)
alpha = k_dh/(rho*cp)

# Pump and hydraulics (district pump always ON)
wdh_des = 0.030                 # m^3/s (30 L/s)
vx_const = wdh_des / Acs        # m/s mean axial speed
R_dh_flow = 2.0e6               # Pa*s^2/m^6 (rough power logging only)

# LUMI HX and loop
Vwater_L = 2.0                  # m^3
Cth_L = rho*cp*Vwater_L         # J/K
U_L = 9.3e3                     # W/K (LUMI exchanger)
day = 24*3600.0
def Q_LUMI(t):
    # mean 95 kW with ±25 kW diurnal swing (shifted for morning peak)
    return 95e3 + 25e3*np.sin(2.0*np.pi*(t-3*3600)/day)

# Buildings (two office: 1 & 3; two residential: 2 & 4)
CB_office = 1.2e8               # J/K
CB_res    = 8.0e7               # J/K
CB = np.array([CB_office, CB_res, CB_office, CB_res], dtype=float)

RB_i0 = 2.5e-3                  # K/W (to ambient)  -> 1/R = 400 W/K
R_shared_12 = 1.0e-2            # K/W (between 1-2) -> 1/R = 100 W/K
R_shared_34 = 1.0e-2            # K/W (between 3-4)

# Building-side water nodes and radiators (pumps forced ON)
Vw_bld = 0.20                   # m^3 water per building loop (≈200 L)
Cth_w = rho*cp*Vw_bld           # ≈ 8.36e5 J/K
U_ex  = 1200.0                  # W/K (pipe ↔ building-water HX)
U_rad = np.array([190.0, 178.0, 190.0, 178.0])  # radiator UA per building (W/K)

# Hydropower reservoir (non-depleting by design)
A_res  = 5000.0     # m^2
h_min  = 35.0       # m  (safety minimum)
h_max  = 50.0       # m  (spill crest)
k_spill = 0.02      # m^3/(s·m) simple linear spill above h_max

# Inflow/outflow scheduling (choose outflow so that h never drops below h_min)
w_in_const  = 0.25  # m^3/s
w_out_cmd   = 0.25  # m^3/s nominal target outflow

def T_env(t):
    # Winter day: mean 0 C with ±5 C diurnal swing
    return 0.0 + 5.0*np.sin(2.0*np.pi*t/day)

# ---------------------------
# Discretization (method of lines)
# ---------------------------
N = 180
x = np.linspace(0.0, Ldh, N, endpoint=False)
dx = x[1] - x[0]

def nearest_idx(xgrid, x0, L=Ldh, dxx=None):
    dxx = dx if dxx is None else dxx
    return int(np.round((x0 % L)/dxx)) % len(xgrid)

x_L = 0.0
x_b = np.array([0.2*Ldh, 0.4*Ldh, 0.6*Ldh, 0.8*Ldh])
iL = nearest_idx(x, x_L)
ib = np.array([nearest_idx(x, xb) for xb in x_b], dtype=int)

loss_coef = G_dh_per_m/(rho*cp*Acs)   # s^-1
adv_coef  = vx_const/dx               # 1/s
diff_coef = alpha/(dx*dx)             # 1/s

# ---------------------------
# State vector layout:
# y = [ T_pipe (N),
#       T_LUMI (1),
#       T_B[4] (4),
#       T_Wbld[4] (4),
#       h (1) ]
# ---------------------------
IDX_T0 = 0
IDX_TL = N
IDX_TB = N+1
IDX_TW = N+1+4
IDX_h  = N+1+4+4
STATE_SIZE = N + 1 + 4 + 4 + 1

def rhs(t, y):
    d = np.zeros_like(y)

    # Unpack
    T  = y[IDX_T0:IDX_T0+N]           # pipe cells
    TL = y[IDX_TL]                    # LUMI water
    TB = y[IDX_TB:IDX_TB+4]           # building air
    TW = y[IDX_TW:IDX_TW+4]           # building-side water
    h  = y[IDX_h]                     # reservoir level

    Ten = T_env(t)

    # --- Pipe PDE (periodic) ---
    T_left  = np.roll(T, 1)
    T_right = np.roll(T, -1)
    adv  = -adv_coef*(T - T_left)                         # upwind (vx>0)
    diff =  diff_coef*(T_right - 2.0*T + T_left)          # centered diffusion
    loss = -loss_coef*(T - Ten)

    # Point HXs
    S = np.zeros_like(T)
    # LUMI HX (pipe ↔ LUMI loop) at iL
    S[iL] += (U_L/(rho*cp*Acs))*(TL - T[iL]) / dx
    # Buildings HX (pipe ↔ building-side water) at ib[k]
    for k in range(4):
        S[ib[k]] += (U_ex/(rho*cp*Acs))*(TW[k] - T[ib[k]]) / dx

    dTdt = adv + diff + loss + S
    d[IDX_T0:IDX_T0+N] = dTdt

    # --- LUMI loop ---
    d[IDX_TL] = (Q_LUMI(t) - U_L*(TL - T[iL]))/Cth_L

    # --- Building water nodes and rooms ---
    # Water nodes: Cth_w dTW/dt = U_ex (Tpipe - TW) - U_rad (TW - TB)
    TWdot = (U_ex*(T[ib] - TW) - U_rad*(TW - TB)) / Cth_w
    d[IDX_TW:IDX_TW+4] = TWdot

    # Rooms: C_B dTB/dt = U_rad (TW - TB) + (1/RB_i0)(Ten - TB) + shared walls
    TBdot = np.zeros(4)
    HX_room = U_rad*(TW - TB)
    Amb_term = (1.0/RB_i0)*(Ten - TB)
    TBdot += (HX_room + Amb_term)/CB
    # Shared walls (1<->2, 3<->4)
    TBdot[0] += (1.0/R_shared_12)*(TB[1] - TB[0])/CB[0]
    TBdot[1] += (1.0/R_shared_12)*(TB[0] - TB[1])/CB[1]
    TBdot[2] += (1.0/R_shared_34)*(TB[3] - TB[2])/CB[2]
    TBdot[3] += (1.0/R_shared_34)*(TB[2] - TB[3])/CB[3]
    d[IDX_TB:IDX_TB+4] = TBdot

    # --- Reservoir level (non-depleting control) ---
    win = w_in_const
    wout = w_out_cmd
    # Do not allow depletion: when below or at h_min, cap outflow to at most inflow
    if h <= h_min:
        wout = min(wout, win)
    # Simple spill above crest
    q_spill = max(0.0, k_spill*(h - h_max))
    d[IDX_h] = (win - wout - q_spill)/A_res

    return d

# ---------------------------
# Initial conditions
# ---------------------------
T_init_uniform = 40.0
TL_init = 40.0
TB_init = np.array([21.0, 20.0, 21.0, 20.0])  # start at setpoints
TW_init = np.full(4, 38.0)                    # building-side water, warm-ish
h_init  = 40.0

y0 = np.zeros(STATE_SIZE)
y0[IDX_T0:IDX_T0+N] = T_init_uniform
y0[IDX_TL] = TL_init
y0[IDX_TB:IDX_TB+4] = TB_init
y0[IDX_TW:IDX_TW+4] = TW_init
y0[IDX_h] = h_init

# ---------------------------
# Time grid and solve
# ---------------------------
t0 = 0.0
tf = 72*3600.0
t_eval = np.arange(t0, tf+60.0, 60.0)

print("Running solve_ivp (Q7)...")
sol = solve_ivp(
    fun=rhs,
    t_span=(t0, tf),
    y0=y0,
    t_eval=t_eval,
    method="BDF",
    atol=1e-6,
    rtol=1e-6,
    max_step=120.0
)
print("Solver status:", sol.message)

t = sol.t
Tpipe = sol.y[IDX_T0:IDX_T0+N, :]
TL_series = sol.y[IDX_TL, :]
TB_series = sol.y[IDX_TB:IDX_TB+4, :]
TW_series = sol.y[IDX_TW:IDX_TW+4, :]
h_series  = sol.y[IDX_h, :]

# Diagnostics (rough hydraulic power)
w_vol = wdh_des
Delta_p = R_dh_flow * (w_vol**2)
P_pump = Delta_p * w_vol  # W

# ---------------------------
# Save CSVs
# ---------------------------
df_main = pd.DataFrame({
    "t_s": t,
    "TB1_C": TB_series[0], "TB2_C": TB_series[1],
    "TB3_C": TB_series[2], "TB4_C": TB_series[3],
    "TW1_C": TW_series[0], "TW2_C": TW_series[1],
    "TW3_C": TW_series[2], "TW4_C": TW_series[3],
    "TLUMI_C": TL_series,
    "h_m": h_series
})
df_main.to_csv(os.path.join(OUT_DIR, "buildings_lumi_reservoir_q7.csv"), index=False)

# Pipe temperature probes
probe_fracs = [0.00, 0.25, 0.50, 0.75]
def idx_at_frac(f): return int(np.round((f*Ldh)/dx)) % N
probe_idx = [idx_at_frac(f) for f in probe_fracs]
df_pipe = pd.DataFrame({"t_s": t})
for f, idxp in zip(probe_fracs, probe_idx):
    df_pipe[f"T_pipe_x{f:.2f}_C"] = Tpipe[idxp, :]
df_pipe.to_csv(os.path.join(OUT_DIR, "pipe_probes_q7.csv"), index=False)

# ---------------------------
# KPIs: energy balance & comfort
# ---------------------------
from numpy import trapz

# Heat from LUMI into pipe (W): U_L * (T_LUMI - T_pipe at iL)
P_LUMI2pipe = U_L * (TL_series - Tpipe[iL, :])             # W
E_LUMI2pipe_MJ = trapz(P_LUMI2pipe, t) / 1e6               # MJ

# Ambient vector per time step (Nt,)
env_t = T_env(t)

# Pipe losses to ambient (W): ∫ G_per_m * (T - Ten) dx
P_pipe_loss = G_dh_per_m * (np.sum(Tpipe - env_t[None, :], axis=0) * dx)  # W
E_pipe_loss_MJ = np.trapz(P_pipe_loss, t) / 1e6

# Heat from pipe into building loops (W): sum U_ex * (T(ib) - TW)
P_pipe2bld = np.sum(U_ex * (Tpipe[ib, :] - TW_series), axis=0)            # W
E_pipe2bld_MJ = trapz(P_pipe2bld, t) / 1e6

# Heat from building water to rooms (radiators)
P_rad = np.sum(U_rad[:,None] * (TW_series - TB_series), axis=0)           # W
E_rad_MJ = trapz(P_rad, t) / 1e6

# Comfort drift and time above/below setpoint
Tset = np.array([21.0, 20.0, 21.0, 20.0])
final_minus_init = (TB_series[:,-1] - TB_series[:,0])
dt = np.diff(t)
step = t[1]-t[0]
time_above = np.sum((TB_series.T - Tset) > 0, axis=0) * step / 3600.0     # h
time_below = np.sum((TB_series.T - Tset) < 0, axis=0) * step / 3600.0     # h

# Save a compact KPI table
kpi = pd.DataFrame({
    "metric": [
        "E_LUMI_to_pipe_MJ", "E_pipe_loss_MJ", "E_pipe_to_buildings_MJ", "E_radiators_MJ",
        "Office1_final_minus_init_C", "Res2_final_minus_init_C",
        "Office3_final_minus_init_C", "Res4_final_minus_init_C",
        "Office1_time_above_set_h", "Res2_time_above_set_h",
        "Office3_time_above_set_h", "Res4_time_above_set_h",
        "Office1_time_below_set_h", "Res2_time_below_set_h",
        "Office3_time_below_set_h", "Res4_time_below_set_h",
        "Reservoir_min_h_m", "Reservoir_violated_min_level"
    ],
    "value": [
        E_LUMI2pipe_MJ, E_pipe_loss_MJ, E_pipe2bld_MJ, E_rad_MJ,
        final_minus_init[0], final_minus_init[1], final_minus_init[2], final_minus_init[3],
        time_above[0], time_above[1], time_above[2], time_above[3],
        time_below[0], time_below[1], time_below[2], time_below[3],
        float(np.min(h_series)), float(np.min(h_series) < h_min)
    ]
})
kpi.to_csv(os.path.join(OUT_DIR, "kpis_q7.csv"), index=False)

# Write a small assumptions file for the appendix
with open(os.path.join(OUT_DIR, "assumptions_q7.txt"), "w") as f:
    f.write(
        "Q7 assumptions:\n"
        f"- District pump ON at w_dh={wdh_des:.3f} m^3/s (v={vx_const:.3f} m/s)\n"
        f"- Reservoir protected: h_min={h_min:.1f} m; spill above h_max={h_max:.1f} m with k_spill={k_spill}\n"
        f"- Winter ambient T_env(t)=0+5*sin(2pi t/24h) C; initial pipe and LUMI 40 C\n"
        f"- Building-side radiators sized U_rad={list(U_rad)} W/K; HX U_ex={U_ex} W/K\n"
    )

# ---------------------------
# Plots
# ---------------------------
# 1) Building indoor temperatures
plt.figure(figsize=(8,4.5))
for i in range(4):
    plt.plot(t/3600.0, TB_series[i], label=f"Building {i+1}")
plt.axhline(21.0, ls="--", alpha=0.6, label="Office set 21 C")
plt.axhline(20.0, ls="--", alpha=0.6, label="Res set 20 C")
plt.xlabel("Time [h]"); plt.ylabel("Indoor temperature [C]")
plt.title("Building indoor temperatures (Q7, pumps ON)")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "buildings_temperatures_q7.png"), dpi=160)

# 2) Building-side water temperatures
plt.figure(figsize=(8,4))
for i in range(4):
    plt.plot(t/3600.0, TW_series[i], label=f"Water {i+1}")
plt.xlabel("Time [h]"); plt.ylabel("Building water [C]")
plt.title("Building-side water temperatures")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "building_water_q7.png"), dpi=160)

# 3) LUMI loop
plt.figure(figsize=(8,4))
plt.plot(t/3600.0, TL_series)
plt.xlabel("Time [h]"); plt.ylabel("T_LUMI [C]")
plt.title("LUMI loop temperature")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lumi_loop_temperature_q7.png"), dpi=160)

# 4) Reservoir level (should stay >= h_min)
plt.figure(figsize=(8,3.6))
plt.plot(t/3600.0, h_series, label="h(t)")
plt.axhline(h_min, ls="--", c="k", alpha=0.7, label="h_min")
plt.axhline(h_max, ls="--", c="k", alpha=0.4, label="h_max (spill crest)")
plt.xlabel("Time [h]"); plt.ylabel("Reservoir level h [m]")
plt.title("Hydropower reservoir level (protected)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reservoir_level_q7.png"), dpi=160)

# 5) Pipe probes
plt.figure(figsize=(8,4))
for f, idxp in zip(probe_fracs, probe_idx):
    plt.plot(t/3600.0, Tpipe[idxp, :], label=f"x={f:.2f} L")
plt.xlabel("Time [h]"); plt.ylabel("T_pipe [C]")
plt.title("Pipe temperatures at probes")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pipe_probes_q7.png"), dpi=160)

# 6) Space-time map
ds = slice(None, None, 10)  # stride in time for plotting
plt.figure(figsize=(8,3.6))
plt.imshow(
    Tpipe[:, ds],
    aspect="auto",
    origin="lower",
    extent=[(t[ds][0]/3600.0), (t[ds][-1]/3600.0), 0.0, Ldh],
)
plt.colorbar(label="T_pipe [C]")
plt.xlabel("Time [h]"); plt.ylabel("Arc length x [m]")
plt.title("District-heating pipe temperature (space-time)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pipe_spacetime_q7.png"), dpi=160)

with open(os.path.join(OUT_DIR, "pump_power_q7.txt"), "w") as f:
    f.write(f"Approx hydraulic power (rough): {P_pump/1e3:.2f} kW\n")

print(f"Done. CSVs and plots written to: {OUT_DIR}")

