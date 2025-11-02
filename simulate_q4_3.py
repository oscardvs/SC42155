import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------
# Matplotlib defaults (stable layout/legends)
# ----------------------------
matplotlib.rcParams["figure.dpi"] = 160
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["figure.constrained_layout.use"] = True

# ----------------------------
# Output directory
# ----------------------------
OUTDIR = "sim_outputs_q4"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------
# Physical constants
# ----------------------------
g = 9.81
rho = 1000.0         # kg/m^3
cp = 4180.0          # J/(kg K)

# ----------------------------
# Pipe and heating network
# ----------------------------
L_dh = 4000.0        # m
A_cs = 0.0314        # m^2  (pipe cross-section)
v_on = 0.50          # m/s at full power
kappa = 1.0e-6       # m^2/s (thermal diffusivity)
G_per_m = 3.0        # W/K per meter (to ambient)
gamma_sink = G_per_m / (rho * cp * A_cs)  # 1/s, ambient linear sink term

# Space grid
N = 400
dx = L_dh / N
x = np.linspace(0.0, L_dh, N)

# ----------------------------
# Reservoir and hydro
# ----------------------------
A_surf = 5.0e4       # m^2 (constant surface area)
h_min, h_max = 5.0, 25.0
h = 18.0             # initial level
eps_h = 1e-5         # tolerance for guards

q_t_max = 2.0        # m^3/s
H_full = h_max - h_min
R_flow = np.sqrt(rho * g * H_full) / q_t_max  # choose R so that q_t_max at full head
eps_eta = 0.85       # turbine-generator efficiency

# ----------------------------
# Battery and pump power
# ----------------------------
P_req = 100.0        # kW required at full pump speed v_on
P_max = eps_eta * rho * g * (H_full) * q_t_max / 1000.0  # kW, hydro rated at full head
P_max_bat = P_max    # battery peak = hydro peak (staff clarification)
E_nom_kWh = 500.0
z = 0.60             # initial SoC (0..1)

# ----------------------------
# Buildings and thermostats
# ----------------------------
C_B = np.array([3.5e8, 2.0e8])                 # J/K (office, residential)
R_B0 = np.array([0.005, 0.005])                # K/W
Q_int = np.array([1500.0, 800.0])              # W
T_r = np.array([21.0, 20.0])                   # °C
eps_T = np.array([0.5, 0.5])                   # K
k_rad = np.array([3000.0, 2000.0])             # W/K (effective radiator)
tap_pos = np.array([1200.0, 2800.0])           # m along loop
tap_idx = np.clip((tap_pos / dx).astype(int), 0, N-1)
p = np.array([0, 0], dtype=int)                # thermostat states
T_build = T_r.copy()

thermo_check_dt = 15*60.0  # s
t_next_check = thermo_check_dt

# ----------------------------
# LUMI heat generation schedule
# ----------------------------
q1, q2, q3 = 0.6e6, 1.2e6, 2.0e6   # W
slots = [q2, q3, q2, q1, q2, q3, q1, q2, q1]  # 9 slots x 4 h = 36 h
slot_hours = 4.0
slot_seconds = int(slot_hours*3600)

def Q_LUMI_at(t):
    idx = int(t // slot_seconds)
    idx = min(idx, len(slots)-1)
    return slots[idx]

def T_env_at(t):
    return -0.5 + 2.5*np.sin(2*np.pi*(t/86400.0))

def w_in_at(t):
    return 1.2 + 0.3*np.sin(2*np.pi*(t/86400.0 - 0.1))

def inlet_T_setpoint_from_Q(Q):
    if Q <= q1 + 1e-3:
        return 35.0
    elif Q <= q2 + 1e-3:
        return 45.0
    else:
        return 55.0

# ----------------------------
# Time step via CFL
# ----------------------------
CFL_adv = 0.9
CFL_diff = 0.45
dt_adv = CFL_adv * dx / max(v_on, 1e-9)
dt_diff = CFL_diff * dx*dx / max(kappa, 1e-12)
dt = min(dt_adv, dt_diff, 5.0)  # cap for robustness
T_end = 36*3600.0
num_steps = int(np.ceil(T_end/dt))
dt = T_end / num_steps

# ----------------------------
# Logging arrays
# ----------------------------
keep_every = max(1, int(10.0/dt))
snapshots = []
snap_times = []

tt = np.zeros(num_steps+1)
HH = np.zeros(num_steps+1)
PP_hydro = np.zeros(num_steps+1)
SP = np.zeros(num_steps+1, dtype=int)
SBG = np.zeros(num_steps+1, dtype=int)  # 0=g, 1=b
Z = np.zeros(num_steps+1)
ONcount = np.zeros(num_steps+1)
T1 = np.zeros(num_steps+1)
T2 = np.zeros(num_steps+1)

T_pipe = np.full(N, 30.0)  # °C initial
s_mode = "NORM"

# helpers
def advective_upwind(T, v):
    if abs(v) < 1e-12:
        return np.zeros_like(T)
    if v >= 0:
        dT = (T - np.roll(T, 1)) / dx
        return -v * dT
    else:
        dT = (np.roll(T, -1) - T) / dx
        return -v * dT

def diffusion_centered(T, kappa):
    return kappa * (np.roll(T, -1) - 2*T + np.roll(T, 1)) / (dx*dx)

# initial logs
Z[0] = z
HH[0] = h
T1[0], T2[0] = T_build
SP[0] = int(np.any(p))
SBG[0] = 0
PP_hydro[0] = 0.0
ONcount[0] = p.sum()
tt[0] = 0.0
snapshots.append(T_pipe.copy())
snap_times.append(0.0)

# safety bounds for temperature
T_MIN, T_MAX = -20.0, 120.0

# ----------------------------
# Main loop
# ----------------------------
for k in range(1, num_steps+1):
    t = k*dt

    # environment and LUMI
    T_env = T_env_at(t)
    Q_L = Q_LUMI_at(t)
    T_inlet = inlet_T_setpoint_from_Q(Q_L)
    w_in = w_in_at(t)

    # supervisor: pump command from thermostats
    S_p = 1 if np.any(p) else 0

    # turbine command: u = 1 when pump requests flow, else 0
    u = 1.0 if S_p == 1 else 0.0

    # hydro flows based on head
    H = max(h - h_min, 0.0)
    q_cmd = u * (1.0 / R_flow) * np.sqrt(rho * g * H) if H > 0 else 0.0
    q_t = min(q_cmd, q_t_max)

    # mode update by guards
    if (h >= h_max - eps_h) and (w_in > q_t + 1e-9):
        s_mode = "SPILL"
    elif (h <= h_min + eps_h) and (w_in < q_t - 1e-9):
        s_mode = "DRY"
    elif (s_mode == "SPILL") and (h >= h_max - eps_h) and (w_in <= q_t + 1e-9):
        s_mode = "NORM"
    elif (s_mode == "DRY") and (h <= h_min + eps_h) and (w_in >= q_t - 1e-9):
        s_mode = "NORM"

    # effective flows
    if s_mode == "NORM":
        q_eff = q_t
        q_spill = 0.0
    elif s_mode == "SPILL":
        q_eff = q_t
        q_spill = max(w_in - q_t, 0.0)
    else:  # DRY
        q_eff = min(q_t, w_in)
        q_spill = 0.0

    # reservoir update
    h_dot = (w_in - q_eff - q_spill) / A_surf
    h = np.clip(h + dt * h_dot, h_min, h_max)

    # hydro power (kW)
    P_hydro = eps_eta * rho * g * H * q_eff / 1000.0

    # source selection S_bg
    S_bg = 0 if ((P_hydro >= P_req) and (h >= h_min + 1.0)) else 1  # 0=g, 1=b

    # available power and pump speed factor
    if S_bg == 0:
        P_avail = min(P_hydro, P_max)
    else:
        P_avail = min(P_max_bat, P_max) if z > 0.0 else 0.0

    eta_pow = min(1.0, P_avail / P_req) if S_p == 1 else 0.0
    v_dh = S_p * eta_pow * v_on

    # battery SoC update
    if (S_bg == 1) and (S_p == 1) and (P_avail > 0.0):
        P_draw = P_req * eta_pow  # kW
        z = max(z - (P_draw / E_nom_kWh) * (dt / 3600.0), 0.0)

    # pipe RHS
    def pipe_rhs(T):
        adv = advective_upwind(T, v_dh)
        diff = diffusion_centered(T, kappa)
        sink = -gamma_sink * (T - T_env)
        return adv + diff + sink

    # SSP-RK3
    k1 = pipe_rhs(T_pipe)
    T1_pipe = T_pipe + dt*k1
    k2 = pipe_rhs(T1_pipe)
    T2_pipe = 0.75*T_pipe + 0.25*(T1_pipe + dt*k2)
    k3 = pipe_rhs(T2_pipe)
    T_pipe = (1.0/3.0)*T_pipe + (2.0/3.0)*(T2_pipe + dt*k3)

    # impose inlet temperature by clamping (acts as LUMI heater)
    if S_p == 1:
        T_pipe[0] = T_inlet

    # safety clip to avoid rare numerical spikes
    np.clip(T_pipe, T_MIN, T_MAX, out=T_pipe)

    # building supply temps from taps
    T_supply_1 = float(T_pipe[tap_idx[0]])
    T_supply_2 = float(T_pipe[tap_idx[1]])

    # building temperature ODEs (explicit Euler)
    dT1 = (T_env - T_build[0])/(R_B0[0]*C_B[0]) \
          + p[0] * (k_rad[0]/C_B[0]) * (T_supply_1 - T_build[0]) \
          + Q_int[0]/C_B[0]
    dT2 = (T_env - T_build[1])/(R_B0[1]*C_B[1]) \
          + p[1] * (k_rad[1]/C_B[1]) * (T_supply_2 - T_build[1]) \
          + Q_int[1]/C_B[1]
    T_build[0] += dt * dT1
    T_build[1] += dt * dT2

    # thermostat evaluation every 15 minutes
    if t >= t_next_check - 1e-9:
        for i in range(2):
            if (p[i] == 0) and (T_build[i] <= T_r[i] - eps_T[i]):
                p[i] = 1
            elif (p[i] == 1) and (T_build[i] >= T_r[i] + eps_T[i]):
                p[i] = 0
        t_next_check += thermo_check_dt

    # logs
    tt[k] = t
    HH[k] = h
    T1[k] = T_build[0]
    T2[k] = T_build[1]
    SP[k] = S_p
    SBG[k] = S_bg
    Z[k] = z
    PP_hydro[k] = P_hydro
    ONcount[k] = p.sum()

    if k % keep_every == 0:
        snapshots.append(T_pipe.copy())
        snap_times.append(t)

# snapshots for spacetime
snapshots = np.vstack(snapshots)         # shape (n_snaps, N)
snap_times = np.array(snap_times)        # seconds
th = tt/3600.0                           # hours for plots

# ----------------------------
# Figure 1: pipe spacetime (fixed layout, no legend)
# ----------------------------
fig1, ax = plt.subplots(figsize=(7.0, 3.0))
extent = [0, snap_times[-1]/3600.0, 0, L_dh/1000.0]
im = ax.imshow(snapshots.T, origin="lower", aspect="auto", extent=extent, interpolation="nearest", cmap="viridis")
cbar = fig1.colorbar(im, ax=ax, pad=0.015)
cbar.set_label("Pipe temperature [°C]")
for pos in tap_pos/1000.0:
    ax.axhline(pos, ls="--", lw=0.8, color="k", alpha=0.6, dashes=(3, 3))
ax.set_xlabel("Time [h]")
ax.set_ylabel("Distance along loop [km]")
ax.set_title("Pipe temperature space–time map")
fig1.savefig(os.path.join(OUTDIR, "pipe_spacetime_hybrid.pdf"))
plt.close(fig1)

# ----------------------------
# Figure 2: building temperatures (legend fixed inside)
# ----------------------------
fig2, ax = plt.subplots(figsize=(7.0, 3.0))
ax.plot(th, T1, lw=2.0, label="Building 1")
ax.plot(th, T2, lw=2.0, label="Building 2")
ax.fill_between(th, T_r[0]-eps_T[0], T_r[0]+eps_T[0], color="#90caf955", zorder=0)
ax.fill_between(th, T_r[1]-eps_T[1], T_r[1]+eps_T[1], color="#ffcc8055", zorder=0)
ax.set_xlabel("Time [h]")
ax.set_ylabel("Air temperature [°C]")
ax.set_title("Building temperatures with hysteresis bands")
ax.grid(True, ls=":", lw=0.6)
ax.legend(loc="upper right", ncol=2, frameon=True, columnspacing=1.0, handlelength=2.2)
fig2.savefig(os.path.join(OUTDIR, "building_temperatures.pdf"))
plt.close(fig2)

# ----------------------------
# Figure 3: hydro level and power (combined legend below)
# ----------------------------
fig3, ax1 = plt.subplots(figsize=(7.0, 3.0))
l_h, = ax1.plot(th, HH, color="#2e7d32", lw=2.0, label="Level h [m]")
ax1.axhline(h_min, color="k", ls="--", lw=0.8, alpha=0.6)
ax1.axhline(h_max, color="k", ls="--", lw=0.8, alpha=0.6)
ax1.set_xlabel("Time [h]")
ax1.set_ylabel("Level h [m]")
ax1.grid(True, ls=":", lw=0.6)

ax2 = ax1.twinx()
l_ph, = ax2.plot(th, PP_hydro, color="#1976d2", lw=2.0, label="Hydro power used [kW]")
l_pd, = ax2.step(th, np.full_like(th, P_req), where="post", color="#6d4c41", lw=1.2, label="Pump demand (reference) [kW]")
ax2.set_ylabel("Power [kW]")

handles = [l_h, l_ph, l_pd]
labels  = [h_.get_label() for h_ in handles]
ax1.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True)
fig3.subplots_adjust(bottom=0.25)
ax1.set_title("Reservoir level and power split")
fig3.savefig(os.path.join(OUTDIR, "hydro_level_power_split.pdf"))
plt.close(fig3)

# ----------------------------
# Figure 4: supervisor signals (legend above)
# ----------------------------
fig4, ax = plt.subplots(figsize=(7.0, 3.0))
l_sp, = ax.step(th, SP, where="post", lw=2.0, label=r"$S_p$ (pump on=1)")
# convert SBG: 0=g,1=b  -> plot grid=1
l_bg, = ax.step(th, 1-SBG, where="post", lw=2.0, label=r"$S_{bg}$ (grid=1, battery=0)")
l_on, = ax.step(th, ONcount, where="post", lw=2.0, label="ON-count (buildings)")
ax.set_ylim(-0.1, max(1.1, np.nanmax(ONcount)+0.3))
ax.set_xlabel("Time [h]")
ax.set_title("Supervisor signals and building ON-count")
ax.grid(True, ls=":", lw=0.6)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True)
fig4.savefig(os.path.join(OUTDIR, "supervisor_signals.pdf"))
plt.close(fig4)

print(f"Wrote figures to: {OUTDIR}")

