import numpy as np
import pandas as pd
import math
from dataclasses import dataclass
from typing import Callable, Dict
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# ---------- Utilities ----------

def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

def trapz(y: np.ndarray, t: np.ndarray) -> float:
    return float(np.trapz(y, t))

# ---------- Parameters (refined) ----------

@dataclass
class PhysParams:
    rho: float = 1000.0                 # kg/m^3
    cp: float = 4180.0                  # J/kg/K
    k_dh: float = 0.60                  # W/m/K
    L_dh: float = 1500.0                # m
    D_i: float = 0.15                   # m
    G_lin: float = 0.25                 # W/K/m
    T_env0: float = 10.0                # degC

    @property
    def A(self) -> float:
        return math.pi * (self.D_i ** 2) / 4.0

    @property
    def alpha(self) -> float:
        return self.k_dh / (self.rho * self.cp)


@dataclass
class PumpParams:
    omega_max_rpm: float = 3000.0
    kappa_pump: float = 9.6e-5          # m^3 per rad
    @staticmethod
    def rpm_to_rads(omega_rpm: float) -> float:
        return omega_rpm * 2.0 * math.pi / 60.0


@dataclass
class LumiParams:
    V_water: float = 2.0                # m^3
    U_L: float = 9300.0                 # W/K (refined to get ~15 min time constant)
    Q_nom: float = 150e3                # W nominal

    def C(self, phys: PhysParams) -> float:
        return phys.rho * phys.cp * self.V_water


@dataclass
class BuildingParams:
    C: float = 1.2e8                    # J/K (office exemplar)
    R_amb: float = 2.5e-3               # K/W
    U_ex: float = 1650.0                # W/K (refined)
    T_set: float = 21.0                 # degC
    hyst: float = 0.5                   # degC (use 0.6 if you want fewer cycles)


@dataclass
class HydroParams:
    A_res: float = 5000.0               # m^2
    h_max: float = 50.0                 # m
    w_out_max: float = 0.40             # m^3/s
    eps_turb: float = 0.85              # -


@dataclass
class BatteryParams:
    E_max_kWh: float = 300.0            # kWh
    eta_ch: float = 0.97
    eta_dis: float = 0.97
    V_dc: float = 600.0                 # V
    def E_max_J(self) -> float:
        return self.E_max_kWh * 1000.0 * 3600.0

# ---------- Pipe (PDE -> Method of Lines) ----------

@dataclass
class PipeGrid:
    N: int
    L: float
    @property
    def dx(self) -> float:
        return self.L / self.N
    @property
    def x(self) -> np.ndarray:
        return np.linspace(0.0, self.L - self.dx, self.N)

def roll(arr: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(arr, shift)

def pipe_rhs_factory(phys: PhysParams, grid: PipeGrid,
                     v_fn: Callable[[float], float],
                     Tenv_fn: Callable[[float, np.ndarray], np.ndarray],
                     S_fn: Callable[[float, np.ndarray], np.ndarray]):
    alpha = phys.alpha
    coeff_loss = phys.G_lin / (phys.rho * phys.cp * phys.A)
    def rhs(t: float, T: np.ndarray) -> np.ndarray:
        v = v_fn(t)
        # upwind advection
        if v >= 0:
            dTdx = (T - roll(T, 1)) / grid.dx
        else:
            dTdx = (roll(T, -1) - T) / grid.dx
        # diffusion
        T_ip1 = roll(T, -1); T_im1 = roll(T, 1)
        d2Tdx2 = (T_im1 - 2.0*T + T_ip1) / (grid.dx**2)
        Tenv = Tenv_fn(t, grid.x)
        S = S_fn(t, grid.x)  # [W/m]
        return -v*dTdx + alpha*d2Tdx2 - coeff_loss*(T - Tenv) + S/(phys.rho*phys.cp*phys.A)
    return rhs

def constant_env_fn(T_env0: float):
    return lambda t, x: np.full_like(x, T_env0, dtype=float)

def point_source_fn(x0: float, qW: Callable[[float], float], grid: PipeGrid):
    idx = int(round((x0 / grid.L) * grid.N)) % grid.N
    def S(t: float, x: np.ndarray) -> np.ndarray:
        Svec = np.zeros(grid.N, dtype=float)
        Svec[idx] = qW(t) / grid.dx
        return Svec
    return S

# ---------- LUMI ODE ----------

def lumi_rhs(t: float, x: np.ndarray, phys: PhysParams, p: LumiParams,
             Q_LUMI: Callable[[float], float], T_dh: Callable[[float], float]) -> np.ndarray:
    T_L = x[0]
    C_L = p.C(phys)
    return np.array([(Q_LUMI(t) - p.U_L*(T_L - T_dh(t))) / C_L])

# ---------- Building ODE + Thermostat ----------

class Thermostat:
    def __init__(self, setpoint: float, hyst: float, initial_state: int = 1):
        self.sp = setpoint; self.h = hyst; self.state = initial_state
    def update(self, T_B: float):
        if T_B < (self.sp - self.h): self.state = 1
        elif T_B > (self.sp + self.h): self.state = 0
        return self.state

def building_rhs(t: float, x: np.ndarray, p: BuildingParams,
                 T_env: Callable[[float], float],
                 T_dh: Callable[[float], float],
                 th: Thermostat) -> np.ndarray:
    T_B = x[0]
    u = th.update(T_B)
    dTdt = (u*p.U_ex*(T_dh(t) - T_B) + (T_env(t) - T_B)/p.R_amb) / p.C
    return np.array([dTdt])

# ---------- Hydro ODE ----------

def hydro_rhs(t: float, x: np.ndarray, p: HydroParams,
              w_in: Callable[[float], float],
              w_out: Callable[[float], float]) -> np.ndarray:
    h = x[0]
    win  = w_in(t)
    wout = min(w_out(t), p.w_out_max)

    # Simple guarding to keep h in [0, h_max]
    if h <= 0.0 and win <= wout:
        wout = win         # prevent further draining
    if h >= p.h_max and win >= wout:
        win = wout         # prevent overflow

    return np.array([(win - wout) / p.A_res])

def hydro_power(h: float, w_out: float, p: HydroParams, phys: PhysParams) -> float:
    h_eff = min(max(h, 0.0), p.h_max)
    return p.eps_turb * phys.rho * 9.81 * h_eff * min(w_out, p.w_out_max)

# ---------- Battery ODE ----------

def battery_rhs(t: float, x: np.ndarray, p: BatteryParams,
                P_ch: Callable[[float], float],
                P_dis: Callable[[float], float]) -> np.ndarray:
    E = x[0]
    return np.array([p.eta_ch*P_ch(t) - (P_dis(t)/p.eta_dis)])

# ---------- Scenarios ----------

def daily_ambient(t: float, T_mean: float = 10.0, amp: float = 5.0, phase: float = 0.0) -> float:
    return T_mean + amp * math.sin(2.0 * math.pi * (t / 86400.0) + phase)

# ---------- Component simulations ----------

OUT_DIR = "sim_outputs_q2.5"

def simulate_pipe() -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)
    phys = PhysParams()
    pump = PumpParams()
    v_nom = pump.kappa_pump * pump.rpm_to_rads(3000.0) / phys.A
    grid = PipeGrid(N=150, L=phys.L_dh)

    q_pulse = lambda t: 50e3 if t <= 300.0 else 0.0
    S_fn = point_source_fn(0.0, q_pulse, grid)
    v_fn = lambda t: v_nom
    Tenv_fn = constant_env_fn(phys.T_env0)
    rhs = pipe_rhs_factory(phys, grid, v_fn, Tenv_fn, S_fn)

    T0 = np.full(grid.N, 20.0, dtype=float)
    t_eval = np.linspace(0.0, 1200.0, 601)
    sol = solve_ivp(lambda t, y: rhs(t, y), (t_eval[0], t_eval[-1]), T0, t_eval=t_eval, rtol=1e-6, atol=1e-8)

    Tmat = sol.y.T
    # Plots
    plt.figure()
    plt.imshow(Tmat, aspect='auto', origin='lower', extent=[grid.x[0], grid.x[-1], t_eval[0], t_eval[-1]])
    plt.xlabel("x [m]"); plt.ylabel("t [s]"); plt.title("Pipe temperature T(x,t) [°C]")
    savefig(os.path.join(OUT_DIR, "pipe_spacetime.png"))

    for t_snap in [0, 300, 600, 1200]:
        idx = np.argmin(np.abs(t_eval - t_snap))
        plt.figure(); plt.plot(grid.x, Tmat[idx, :])
        plt.xlabel("x [m]"); plt.ylabel("T [°C]"); plt.title(f"Pipe snapshot t={t_snap}s")
        savefig(os.path.join(OUT_DIR, f"pipe_snapshot_t{t_snap}.png"))

    # KPIs
    Pe = abs(v_nom) * phys.L_dh / phys.alpha
    E_pipe = phys.rho * phys.cp * phys.A * grid.dx * np.sum((Tmat - 20.0), axis=1)
    Ein = np.array([trapz(np.array([q_pulse(t) for t in t_eval[:i+1]]), t_eval[:i+1]) for i in range(len(t_eval))])
    loss = Ein - E_pipe  # J
    plt.figure()
    plt.plot(t_eval, E_pipe, label="Pipe energy (J, rel. 20°C)")
    plt.plot(t_eval, Ein,    label="Cumulative injected (J)")
    plt.plot(t_eval, loss,   label="Loss to ambient (J)")
    plt.xlabel("t [s]"); plt.ylabel("Energy [J]"); plt.title("Pipe energy balance"); plt.legend()
    savefig(os.path.join(OUT_DIR, "pipe_energy.png"))

    df = pd.DataFrame({"Pe":[Pe], "v_nom [m/s]":[v_nom], "N":[grid.N], "L [m]":[phys.L_dh],
                       "alpha [m^2/s]":[phys.alpha], "G_lin [W/K/m]":[phys.G_lin]})
    df.to_csv(os.path.join(OUT_DIR, "pipe_kpi.csv"), index=False)
    return df

def simulate_lumi() -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)
    phys = PhysParams(); p = LumiParams()

    Q_LUMI = lambda t: 50e3 if t < 600.0 else 150e3
    T_dh = lambda t: 40.0
    x0 = np.array([35.0])
    t_eval = np.linspace(0.0, 3600.0, 721)
    sol = solve_ivp(lambda t, x: lumi_rhs(t, x, phys, p, Q_LUMI, T_dh),
                    (t_eval[0], t_eval[-1]), x0, t_eval=t_eval, rtol=1e-7, atol=1e-9)
    T_L = sol.y[0, :]
    q_L = p.U_L * (T_L - 40.0)

    # Rough tau via 63% criterion after step at 600s
    idx_step = np.argmin(np.abs(t_eval - 600.0))
    y0 = T_L[idx_step]; yss = float(np.mean(T_L[-50:]))
    target = y0 + 0.632*(yss - y0)
    idx_tau = np.where(T_L[idx_step:] >= target)[0]
    tau = float(t_eval[idx_step + idx_tau[0]] - 600.0) if len(idx_tau)>0 else float('nan')

    plt.figure(); plt.plot(t_eval, T_L); plt.xlabel("t [s]"); plt.ylabel("T_L [°C]"); plt.title("LUMI loop temperature")
    savefig(os.path.join(OUT_DIR, "lumi_temperature.png"))
    plt.figure(); plt.plot(t_eval, q_L); plt.xlabel("t [s]"); plt.ylabel("q_L [W]"); plt.title("LUMI heat exchange power")
    savefig(os.path.join(OUT_DIR, "lumi_heat_exchange.png"))

    df = pd.DataFrame({"C_L [J/K]":[p.C(phys)], "U_L [W/K]":[p.U_L], "tau_est [s]":[tau], "T_ss [°C]":[yss]})
    df.to_csv(os.path.join(OUT_DIR, "lumi_kpi.csv"), index=False)
    return df

def simulate_building(preheat: float = 20.5, hyst: float = 0.5) -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)
    pB = BuildingParams(hyst=hyst)

    # Ambient and DH supply
    T_env = lambda t: daily_ambient(t, T_mean=10.0)
    T_dh  = lambda t: 45.0

    # Thermostat used inside the ODE (keeps its internal state)
    th = Thermostat(setpoint=pB.T_set, hyst=pB.hyst, initial_state=1)

    # Integrate
    x0 = np.array([preheat])
    t_eval = np.linspace(0.0, 24.0*3600.0, 1441)  # 1-min grid
    sol = solve_ivp(lambda t, x: building_rhs(t, x, pB, T_env, T_dh, th),
                    (t_eval[0], t_eval[-1]), x0, t_eval=t_eval, rtol=1e-6, atol=1e-8)
    T_B = sol.y[0, :]

    # Reconstruct u(t) from the temperature trace with hysteresis memory
    th_replay = Thermostat(setpoint=pB.T_set, hyst=pB.hyst, initial_state=1)
    u_t = np.zeros_like(t_eval)
    for i in range(len(t_eval)):
        u_t[i] = th_replay.update(T_B[i])

    # Gate exchanger power by thermostat and enforce non-negativity
    Tsup = np.array([T_dh(t) for t in t_eval])   # allow time-varying supply
    q_raw = pB.U_ex * (Tsup - T_B)               # W
    q_i   = u_t * np.maximum(q_raw, 0.0)         # W


    # KPIs
    comfort = float(np.mean(np.abs(T_B - pB.T_set) <= 0.5))
    duty    = float(np.mean(u_t))
    E_daily = trapz(q_i, t_eval)                 # J

    # Plots
    plt.figure()
    plt.plot(t_eval/3600.0, T_B)
    plt.axhline(pB.T_set-0.5, linestyle="--")
    plt.axhline(pB.T_set+0.5, linestyle="--")
    plt.xlabel("t [h]"); plt.ylabel("T_B [°C]")
    plt.title(f"Building temperature (U_ex={pB.U_ex:.0f} W/K, hyst={pB.hyst:.1f} °C)")
    savefig(os.path.join(OUT_DIR, "building_temperature.png"))

    plt.figure()
    plt.plot(t_eval/3600.0, u_t)
    plt.xlabel("t [h]"); plt.ylabel("u(t)")
    plt.title("Thermostat state (1=on, 0=off)")
    savefig(os.path.join(OUT_DIR, "building_u.png"))

    plt.figure()
    plt.plot(t_eval/3600.0, q_i)
    plt.xlabel("t [h]"); plt.ylabel("q_i [W]")
    plt.title("Building heat draw (gated by thermostat)")
    savefig(os.path.join(OUT_DIR, "building_heat.png"))

    # Table
    df = pd.DataFrame({
        "U_ex [W/K]":[pB.U_ex],
        "hysteresis [°C]":[pB.hyst],
        "preheat [°C]":[preheat],
        "comfort_fraction":[comfort],
        "duty_cycle":[duty],
        "daily_energy_MJ":[E_daily/1e6],
    })
    df.to_csv(os.path.join(OUT_DIR, "building_kpi.csv"), index=False)
    return df


def simulate_pump() -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)
    pump = PumpParams()
    omegas = np.linspace(0.0, pump.omega_max_rpm, 50)
    flows = pump.kappa_pump * pump.rpm_to_rads(omegas)

    plt.figure(); plt.plot(omegas, flows); plt.xlabel("Pump speed [rpm]"); plt.ylabel("Flow w_dh [m^3/s]"); plt.title("Pump characteristic (ideal)")
    savefig(os.path.join(OUT_DIR, "pump_characteristic.png"))

    slope = float(np.polyfit(omegas, flows, 1)[0])
    df = pd.DataFrame({"omega_max_rpm":[pump.omega_max_rpm], "slope_m3s_per_rpm":[slope], "flow_at_max_m3s":[flows[-1]]})
    df.to_csv(os.path.join(OUT_DIR, "pump_kpi.csv"), index=False)
    return df

def simulate_hydro() -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)
    phys = PhysParams(); p = HydroParams()
    w_in = lambda t: 0.15 + 0.05 * math.sin(2.0*math.pi*(t/(24*3600.0)))
    w_out = lambda t: 0.10 if t < 12*3600.0 else 0.30
    x0 = np.array([20.0])
    t_eval = np.linspace(0.0, 36.0*3600.0, 721)
    sol = solve_ivp(lambda t, x: hydro_rhs(t, x, p, w_in, w_out),
                    (t_eval[0], t_eval[-1]), x0, t_eval=t_eval, rtol=1e-6, atol=1e-8)
    h = sol.y[0, :]
    P = np.array([hydro_power(h[i], w_out(t_eval[i]), p, phys) for i in range(len(t_eval))])
    E = trapz(P, t_eval)

    plt.figure(); plt.plot(t_eval/3600.0, h); plt.xlabel("t [h]"); plt.ylabel("Head h [m]"); plt.title("Reservoir head")
    savefig(os.path.join(OUT_DIR, "hydro_head.png"))
    plt.figure(); plt.plot(t_eval/3600.0, P/1e3); plt.xlabel("t [h]"); plt.ylabel("P_hydro [kW]"); plt.title("Hydro power")
    savefig(os.path.join(OUT_DIR, "hydro_power.png"))

    df = pd.DataFrame({"h_initial_m":[x0[0]], "h_final_m":[h[-1]], "E_generated_MJ":[E/1e6], "P_max_kW":[np.max(P)/1e3]})
    df.to_csv(os.path.join(OUT_DIR, "hydro_kpi.csv"), index=False)
    return df

def simulate_battery() -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)
    p = BatteryParams()
    Emax = p.E_max_J()
    P_LIMIT = 167e3  # W  (same magnitude as hydropower max)

    # Commanded power schedules
    P_ch_cmd  = lambda t: 80e3  if 0.0 <= t < 3.0*3600.0 else 0.0
    P_dis_cmd = lambda t: 120e3 if 3.5*3600.0 <= t < 6.5*3600.0 else 0.0

    # Time grid and storage
    t_eval = np.linspace(0.0, 8.0*3600.0, 801)           # dt = 36 s
    dt = t_eval[1] - t_eval[0]
    E   = np.zeros_like(t_eval)
    E[0] = 0.20 * Emax                                    # initial 20% SOC

    P_ch_applied  = np.zeros_like(t_eval)
    P_dis_applied = np.zeros_like(t_eval)

    # Stepwise update with saturation by SOC & power limits
    for k in range(len(t_eval)-1):
        t = t_eval[k]

        # Commands
        Pch  = max(0.0, min(P_ch_cmd(t),  P_LIMIT))
        Pdis = max(0.0, min(P_dis_cmd(t), P_LIMIT))

        # SOC-based availability (accounting for efficiencies in the energy update)
        headroom_J   = Emax - E[k]                       # how much energy we can add
        available_J  = E[k]                               # how much we can take out

        # For charge:  ΔE = η_ch * Pch * dt  ⇒ Pch ≤ headroom / (η_ch * dt)
        if headroom_J <= 0.0:
            Pch = 0.0
        else:
            Pch = min(Pch, headroom_J / (p.eta_ch * dt))

        # For discharge:  ΔE = (Pdis/η_dis) * dt  ⇒ Pdis ≤ available * η_dis / dt
        if available_J <= 0.0:
            Pdis = 0.0
        else:
            Pdis = min(Pdis, available_J * p.eta_dis / dt)

        # Apply and integrate one step
        P_ch_applied[k]  = Pch
        P_dis_applied[k] = Pdis
        dE = p.eta_ch * Pch * dt - (Pdis / p.eta_dis) * dt
        E[k+1] = np.clip(E[k] + dE, 0.0, Emax)

    # Final sample powers (for plotting alignment)
    P_ch_applied[-1]  = P_ch_applied[-2]
    P_dis_applied[-1] = P_dis_applied[-2]

    # Metrics & plots
    SOC  = E / Emax
    Pnet = p.eta_ch * P_ch_applied - (P_dis_applied / p.eta_dis)
    I_dc = Pnet / p.V_dc

    Ein  = trapz(P_ch_applied,  t_eval) * p.eta_ch
    Eout = trapz(P_dis_applied, t_eval) / p.eta_dis
    rte  = Eout / Ein if Ein > 0 else float("nan")

    plt.figure(); plt.plot(t_eval/3600.0, SOC)
    plt.xlabel("t [h]"); plt.ylabel("SOC [-]"); plt.title("Battery SOC (with saturation)")
    savefig(os.path.join(OUT_DIR, "battery_soc.png"))

    plt.figure(); plt.plot(t_eval/3600.0, I_dc)
    plt.xlabel("t [h]"); plt.ylabel("I_dc [A]"); plt.title("DC current")
    savefig(os.path.join(OUT_DIR, "battery_idc.png"))

    df = pd.DataFrame({
        "SOC_initial":[SOC[0]], "SOC_final":[SOC[-1]],
        "round_trip_efficiency":[rte],
        "I_dc_max_A":[np.max(np.abs(I_dc))]
    })
    df.to_csv(os.path.join(OUT_DIR, "battery_kpi.csv"), index=False)
    return df


# ---------- Run all & Summary ----------

def run_all(preheat: float = 20.5, hyst: float = 0.5) -> str:
    kpi: Dict[str, pd.DataFrame] = {}
    kpi["pipe"] = simulate_pipe()
    kpi["lumi"] = simulate_lumi()
    kpi["building"] = simulate_building(preheat=preheat, hyst=hyst)
    kpi["pump"] = simulate_pump()
    kpi["hydro"] = simulate_hydro()
    kpi["battery"] = simulate_battery()
    # summary
    flat = {}
    for comp, df in kpi.items():
        for col in df.columns:
            flat[f"{comp}:{col}"] = df.iloc[0][col]
    summary = pd.DataFrame([flat])
    path = os.path.join(OUT_DIR, "summary_kpis.csv")
    summary.to_csv(path, index=False)
    return path

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    path = run_all(preheat=20.5, hyst=0.5)
    print("Done. Summary KPIs at:", path)
