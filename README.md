# SC42155 mini project

Simulation scripts I wrote for the mini project of SC42155, Modelling of Dynamical Systems, at TU Delft (November 2025). The project models a small district heating network: a supply pipe fed by a heat source the assignment calls LUMI, four buildings (two offices, two residential) with thermostats, a circulation pump, a hydropower reservoir and a backup battery. Each script answers one or two questions of the project and writes its figures and CSV tables to its own output folder.

## Scripts

- `simulations_q2.5.py`: simulates each component on its own: the pipe as a 1D PDE solved by the method of lines, the LUMI heat exchanger, a building with a hysteresis thermostat, the pump characteristic, the hydro reservoir and the battery with state-of-charge limits. Writes plots and KPI tables to `sim_outputs_q2.5/`.
- `simulations_q2.7.py`: couples the pipe, the LUMI loop, the four buildings and the reservoir into one ODE system and simulates a winter day with `solve_ivp`. Writes time series, KPIs, an assumptions file and plots to `sim_outputs_q7/`.
- `simulations_q3.2_q3.4.py`: draws the two-state timed automata for the office and residential pump rules and their forced-off windows (Q3.2), then plots temperatures and pump commands over 24 h for a given temperature sequence (Q3.4). Writes PNGs to `figs/`, or to the folder named by the `OUTDIR` environment variable.
- `simulations_q3.7.py`: discrete-time simulation of the battery automaton (Q3.5) and the supervisor (Q3.6) over two scenarios, a weekday and a cold snap, in 12 ticks of 3 h. Writes CSV timelines and figures to `figs/`.
- `simulate_q4_3.py`: hybrid simulation of the whole network with an explicit finite-difference pipe solver (upwind advection, SSP-RK3 time stepping), reservoir modes (normal, spill, dry), switching between hydro and battery power, and the thermostat-driven pump command. Writes PDF figures to `sim_outputs_q4/`.

## Running

Each script is standalone, for example:

```
python simulations_q2.5.py
```

Requires Python 3 with numpy, scipy, pandas and matplotlib.
