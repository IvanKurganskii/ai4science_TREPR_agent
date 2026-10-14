"""Run from project root: python -m src.mars_simulation.examples.workflow."""

import json

from src.mars_simulation import MarsSimulationTools


def main():
    tools = MarsSimulationTools()
    h = {"g0": 1.985, "delta_g": 0.02, "D": 180e6, "E": 10e6,
         "J": 1e8, "gauss": 1e-3, "lorentz": 1e-3}
    p = {"p_s": 0.0, "p_x": 0.5, "p_y": 1.0, "p_z": 0.0}
    field_axis = {"start": 0.325, "stop": 0.370, "points": 101}
    one = tools.simulate_moderate_j_1d(
        hamiltonian=h, populations=p, field_axis=field_axis,
        settings={"mesh": 25, "freq": 9.8e9, "temperature": 80, "device": "cpu"})
    ctx = {**p, "tau_outer": 50.0, "tau_inner": 0.5,
           "tau_out_s": 0.01, "tau_out_x": 5.0, "tau_out_y": 1.0, "tau_out_z": 10.0}
    two = tools.simulate_moderate_j_2d(
        spin_model=one["spin_model"], context=ctx, field_axis=field_axis,
        time_axis={"start": 0.0, "stop": 5e-6, "points": 31}, rise_time_us=0.2)
    full = tools.simulate_moderate_j_2d_full(
        hamiltonian={**h, "J": 2e8}, context=ctx, field_axis=field_axis,
        time_axis={"start": 0.0, "stop": 5e-6, "points": 31}, rise_time_us=0.2,
        settings={"mesh": 25, "device": "cpu"})
    print(json.dumps({"one": one["artifacts"], "two": two["artifacts"],
                      "full": full["artifacts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
