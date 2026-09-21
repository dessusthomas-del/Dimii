"""DIMENSIONNEMENT GÉO V47 — pv_battery_v47.py"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from solar_thermal_v47 import irradiance_tilted_plane, solar_position_8760, synthetic_solar_8760

@dataclass
class PVSystemParams:
    peak_kw: float = 10.0
    tilt_deg: float = 30.0
    azimuth_deg: float = 0.0
    latitude_deg: float = 45.75
    longitude_deg: float = 4.85
    gamma_pv: float = -0.0040
    noct_c: float = 45.0
    eta_inverter: float = 0.97
    eta_cabling: float = 0.98
    eta_soiling: float = 0.97
    eta_mismatch: float = 0.98
    degradation_rate_per_year: float = 0.005
    @property
    def system_efficiency(self):
        return self.eta_inverter*self.eta_cabling*self.eta_soiling*self.eta_mismatch

@dataclass
class BatteryParams:
    capacity_kwh: float = 10.0
    soc_min_kwh: float = 1.0
    soc_max_kwh: float = 9.5
    soc_initial_kwh: float = 5.0
    power_charge_kw: float = 5.0
    power_discharge_kw: float = 5.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    self_discharge_per_hour: float = 0.0001
    capex_eur_kwh: float = 600.0
    lifetime_cycles: int = 4000
    @property
    def usable_capacity_kwh(self): return self.soc_max_kwh-self.soc_min_kwh

class PVSimulator:
    def __init__(self, params):
        self.p = params
        self._elev, self._azim = solar_position_8760(params.latitude_deg, params.longitude_deg)

    def simulate_8760(self, ghi, dhi, t_ext_c, year=0):
        p = self.p; n = 8760
        ghi_a = np.array(ghi[:n], dtype=float)
        dhi_a = np.array(dhi[:n], dtype=float)
        t_ext = np.array(t_ext_c[:n], dtype=float)
        g_incl = irradiance_tilted_plane(ghi_a, dhi_a, self._elev, self._azim, p.tilt_deg, p.azimuth_deg)
        t_cell = t_ext+g_incl*(p.noct_c-20.0)/800.0
        eta_temp = 1.0+p.gamma_pv*(t_cell-25.0)
        eta_temp = np.clip(eta_temp, 0.5, 1.1)
        degradation = (1.0-p.degradation_rate_per_year)**year
        p_dc = p.peak_kw*(g_incl/1000.0)*eta_temp*degradation
        p_dc = np.maximum(0.0, p_dc)
        p_ac = p_dc*p.system_efficiency
        p_ac = np.where(g_incl<10.0, 0.0, p_ac)
        annual_yield = float(np.sum(p_ac))
        sp_yield = annual_yield/max(p.peak_kw,0.001)
        h_irr = float(np.sum(g_incl/1000.0))
        pr = annual_yield/max(p.peak_kw*h_irr,1e-6)
        return {
            "p_pv_kw": p_ac, "p_pv_dc_kw": p_dc, "t_cell_c": t_cell,
            "eta_pv_h": eta_temp, "g_incl_wm2": g_incl,
            "annual_yield_kwh": round(annual_yield,1),
            "specific_yield_kwh_kw": round(sp_yield,0),
            "performance_ratio": round(pr,3), "status": "estimatif",
        }

class PVBatterySimulator:
    def __init__(self, pv_params, battery_params):
        self.pv = PVSimulator(pv_params)
        self.bat = battery_params

    def simulate_8760(self, ghi, dhi, t_ext_c, p_load_kw, year=0, p_ev_charge_kw=None):
        bat = self.bat; n = 8760
        pv_res = self.pv.simulate_8760(ghi, dhi, t_ext_c, year)
        p_pv = pv_res["p_pv_kw"]
        p_load = np.array(p_load_kw[:n], dtype=float)
        if p_ev_charge_kw is not None:
            p_load = p_load+np.array(p_ev_charge_kw[:n], dtype=float)
        p_self = np.zeros(n); p_inject = np.zeros(n)
        p_grid = np.zeros(n); p_bat_flow = np.zeros(n)
        soc = np.zeros(n+1); soc[0] = bat.soc_initial_kwh; bat_throughput = 0.0
        for h in range(n):
            soc[h] *= (1.0-bat.self_discharge_per_hour)
            soc[h] = max(bat.soc_min_kwh, soc[h])
            pv = p_pv[h]; load = p_load[h]; balance = pv-load
            if balance >= 0:
                p_self[h] = load; excess = balance
                soc_available = bat.soc_max_kwh-soc[h]
                charge_possible = min(excess, bat.power_charge_kw, soc_available/bat.eta_charge)
                charge_possible = max(0.0, charge_possible)
                energy_stored = charge_possible*bat.eta_charge
                soc[h+1] = soc[h]+energy_stored; p_bat_flow[h] = charge_possible
                p_inject[h] = max(0.0, excess-charge_possible); bat_throughput += charge_possible
            else:
                deficit = -balance; p_self[h] = pv
                soc_avail = soc[h]-bat.soc_min_kwh
                disch_possible = min(deficit, bat.power_discharge_kw, soc_avail*bat.eta_discharge)
                disch_possible = max(0.0, disch_possible)
                energy_drawn = disch_possible/bat.eta_discharge
                soc[h+1] = soc[h]-energy_drawn; p_bat_flow[h] = -disch_possible
                p_grid[h] = max(0.0, deficit-disch_possible); bat_throughput += disch_possible
        soc_out = soc[:n]
        annual_pv = float(np.sum(p_pv)); annual_self = float(np.sum(p_self))
        annual_inject = float(np.sum(p_inject)); annual_grid = float(np.sum(p_grid))
        annual_load = float(np.sum(p_load))
        bat_cycles = bat_throughput/max(bat.usable_capacity_kwh,0.001)
        return {
            "p_pv_kw": p_pv, "p_self_consumed_kw": p_self, "p_injected_kw": p_inject,
            "p_grid_kw": p_grid, "p_battery_kw": p_bat_flow, "soc_kwh": soc_out,
            "pv_results": pv_res,
            "annual_pv_kwh": round(annual_pv,1), "annual_self_kwh": round(annual_self,1),
            "annual_injected_kwh": round(annual_inject,1), "annual_grid_kwh": round(annual_grid,1),
            "annual_load_kwh": round(annual_load,1),
            "autoconsumption_rate": round(annual_self/max(annual_pv,1e-6),3),
            "self_sufficiency": round(annual_self/max(annual_load,1e-6),3),
            "battery_cycles": round(bat_cycles,0),
            "battery_throughput_kwh": round(bat_throughput,1),
            "status": "estimatif",
        }

def economics_pv_battery(pv_params, bat_params, results,
                          electricity_buy_eur_kwh=0.2516,
                          electricity_sell_eur_kwh=0.10,
                          capex_pv_eur_kwp=1200.0):
    e_self = results["annual_self_kwh"]
    e_inject = results["annual_injected_kwh"]
    capex_pv = pv_params.peak_kw*capex_pv_eur_kwp
    capex_bat = bat_params.capacity_kwh*bat_params.capex_eur_kwh
    capex_total = capex_pv+capex_bat
    savings_self = e_self*electricity_buy_eur_kwh
    revenue_inject = e_inject*electricity_sell_eur_kwh
    net_annual = savings_self+revenue_inject
    trs = capex_total/max(net_annual,1.0)
    return {
        "capex_pv_eur": round(capex_pv,0), "capex_battery_eur": round(capex_bat,0),
        "capex_total_eur": round(capex_total,0),
        "annual_savings_eur": round(savings_self,0),
        "annual_revenue_eur": round(revenue_inject,0),
        "net_annual_benefit_eur": round(net_annual,0),
        "trs_years": round(trs,1), "status": "estimatif",
    }
