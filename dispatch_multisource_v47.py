"""DIMENSIONNEMENT GÉO V47 — dispatch_multisource_v47.py"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from gshp_geofield_v47 import GSHPSimulator, GSHPFieldConfig
from ashp_v47 import ASHPSimulator, ASHPParams
from solar_thermal_v47 import SolarThermalSimulator, SolarThermalParams, PVTSimulator, PVTParams
from pv_battery_v47 import PVBatterySimulator, PVSystemParams, BatteryParams

@dataclass
class ThermalStorageParams:
    volume_m3: float = 1.0
    t_min_c: float = 45.0
    t_max_c: float = 80.0
    t_initial_c: float = 55.0
    t_ambient_c: float = 18.0
    u_value_wpm2k: float = 0.5
    surface_m2: float = 3.0
    cp_kj_kgk: float = 4.18
    rho_kg_m3: float = 985.0
    max_charge_kw: float = 20.0
    max_discharge_kw: float = 20.0
    @property
    def capacity_kwh(self):
        m = self.volume_m3*self.rho_kg_m3
        return m*self.cp_kj_kgk*(self.t_max_c-self.t_min_c)/3600.0
    @property
    def losses_kw_per_k(self):
        return self.u_value_wpm2k*self.surface_m2/1000.0

class ThermalStorageSimulator:
    def __init__(self, params):
        self.p = params
    def step(self, t_current_c, q_in_kw, q_out_kw, dt_h=1.0):
        p = self.p; m = p.volume_m3*p.rho_kg_m3
        q_loss = max(0.0, p.losses_kw_per_k*(t_current_c-p.t_ambient_c))
        q_net_kw = q_in_kw-q_out_kw-q_loss
        delta_t = q_net_kw*3600.0*dt_h/(m*p.cp_kj_kgk)
        t_new = t_current_c+delta_t
        if t_new > p.t_max_c: t_new = p.t_max_c
        if t_new < p.t_min_c:
            q_available = m*p.cp_kj_kgk*(t_current_c-p.t_min_c)/(3600.0*dt_h)
            q_out_kw = min(q_out_kw, max(0.0, q_available+q_in_kw-q_loss))
            q_net_kw = q_in_kw-q_out_kw-q_loss
            delta_t = q_net_kw*3600.0*dt_h/(m*p.cp_kj_kgk)
            t_new = max(p.t_min_c, t_current_c+delta_t)
        return t_new, q_out_kw

@dataclass
class MultiSourceConfig:
    gshp: Optional[GSHPFieldConfig] = None
    ashp: Optional[ASHPParams] = None
    solar_thermal: Optional[SolarThermalParams] = None
    pvt: Optional[PVTParams] = None
    thermal_storage: Optional[ThermalStorageParams] = None
    pv: Optional[PVSystemParams] = None
    battery: Optional[BatteryParams] = None
    priority_heat: list = field(default_factory=lambda: ["solar_thermal","pvt","thermal_storage","gshp","ashp","backup_elec"])
    priority_cool: list = field(default_factory=lambda: ["gshp_passive","gshp","ashp"])
    backup_elec_kw: float = 10.0

    @classmethod
    def from_dict(cls, d):
        cfg = cls()
        if d.get("gshp", {}).get("enabled"):
            cfg.gshp = GSHPFieldConfig.from_dict(d["gshp"])
        if d.get("ashp", {}).get("enabled"):
            cfg.ashp = ASHPParams(**{k: v for k, v in d["ashp"].items() if k != "enabled"})
        if d.get("solar_thermal", {}).get("enabled"):
            cfg.solar_thermal = SolarThermalParams(**{k: v for k, v in d["solar_thermal"].items() if k != "enabled"})
        if d.get("pvt", {}).get("enabled"):
            cfg.pvt = PVTParams(**{k: v for k, v in d["pvt"].items() if k != "enabled"})
        if d.get("thermal_storage", {}).get("enabled"):
            cfg.thermal_storage = ThermalStorageParams(**{k: v for k, v in d["thermal_storage"].items() if k != "enabled"})
        if d.get("pv", {}).get("enabled"):
            cfg.pv = PVSystemParams(**{k: v for k, v in d["pv"].items() if k != "enabled"})
        if d.get("battery", {}).get("enabled"):
            cfg.battery = BatteryParams(**{k: v for k, v in d["battery"].items() if k != "enabled"})
        cfg.backup_elec_kw = d.get("backup_elec_kw", 10.0)
        return cfg

def presimulate_sources(config, ghi, dhi, t_ext_c, q_heat_kw, q_cool_kw, q_dhw_kw):
    n = 8760; sources = {}
    q_heat_total = q_heat_kw+q_dhw_kw
    if config.solar_thermal is not None:
        st_sim = SolarThermalSimulator(config.solar_thermal)
        sources["solar_thermal"] = st_sim.simulate_8760(ghi, dhi, t_ext_c, demand_kw=q_heat_total)
    if config.pvt is not None:
        pvt_sim = PVTSimulator(config.pvt)
        sources["pvt"] = pvt_sim.simulate_8760(ghi, dhi, t_ext_c, heat_demand_kw=q_heat_total)
    if config.ashp is not None:
        ashp_sim = ASHPSimulator(config.ashp)
        sources["ashp"] = ashp_sim.simulate_8760(q_heat_total, q_cool_kw, t_ext_c)
    if config.gshp is not None:
        gshp_sim = GSHPSimulator(config.gshp)
        sources["gshp"] = gshp_sim.simulate_8760(q_heat_total, q_cool_kw)
    return sources

class MultiSourceDispatcher:
    def __init__(self, config):
        self.config = config

    def dispatch_8760(self, ghi, dhi, t_ext_c, q_heat_kw, q_cool_kw, q_dhw_kw, p_elec_load_kw, year=0):
        cfg = self.config; n = 8760
        q_heat = np.array(q_heat_kw[:n], dtype=float)
        q_cool = np.array(q_cool_kw[:n], dtype=float)
        q_dhw = np.array(q_dhw_kw[:n], dtype=float)
        q_demand_th = q_heat+q_dhw
        src = presimulate_sources(cfg, ghi, dhi, t_ext_c, q_heat, q_cool, q_dhw)
        out = {
            "q_solar_th_kw": np.zeros(n), "q_pvt_th_kw": np.zeros(n),
            "q_gshp_heat_kw": np.zeros(n), "q_ashp_heat_kw": np.zeros(n),
            "q_storage_out_kw": np.zeros(n), "q_storage_in_kw": np.zeros(n),
            "q_backup_kw": np.zeros(n), "q_gshp_cool_kw": np.zeros(n),
            "q_ashp_cool_kw": np.zeros(n), "unmet_heat_kw": np.zeros(n),
            "unmet_cool_kw": np.zeros(n), "p_elec_gshp_kw": np.zeros(n),
            "p_elec_ashp_kw": np.zeros(n), "p_elec_backup_kw": np.zeros(n),
            "cop_gshp_h": np.zeros(n), "cop_ashp_h": np.zeros(n),
            "t_source_gshp_c": np.zeros(n), "t_storage_c": np.zeros(n),
        }
        if cfg.thermal_storage:
            storage = ThermalStorageSimulator(cfg.thermal_storage)
            t_storage = cfg.thermal_storage.t_initial_c
        else:
            storage = None; t_storage = 0.0
        for h in range(n):
            remaining_heat = q_demand_th[h]; remaining_cool = q_cool[h]
            if "solar_thermal" in src and remaining_heat>0:
                q_av = src["solar_thermal"]["q_th_useful_kw"][h]
                q_used = min(remaining_heat, q_av)
                out["q_solar_th_kw"][h] = q_used; remaining_heat -= q_used
                q_excess_sol = q_av-q_used
                if storage and q_excess_sol>0.01:
                    out["q_storage_in_kw"][h] += min(q_excess_sol, cfg.thermal_storage.max_charge_kw)
            if "pvt" in src and remaining_heat>0:
                q_av = src["pvt"]["q_th_useful_kw"][h]
                q_used = min(remaining_heat, q_av)
                out["q_pvt_th_kw"][h] = q_used; remaining_heat -= q_used
                q_excess_pvt = q_av-q_used
                if storage and q_excess_pvt>0.01:
                    out["q_storage_in_kw"][h] += min(q_excess_pvt, cfg.thermal_storage.max_charge_kw)
            if storage and remaining_heat>0:
                if t_storage > cfg.thermal_storage.t_min_c+1.0:
                    q_av = min(remaining_heat, cfg.thermal_storage.max_discharge_kw)
                    t_storage, q_deliv = storage.step(t_storage, out["q_storage_in_kw"][h], q_av)
                    out["q_storage_out_kw"][h] = q_deliv; remaining_heat -= q_deliv
                else:
                    t_storage, _ = storage.step(t_storage, out["q_storage_in_kw"][h], 0.0)
                out["t_storage_c"][h] = t_storage
            if cfg.gshp and remaining_heat>0:
                q_gshp_max = src["gshp"]["q_heat_delivered_kw"][h]
                q_used = min(remaining_heat, q_gshp_max)
                out["q_gshp_heat_kw"][h] = q_used
                out["p_elec_gshp_kw"][h] = (q_used/max(src["gshp"]["cop_h"][h],0.1) if src["gshp"]["cop_h"][h]>0.1 else q_used)
                out["cop_gshp_h"][h] = src["gshp"]["cop_h"][h]
                out["t_source_gshp_c"][h] = src["gshp"]["t_source_c"][h]
                remaining_heat -= q_used
            if cfg.ashp and remaining_heat>0:
                q_ashp_max = src["ashp"]["q_heat_delivered_kw"][h]
                q_used = min(remaining_heat, q_ashp_max)
                out["q_ashp_heat_kw"][h] = q_used
                cop_a = src["ashp"]["cop_h"][h]
                out["p_elec_ashp_kw"][h] = (q_used/max(cop_a,0.1) if cop_a>0.1 else q_used)
                out["cop_ashp_h"][h] = cop_a; remaining_heat -= q_used
            if remaining_heat>0:
                q_backup = min(remaining_heat, cfg.backup_elec_kw)
                out["q_backup_kw"][h] = q_backup; out["p_elec_backup_kw"][h] = q_backup; remaining_heat -= q_backup
            out["unmet_heat_kw"][h] = max(0.0, remaining_heat)
            if cfg.gshp and remaining_cool>0:
                q_cool_gshp = src["gshp"]["q_cool_delivered_kw"][h]
                q_used = min(remaining_cool, q_cool_gshp)
                out["q_gshp_cool_kw"][h] = q_used; remaining_cool -= q_used
            if cfg.ashp and remaining_cool>0:
                q_cool_ashp = src["ashp"]["q_cool_delivered_kw"][h]
                q_used = min(remaining_cool, q_cool_ashp)
                out["q_ashp_cool_kw"][h] = q_used
                out["p_elec_ashp_kw"][h] += (q_used/max(src["ashp"]["eer_h"][h],0.1) if src["ashp"]["eer_h"][h]>0.1 else q_used)
                remaining_cool -= q_used
            out["unmet_cool_kw"][h] = max(0.0, remaining_cool)
        p_elec_pac = out["p_elec_gshp_kw"]+out["p_elec_ashp_kw"]+out["p_elec_backup_kw"]
        p_elec_load_full = np.array(p_elec_load_kw[:n], dtype=float)+p_elec_pac
        if cfg.pv:
            bat_params = cfg.battery or BatteryParams(capacity_kwh=0.0, soc_max_kwh=0.0)
            pv_bat_sim = PVBatterySimulator(cfg.pv, bat_params)
            pv_bat_res = pv_bat_sim.simulate_8760(ghi, dhi, t_ext_c, p_elec_load_full, year=year)
        else:
            pv_bat_res = {
                "p_pv_kw": np.zeros(n), "p_self_consumed_kw": np.zeros(n),
                "p_injected_kw": np.zeros(n), "p_grid_kw": p_elec_load_full.copy(),
                "p_battery_kw": np.zeros(n), "soc_kwh": np.zeros(n),
                "annual_pv_kwh": 0.0, "annual_self_kwh": 0.0,
                "annual_injected_kwh": 0.0, "annual_grid_kwh": float(np.sum(p_elec_load_full)),
                "autoconsumption_rate": 0.0, "self_sufficiency": 0.0, "battery_cycles": 0.0,
            }
        out.update({
            "p_pv_kw": pv_bat_res["p_pv_kw"],
            "p_pvt_elec_kw": (src["pvt"]["p_elec_kw"] if "pvt" in src else np.zeros(n)),
            "p_self_consumed_kw": pv_bat_res["p_self_consumed_kw"],
            "p_injected_kw": pv_bat_res["p_injected_kw"],
            "p_grid_kw": pv_bat_res["p_grid_kw"],
            "p_battery_kw": pv_bat_res["p_battery_kw"],
            "soc_battery_kwh": pv_bat_res["soc_kwh"],
            "p_elec_total_kw": p_elec_load_full,
        })
        def s(key): return round(float(np.sum(out[key])),1)
        annual = {
            "heat_demand_kwh": round(float(np.sum(q_demand_th)),1),
            "cool_demand_kwh": round(float(np.sum(q_cool)),1),
            "solar_th_kwh": s("q_solar_th_kw"), "pvt_th_kwh": s("q_pvt_th_kw"),
            "gshp_heat_kwh": s("q_gshp_heat_kw"), "ashp_heat_kwh": s("q_ashp_heat_kw"),
            "storage_discharge_kwh": s("q_storage_out_kw"), "backup_kwh": s("q_backup_kw"),
            "gshp_cool_kwh": s("q_gshp_cool_kw"), "ashp_cool_kwh": s("q_ashp_cool_kw"),
            "unmet_heat_kwh": s("unmet_heat_kw"), "unmet_cool_kwh": s("unmet_cool_kw"),
            "elec_gshp_kwh": s("p_elec_gshp_kw"), "elec_ashp_kwh": s("p_elec_ashp_kw"),
            "elec_backup_kwh": s("p_elec_backup_kw"),
            "pv_yield_kwh": pv_bat_res["annual_pv_kwh"],
            "pv_self_kwh": pv_bat_res["annual_self_kwh"],
            "pv_injected_kwh": pv_bat_res["annual_injected_kwh"],
            "grid_kwh": pv_bat_res["annual_grid_kwh"],
            "battery_cycles": pv_bat_res["battery_cycles"],
        }
        e_elec_pac = annual["elec_gshp_kwh"]+annual["elec_ashp_kwh"]+annual["elec_backup_kwh"]
        annual["scop_system"] = round(annual["heat_demand_kwh"]/max(e_elec_pac,1.0),2)
        sources_used = []
        if annual["solar_th_kwh"]>0: sources_used.append("solar_thermal")
        if annual["pvt_th_kwh"]>0: sources_used.append("pvt")
        if annual["gshp_heat_kwh"]>0: sources_used.append("gshp")
        if annual["ashp_heat_kwh"]>0: sources_used.append("ashp")
        if annual["backup_kwh"]>0: sources_used.append("backup_elec")
        if annual["pv_yield_kwh"]>0: sources_used.append("pv")
        return {
            "hourly": out, "annual": annual, "sources_results": src,
            "pv_battery_results": pv_bat_res, "sources_active": sources_used, "status": "estimatif",
        }
