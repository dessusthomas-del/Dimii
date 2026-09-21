"""
DIMENSIONNEMENT GÉO V47 — ashp_v47.py
PAC aérothermique (Air Source Heat Pump)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

_REF_POINTS_HEAT_35 = [
    (-15.0, 1.80), (-7.0, 2.70), (2.0, 3.50),
    (7.0, 4.10), (12.0, 4.60), (20.0, 5.20),
]
_REF_POINTS_HEAT_55 = [
    (-15.0, 1.20), (-7.0, 1.80), (2.0, 2.50),
    (7.0, 2.90), (12.0, 3.20), (20.0, 3.80),
]
_REF_POINTS_POWER = [
    (-20.0, 0.70), (-15.0, 0.78), (-7.0, 0.88),
    (2.0, 0.95), (7.0, 1.00), (12.0, 1.05), (20.0, 1.10),
]

def _interp_table(x, table):
    xs = [p[0] for p in table]; ys = [p[1] for p in table]
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(len(xs)-1):
        if xs[i] <= x <= xs[i+1]:
            t = (x-xs[i])/(xs[i+1]-xs[i])
            return ys[i] + t*(ys[i+1]-ys[i])
    return ys[-1]

@dataclass
class ASHPParams:
    rated_heat_kw: float = 20.0
    rated_cool_kw: float = 15.0
    t_supply_heat_c: float = 35.0
    t_supply_cool_c: float = 7.0
    cop_b7w35: float = 4.1
    eer_b35w7: float = 3.5
    t_ext_heat_min_c: float = -20.0
    t_ext_heat_max_c: float = 20.0
    t_ext_cool_min_c: float = 10.0
    t_ext_cool_max_c: float = 46.0
    defrost_enabled: bool = True
    defrost_t_min_c: float = -5.0
    defrost_t_max_c: float = 5.0
    defrost_penalty: float = 0.12
    inverter: bool = True
    min_load_fraction: float = 0.20
    cop_t_correction: float = -0.05
    eer_t_correction: float = -0.05
    cop_custom_table: Optional[List[Tuple[float,float]]] = None
    eer_custom_table: Optional[List[Tuple[float,float]]] = None
    power_custom_table: Optional[List[Tuple[float,float]]] = None

    def cop_at(self, t_ext_c, t_supply_c=None):
        if t_supply_c is None: t_supply_c = self.t_supply_heat_c
        if self.cop_custom_table:
            cop_base = _interp_table(t_ext_c, self.cop_custom_table)
        elif abs(t_supply_c-55.0) < abs(t_supply_c-35.0):
            cop_base = _interp_table(t_ext_c, _REF_POINTS_HEAT_55)
            cop_base += self.cop_t_correction*(t_supply_c-55.0)
        else:
            cop_base = _interp_table(t_ext_c, _REF_POINTS_HEAT_35)
            cop_base += self.cop_t_correction*(t_supply_c-35.0)
        cop_en_b7 = _interp_table(7.0, _REF_POINTS_HEAT_35)
        return max(1.0, min(8.0, cop_base/max(cop_en_b7,0.1)*self.cop_b7w35))

    def power_available_kw(self, t_ext_c):
        table = self.power_custom_table or _REF_POINTS_POWER
        return self.rated_heat_kw * _interp_table(t_ext_c, table)

    def eer_at(self, t_ext_c, t_supply_c=None):
        if t_supply_c is None: t_supply_c = self.t_supply_cool_c
        if self.eer_custom_table:
            eer_base = _interp_table(t_ext_c, self.eer_custom_table)
        else:
            eer_base = self.eer_b35w7*(1.0-0.02*(t_ext_c-35.0))
            eer_base += self.eer_t_correction*(t_supply_c-7.0)
        return max(1.0, min(12.0, eer_base))

    def defrost_factor(self, t_ext_c):
        if not self.defrost_enabled: return 1.0
        t_min, t_max = self.defrost_t_min_c, self.defrost_t_max_c
        if t_ext_c <= t_min or t_ext_c >= t_max: return 1.0
        t_mid = (t_min+t_max)/2.0
        penalty = self.defrost_penalty*(1.0-abs(t_ext_c-t_mid)/abs(t_mid-t_min))
        return 1.0-penalty

class ASHPSimulator:
    def __init__(self, params):
        self.params = params

    def simulate_8760(self, q_heat_kw, q_cool_kw, t_ext_c):
        p = self.params; n = 8760
        q_heat = np.array(q_heat_kw[:n], dtype=float)
        q_cool = np.array(q_cool_kw[:n], dtype=float)
        t_ext = np.array(t_ext_c[:n], dtype=float)
        q_heat_del = np.zeros(n); q_cool_del = np.zeros(n)
        p_elec = np.zeros(n); cop_h = np.zeros(n)
        eer_h = np.zeros(n); dfrost = np.zeros(n)
        unmet_heat = np.zeros(n); unmet_cool = np.zeros(n)
        hours_limit_heat = 0; hours_limit_cool = 0
        for h in range(n):
            te = t_ext[h]
            if q_heat[h] > 0:
                if te < p.t_ext_heat_min_c:
                    unmet_heat[h] = q_heat[h]; hours_limit_heat += 1
                else:
                    cop = p.cop_at(te, p.t_supply_heat_c)
                    df = p.defrost_factor(te)
                    q_avail = p.power_available_kw(te)*df
                    q_del = min(q_heat[h], q_avail)
                    q_heat_del[h] = q_del; p_elec[h] += q_del/cop
                    cop_h[h] = cop; dfrost[h] = df
                    unmet_heat[h] = max(0.0, q_heat[h]-q_del)
            if q_cool[h] > 0:
                if te > p.t_ext_cool_max_c:
                    unmet_cool[h] = q_cool[h]; hours_limit_cool += 1
                elif te < p.t_ext_cool_min_c:
                    q_cool_del[h] = min(q_cool[h], p.rated_cool_kw); eer_h[h] = 99.0
                else:
                    eer = p.eer_at(te, p.t_supply_cool_c)
                    q_del = min(q_cool[h], p.rated_cool_kw)
                    q_cool_del[h] = q_del; p_elec[h] += q_del/eer; eer_h[h] = eer
                    unmet_cool[h] = max(0.0, q_cool[h]-q_del)
        e_heat = float(np.sum(q_heat_del))
        e_elec_heat = float(np.sum(p_elec*(q_heat_del>0).astype(float)))
        e_cool = float(np.sum(q_cool_del))
        e_elec_cool = float(np.sum(p_elec*(q_cool_del>0).astype(float)))
        return {
            "q_heat_delivered_kw": q_heat_del, "q_cool_delivered_kw": q_cool_del,
            "p_elec_kw": p_elec, "cop_h": cop_h, "eer_h": eer_h,
            "defrost_factor_h": dfrost, "unmet_heat_kw": unmet_heat,
            "unmet_cool_kw": unmet_cool,
            "annual_scop": round(e_heat/max(e_elec_heat,1e-6),2),
            "annual_seer": round(e_cool/max(e_elec_cool,1e-6),2),
            "hours_at_limit_heat": hours_limit_heat,
            "hours_at_limit_cool": hours_limit_cool,
            "annual_heat_kwh": round(e_heat,1),
            "annual_cool_kwh": round(e_cool,1),
            "annual_elec_kwh": round(float(np.sum(p_elec)),1),
            "status": "estimatif",
        }

def synthetic_t_ext(t_annual_mean_c=11.5, t_annual_amplitude_c=10.0, t_daily_amplitude_c=5.0):
    h = np.arange(8760, dtype=float)
    t_seasonal = t_annual_amplitude_c*np.cos(2*math.pi*(h-2880)/8760)
    t_daily = t_daily_amplitude_c*np.cos(2*math.pi*(h%24)/24)
    return t_annual_mean_c + t_seasonal + t_daily
