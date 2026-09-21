"""DIMENSIONNEMENT GÉO V47 — solar_thermal_v47.py"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

def solar_position_8760(latitude_deg, longitude_deg=0.0, tz_offset_h=1.0):
    lat = math.radians(latitude_deg)
    hours = np.arange(8760, dtype=float)
    day = (hours//24).astype(int)
    decl = np.radians(23.45*np.sin(np.radians(360.0/365.0*(day-81))))
    B = np.radians(360.0*(day-81)/364.0)
    EoT_min = 9.87*np.sin(2*B)-7.53*np.cos(B)-1.5*np.sin(B)
    hour_local = hours%24.0
    solar_time = hour_local+longitude_deg/15.0-tz_offset_h+EoT_min/60.0
    omega = np.radians(15.0*(solar_time-12.0))
    sin_elev = np.sin(lat)*np.sin(decl)+np.cos(lat)*np.cos(decl)*np.cos(omega)
    elevation = np.arcsin(np.clip(sin_elev,-1.0,1.0))
    cos_az = ((np.sin(decl)*np.cos(lat)-np.cos(decl)*np.sin(lat)*np.cos(omega))
               /np.cos(elevation+1e-9))
    azimuth = np.sign(omega)*np.arccos(np.clip(cos_az,-1.0,1.0))
    return elevation, azimuth

def irradiance_tilted_plane(ghi, dhi, elevation_rad, azimuth_rad,
                             tilt_deg, panel_azimuth_deg=0.0, albedo=0.20):
    beta = math.radians(tilt_deg)
    gamma_p = math.radians(panel_azimuth_deg)
    sin_elev = np.sin(elevation_rad)
    with np.errstate(divide='ignore', invalid='ignore'):
        dni = np.where(sin_elev>0.01, (ghi-dhi)/sin_elev, 0.0)
    dni = np.clip(dni, 0, 1400)
    cos_theta = (np.sin(elevation_rad)*math.cos(beta)
                 +np.cos(elevation_rad)*math.sin(beta)*np.cos(azimuth_rad-gamma_p))
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    g_dir = dni*cos_theta
    g_diff = dhi*(1.0+math.cos(beta))/2.0
    g_refl = ghi*albedo*(1.0-math.cos(beta))/2.0
    return np.maximum(0.0, g_dir+g_diff+g_refl)

@dataclass
class SolarThermalParams:
    area_m2: float = 20.0
    eta0: float = 0.78
    a1_wpm2k: float = 3.5
    a2_wpm2k2: float = 0.015
    t_in_c: float = 45.0
    t_out_c: float = 65.0
    tilt_deg: float = 35.0
    azimuth_deg: float = 0.0
    iam_b0: float = 0.05
    g_min_wm2: float = 50.0
    latitude_deg: float = 45.75
    longitude_deg: float = 4.85
    @property
    def t_mean_c(self): return (self.t_in_c+self.t_out_c)/2.0

@dataclass
class PVTParams:
    area_m2: float = 20.0
    eta_pv_ref: float = 0.16
    gamma_pv: float = -0.0040
    noct_c: float = 46.0
    eta_inverter: float = 0.96
    eta0_th: float = 0.45
    a1_th_wpm2k: float = 8.0
    a2_th_wpm2k2: float = 0.02
    t_in_c: float = 25.0
    t_out_c: float = 45.0
    tilt_deg: float = 35.0
    azimuth_deg: float = 0.0
    latitude_deg: float = 45.75
    longitude_deg: float = 4.85
    g_min_wm2: float = 50.0
    @property
    def t_mean_c(self): return (self.t_in_c+self.t_out_c)/2.0

class SolarThermalSimulator:
    def __init__(self, params):
        self.p = params
        self._elev, self._azim = solar_position_8760(params.latitude_deg, params.longitude_deg)

    def simulate_8760(self, ghi, dhi, t_ext_c, demand_kw=None):
        p = self.p; n = 8760
        ghi_a = np.array(ghi[:n], dtype=float)
        dhi_a = np.array(dhi[:n], dtype=float)
        t_ext = np.array(t_ext_c[:n], dtype=float)
        g_incl = irradiance_tilted_plane(ghi_a, dhi_a, self._elev, self._azim, p.tilt_deg, p.azimuth_deg)
        tm = p.t_mean_c
        with np.errstate(divide='ignore', invalid='ignore'):
            dt_red = np.where(g_incl>p.g_min_wm2, (tm-t_ext)/g_incl, 0.0)
            eta = p.eta0 - p.a1_wpm2k*dt_red - p.a2_wpm2k2*g_incl*dt_red**2
        eta = np.where(g_incl<p.g_min_wm2, 0.0, eta)
        eta = np.maximum(0.0, eta)
        q_th = eta*g_incl*p.area_m2/1000.0
        if demand_kw is not None:
            dem = np.array(demand_kw[:n], dtype=float)
            q_useful = np.minimum(q_th, dem)
            solar_fraction = float(np.sum(q_useful))/max(float(np.sum(dem)),1e-6)
        else:
            q_useful = q_th.copy(); solar_fraction = None
        annual_yield = float(np.sum(q_useful))
        return {
            "q_th_kw": q_th, "q_th_useful_kw": q_useful, "eta_h": eta, "g_incl_wm2": g_incl,
            "annual_yield_kwh": round(annual_yield,1),
            "specific_yield_kwh_m2": round(annual_yield/max(p.area_m2,0.1),1),
            "solar_fraction": round(solar_fraction,3) if solar_fraction is not None else None,
            "status": "estimatif",
        }

class PVTSimulator:
    def __init__(self, params):
        self.p = params
        self._elev, self._azim = solar_position_8760(params.latitude_deg, params.longitude_deg)

    def simulate_8760(self, ghi, dhi, t_ext_c, heat_demand_kw=None):
        p = self.p; n = 8760
        ghi_a = np.array(ghi[:n], dtype=float)
        dhi_a = np.array(dhi[:n], dtype=float)
        t_ext = np.array(t_ext_c[:n], dtype=float)
        g_incl = irradiance_tilted_plane(ghi_a, dhi_a, self._elev, self._azim, p.tilt_deg, p.azimuth_deg)
        t_cell = t_ext+g_incl*(p.noct_c-20.0)/800.0
        eta_pv = p.eta_pv_ref*(1.0+p.gamma_pv*(t_cell-25.0))
        eta_pv = np.clip(eta_pv, 0.0, p.eta_pv_ref*1.1)
        p_elec_dc = eta_pv*g_incl*p.area_m2/1000.0
        p_elec = p_elec_dc*p.eta_inverter
        p_elec = np.where(g_incl<p.g_min_wm2, 0.0, p_elec)
        tm_th = p.t_mean_c
        with np.errstate(divide='ignore', invalid='ignore'):
            dt_red = np.where(g_incl>p.g_min_wm2, (tm_th-t_ext)/g_incl, 0.0)
            eta_th = p.eta0_th-p.a1_th_wpm2k*dt_red-p.a2_th_wpm2k2*g_incl*dt_red**2
        eta_th = np.where(g_incl<p.g_min_wm2, 0.0, eta_th)
        eta_th = np.maximum(0.0, eta_th)
        total_eta = eta_pv+eta_th
        mask_exceed = total_eta>0.90
        eta_th = np.where(mask_exceed, eta_th*0.90/total_eta, eta_th)
        q_th = eta_th*g_incl*p.area_m2/1000.0
        if heat_demand_kw is not None:
            dem = np.array(heat_demand_kw[:n], dtype=float)
            q_useful = np.minimum(q_th, dem)
            solar_th_fraction = float(np.sum(q_useful))/max(float(np.sum(dem)),1e-6)
        else:
            q_useful = q_th.copy(); solar_th_fraction = None
        return {
            "p_elec_kw": p_elec, "q_th_kw": q_th, "q_th_useful_kw": q_useful,
            "eta_pv_h": eta_pv, "eta_th_h": eta_th, "t_cell_c": t_cell, "g_incl_wm2": g_incl,
            "annual_elec_kwh": round(float(np.sum(p_elec)),1),
            "annual_th_kwh": round(float(np.sum(q_useful)),1),
            "solar_th_fraction": round(solar_th_fraction,3) if solar_th_fraction is not None else None,
            "status": "estimatif",
        }

def synthetic_solar_8760(latitude_deg=45.75, annual_ghi_kwh_m2=1200.0, diffuse_fraction=0.40):
    hours = np.arange(8760, dtype=float)
    day = hours//24.0; hour_of_day = hours%24.0
    lat = math.radians(latitude_deg)
    decl = np.radians(23.45*np.sin(np.radians(360.0/365.0*(day-81))))
    omega = np.radians(15.0*(hour_of_day-12.0))
    sin_elev = np.sin(lat)*np.sin(decl)+np.cos(lat)*np.cos(decl)*np.cos(omega)
    elev_pos = np.maximum(0.0, sin_elev)
    ec_factor = 1.0+0.033*np.cos(np.radians(360.0*day/365.0))
    i0 = 1361.0*ec_factor
    ghi_clear = i0*elev_pos*0.75
    ghi = ghi_clear
    annual_current = float(np.sum(ghi))/1000.0
    if annual_current>0: ghi *= annual_ghi_kwh_m2/annual_current
    dhi = ghi*diffuse_fraction
    return ghi, dhi
