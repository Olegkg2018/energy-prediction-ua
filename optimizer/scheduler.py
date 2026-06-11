import numpy as np
from datetime import datetime, timedelta
import pulp

class BatteryOptimizer:
    BATTERY_CAPACITY = 4
    MAX_CHARGE_POWER = 1
    MAX_DISCHARGE_POWER = 1
    CHARGE_EFFICIENCY = 0.92
    DISCHARGE_EFFICIENCY = 0.92
    MAX_CYCLES_PER_DAY = 2.0
    TARIFF_TRANSMISSION = 150
    TARIFF_DISPATCH = 50
    INITIAL_SOC = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            key = k.upper()
            if hasattr(self, key):
                setattr(self, key, v)
            elif k == 'initial_soc':
                self.INITIAL_SOC = v

    def optimize(self, prices, factors=None):
        n = len(prices)
        prob = pulp.LpProblem("Battery_Optimization", pulp.LpMaximize)

        charge = pulp.LpVariable.dicts("charge", range(n), lowBound=0, upBound=self.MAX_CHARGE_POWER)
        discharge = pulp.LpVariable.dicts("discharge", range(n), lowBound=0, upBound=self.MAX_DISCHARGE_POWER)
        soc = pulp.LpVariable.dicts("soc", range(n + 1), lowBound=0, upBound=self.BATTERY_CAPACITY)

        soc_pct = self.INITIAL_SOC
        if soc_pct > 1:
            soc_pct = soc_pct / 100.0
        if soc_pct > 1:
            soc_pct = 1.0
        if soc_pct < 0:
            soc_pct = 0
        initial = soc_pct * self.BATTERY_CAPACITY
        prob += soc[0] == initial

        for t in range(n):
            prob += soc[t + 1] == soc[t] + charge[t] * self.CHARGE_EFFICIENCY - discharge[t] / self.DISCHARGE_EFFICIENCY
            prob += soc[t + 1] >= 0
            prob += soc[t + 1] <= self.BATTERY_CAPACITY

        total_cycles = pulp.lpSum([charge[t] for t in range(n)]) * self.CHARGE_EFFICIENCY / self.BATTERY_CAPACITY
        prob += total_cycles <= self.MAX_CYCLES_PER_DAY

        effective_prices = list(prices)
        if factors:
            for t in range(n):
                if factors.get('nuclear_outage') and t < 24:
                    effective_prices[t] *= 1.3
                if factors.get('missile_risk') and factors['missile_risk'] > 0:
                    risk_hours = (
                        [t for t in range(23, 24)] +
                        [t for t in range(0, 6)]
                    )
                    if t in risk_hours and factors['missile_risk'] > 0.5:
                        effective_prices[t] *= 1.5

        revenue = pulp.lpSum([
            discharge[t] * (effective_prices[t] - self.TARIFF_TRANSMISSION - self.TARIFF_DISPATCH) -
            charge[t] * effective_prices[t]
            for t in range(n)
        ])
        prob += revenue

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        status = pulp.LpStatus[prob.status]
        if status != 'Optimal':
            return None, status

        schedule = []
        for t in range(n):
            ch = pulp.value(charge[t]) or 0
            dc = pulp.value(discharge[t]) or 0
            soc_val = pulp.value(soc[t + 1]) or 0
            net = dc - ch
            action = 'charge' if ch > 0.05 else ('discharge' if dc > 0.05 else 'idle')
            schedule.append({
                'hour': t + 1,
                'charge_mw': round(ch, 2),
                'discharge_mw': round(dc, 2),
                'net_mw': round(net, 2),
                'soc_mwh': round(soc_val, 2),
                'soc_pct': round(soc_val / self.BATTERY_CAPACITY * 100, 1) if self.BATTERY_CAPACITY > 0 else 0,
                'price_uah': round(effective_prices[t], 2),
                'action': action
            })

        total_revenue = pulp.value(revenue) or 0
        total_charge = sum(s['charge_mw'] for s in schedule)
        total_discharge = sum(s['discharge_mw'] for s in schedule)

        return {
            'schedule': schedule,
            'total_revenue': round(total_revenue, 2),
            'total_charge_mwh': round(total_charge * self.CHARGE_EFFICIENCY, 2),
            'total_discharge_mwh': round(total_discharge / self.DISCHARGE_EFFICIENCY, 2),
            'cycles_used': round(total_charge * self.CHARGE_EFFICIENCY / self.BATTERY_CAPACITY, 2),
            'status': status,
            'profit_per_mwh': round(total_revenue / (total_discharge / self.DISCHARGE_EFFICIENCY), 2) if total_discharge > 0 else 0,
            'battery_capacity_mwh': self.BATTERY_CAPACITY,
            'battery_power_mw': self.MAX_CHARGE_POWER,
            'initial_soc_pct': self.INITIAL_SOC
        }, status


class SimpleBatteryOptimizer:
    def __init__(self, **kwargs):
        self.capacity = 4
        self.power = 1
        self.initial_soc = 0
        for k, v in kwargs.items():
            if k == 'capacity':
                self.capacity = v
            elif k == 'power':
                self.power = v
            elif k == 'initial_soc':
                self.initial_soc = v

    def optimize(self, prices, factors=None):
        n = len(prices)
        effective_prices = list(prices)
        if factors:
            for t in range(n):
                if factors.get('nuclear_outage') and t < 24:
                    effective_prices[t] *= 1.3
                if factors.get('missile_risk') and factors['missile_risk'] > 0:
                    risk_hours = list(range(23, 24)) + list(range(0, 6))
                    if t in risk_hours and factors['missile_risk'] > 0.5:
                        effective_prices[t] *= 1.5

        sorted_idx = np.argsort(effective_prices)
        charge_hours = sorted_idx[:6]
        discharge_hours = sorted_idx[-6:]

        overlap = set(charge_hours) & set(discharge_hours)
        charge_hours = [h for h in charge_hours if h not in overlap][:4]
        discharge_hours = [h for h in discharge_hours if h not in overlap][:4]

        soc_pct = self.initial_soc
        if soc_pct > 1:
            soc_pct = soc_pct / 100.0
        if soc_pct > 1:
            soc_pct = 1.0
        if soc_pct < 0:
            soc_pct = 0
        schedule = []
        soc_mwh = soc_pct * self.capacity
        for t in range(n):
            action = 'idle'
            ch = 0
            dc = 0
            if t in charge_hours and soc_mwh < self.capacity:
                ch = min(self.power, self.capacity - soc_mwh)
                soc_mwh += ch * 0.92
                action = 'charge'
            elif t in discharge_hours and soc_mwh > 0:
                dc = min(self.power, soc_mwh / 0.92)
                soc_mwh -= dc / 0.92
                action = 'discharge'
            schedule.append({
                'hour': t + 1,
                'action': action,
                'price_uah': round(effective_prices[t], 2),
                'charge_mw': round(ch, 2),
                'discharge_mw': round(dc, 2),
                'net_mw': round(dc - ch, 2),
                'soc_mwh': round(soc_mwh, 2),
                'soc_pct': round(soc_mwh / self.capacity * 100, 1) if self.capacity > 0 else 0,
            })

        total_charge = sum(s['charge_mw'] for s in schedule)
        total_discharge = sum(s['discharge_mw'] for s in schedule)

        return {
            'schedule': schedule,
            'total_revenue': 0,
            'total_charge_mwh': round(total_charge * 0.92, 2),
            'total_discharge_mwh': round(total_discharge / 0.92, 2),
            'cycles_used': round(total_charge * 0.92 / self.capacity, 2),
            'status': 'Optimal (simple)',
            'profit_per_mwh': 0,
            'battery_capacity_mwh': self.capacity,
            'battery_power_mw': self.power,
            'initial_soc_pct': self.initial_soc,
            'note': 'Simple sort method - install PuLP for full LP optimization'
        }, 'Optimal'


def create_optimizer(use_lp=True, **kwargs):
    if use_lp:
        try:
            return BatteryOptimizer(**kwargs)
        except Exception:
            return SimpleBatteryOptimizer(**kwargs)
    return SimpleBatteryOptimizer(**kwargs)
