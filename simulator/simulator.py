import time
import random
import json
from datetime import datetime, timedelta


class PassengerGenerator:
    """乘客生成器类，用于管理各站点的乘客生成"""

    def __init__(self, stations, base_rates, time_patterns):
        """
        Args:
            stations (list): Station对象列表
            base_rates (dict): 各站点的基准到达率 {station_id: rate}
            time_patterns (dict): 时间段流量模式
        """
        self.stations = {s.id: s for s in stations}
        self.base_rates = base_rates
        self.time_patterns = time_patterns

    def generate_passengers(self, current_time):
        """根据当前时间生成乘客"""
        total_generated = 0

        # 应用时间段流量模式
        multiplier = 1.0
        for pattern in self.time_patterns.values():
            if pattern['start_time'] <= current_time <= pattern['end_time']:
                multiplier = pattern['multiplier']
                break

        # 为每个站点生成乘客
        for station_id, base_rate in self.base_rates.items():
            if station_id in self.stations:
                station = self.stations[station_id]
                # 计算实际到达率
                actual_rate = base_rate * multiplier
                # 生成泊松分布的乘客
                passengers = random.poisson(actual_rate)
                station.waiting_passengers += passengers
                total_generated += passengers

        return total_generated


class RailSystemSimulator:
    """（保持原有docstring不变）"""

    def __init__(self, line, trains, passenger_generator, config=None):  # 修改构造函数
        self.line = line
        self.trains = trains
        self.passenger_generator = passenger_generator  # 新增属性

        # 原有配置初始化代码...

    def _update_station_passengers(self, time_step):
        """更新站点乘客（新增方法）"""
        # 调用passenger_generator生成新乘客
        self.passenger_generator.generate_passengers(self.simulation_time)

        # 更新所有站点的等待乘客
        station_updates = []
        for station in self.line.stations:
            # 原有乘客增长逻辑（如果passenger_generator没有完全替代）
            station.waiting_passengers += random.poisson(station.passenger_arrival_rate * time_step / 60)
            station_updates.append({
                'station_id': station.id,
                'waiting_passengers': station.waiting_passengers
            })
        return station_updates

    def get_waiting_passengers(self, station_id):
        """获取指定站点的等待乘客（新增方法）"""
        station = next((s for s in self.line.stations if s.id == station_id), None)
        return station.waiting_passengers if station else 0

    def get_statistics(self):
        """获取统计信息（补充实现）"""
        self._update_statistics()  # 确保统计信息最新
        return {
            'total_passengers_served': self.statistics['passengers_served'],
            'average_waiting_time': self.statistics['average_waiting_time'],
            'total_energy_consumed': self.statistics['total_energy_consumed'],
            'train_utilization': self.statistics['train_utilization']
        }

    # 修改原有simulate_step方法
    def simulate_step(self):
        # 在原有代码前添加乘客生成
        self._update_station_passengers(self.config['time_step'])

        # 保持原有逻辑...

    # 在Line类添加以下方法


class Line:
    def get_line_statistics(self):
        """获取线路统计信息"""
        total_waiting = sum(s.waiting_passengers for s in self.stations)
        total_served = sum(s.total_passengers_served for s in self.stations)
        avg_wait = sum(s.total_waiting_time for s in self.stations) / total_served if total_served > 0 else 0

        return {
            'total_waiting_passengers': total_waiting,
            'total_passengers_served': total_served,
            'average_waiting_time': avg_wait
        }

    def reset_all_stations(self):
        """重置所有车站状态"""
        for s in self.stations:
            s.waiting_passengers = 0
            s.total_passengers_served = 0
            s.total_waiting_time = 0


# 在Station类添加属性
class Station:
    def __init__(self, id, name, passenger_rate=10):
        self.id = id
        self.name = name
        self.passenger_arrival_rate = passenger_rate
        self.waiting_passengers = 0  # 新增等待乘客计数
        self.total_passengers_served = 0  # 新增服务乘客统计
        self.total_waiting_time = 0  # 新增等待时间统计

    def get_statistics(self):
        return {
            'id': self.id,
            'waiting_passengers': self.waiting_passengers,
            'total_served': self.total_passengers_served,
            'total_waiting_time': self.total_waiting_time
        }


# 在Train类补充乘客管理
class Train:
    def __init__(self, id, capacity=200, max_speed=80):
        self.id = id
        self.capacity = capacity
        self.max_speed = max_speed
        self.passengers = []  # 新增乘客列表

    def board_passengers(self, station):
        """乘客上车"""
        available_seats = self.capacity - len(self.passengers)
        boarding = min(available_seats, station.waiting_passengers)

        # 记录等待时间
        boarding_passengers = station.waiting_passengers[:boarding]
        self.passengers.extend(boarding_passengers)

        total_wait = sum(p.wait_time for p in boarding_passengers)
        station.total_waiting_time += total_wait
        station.total_passengers_served += boarding
        station.waiting_passengers = station.waiting_passengers[boarding:]

        return boarding

    def alight_passengers(self, station, probability):
        """乘客下车"""
        alighting = [p for p in self.passengers if p.destination == station.id]
        num_alight = min(len(alighting), int(len(self.passengers) * probability))

        alighted = alighting[:num_alight]
        self.passengers = [p for p in self.passengers if p not in alighted]

        return num_alight
