from flask import Flask, render_template, request, jsonify, send_file
import json
import os
import logging
import threading
import time
from datetime import datetime
import io


from simulator.simulator import RailSystemSimulator  # 用实际类名替换Simulator
from simulator.optimizer import RailSystemOptimizer  # 用实际类名替换Optimizer
from models.line import Line
from models.train import Train
from models.station import Station

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# 全局变量
simulator = None
optimizer = None
optimization_thread = None
optimization_status = {
    'status': 'not_started',
    'progress': 0,
    'best_score': 0,
    'message': 'Optimizer not initialized',
    'is_running': False
}


# 初始化模拟器和相关对象
stations = [
    Station(id=1, name="Central Station", passenger_rate=10),
    Station(id=2, name="North Terminal", passenger_rate=8),
    Station(id=3, name="East Junction", passenger_rate=5),
    Station(id=4, name="South Gardens", passenger_rate=7),
    Station(id=5, name="West Hills", passenger_rate=6),
    Station(id=6, name="University", passenger_rate=12),
    Station(id=7, name="City Center", passenger_rate=15),
    Station(id=8, name="Airport", passenger_rate=9)
]

# 手动添加坐标和重要性属性
coordinates = [(0,0), (5,10), (15,5), (10,-5), (-5,0), (-10,15), (7,7), (20,0)]
importance_values = [0.8, 0.6, 0.4, 0.5, 0.5, 0.7, 0.9, 0.8]

for i, station in enumerate(stations):
    station.x, station.y = coordinates[i]
    station.importance = importance_values[i]

    # 创建线路
    line = Line(id=1, name="Downtown Express", stations=stations)

    # 设置站点之间的连接
    line.set_connections([
        (1, 2, 8),  # Central Station to North Terminal, 8 minutes
        (2, 7, 6),  # North Terminal to City Center, 6 minutes
        (7, 3, 7),  # City Center to East Junction, 7 minutes
        (3, 8, 10),  # East Junction to Airport, 10 minutes
        (3, 4, 8),  # East Junction to South Gardens, 8 minutes
        (4, 1, 7),  # South Gardens to Central Station, 7 minutes
        (1, 5, 6),  # Central Station to West Hills, 6 minutes
        (5, 6, 9)  # West Hills to University, 9 minutes
    ])

    # 创建列车
    trains = [
        Train(id=1, capacity=200, max_speed=80),
        Train(id=2, capacity=200, max_speed=80),
        Train(id=3, capacity=200, max_speed=80),
        Train(id=4, capacity=200, max_speed=80),
        Train(id=5, capacity=200, max_speed=80),
        Train(id=6, capacity=200, max_speed=80),
        Train(id=7, capacity=200, max_speed=80),
        Train(id=8, capacity=200, max_speed=80)
    ]

    # 设置乘客生成器
    passenger_generator = PassengerGenerator(
        stations=stations,
        base_rates={station.id: station.passenger_arrival_rate for station in stations},
        time_patterns={
            'morning_rush': {
                'start_time': 420,  # 7:00 AM in minutes
                'end_time': 540,  # 9:00 AM in minutes
                'multiplier': 2.0
            },
            'evening_rush': {
                'start_time': 1020,  # 5:00 PM in minutes
                'end_time': 1140,  # 7:00 PM in minutes
                'multiplier': 1.8
            }
        }
    )

    # 创建模拟器
    sim = RailSystemSimulator(
        line=line,
        trains=trains,
        passenger_generator=passenger_generator,
        config={
            'time_step': 1,  # 1 minute time steps
            'simulation_speed': 10,  # 10x real-time
            'random_delays': True,  # Simulate random delays
            'energy_model': 'basic',  # Use basic energy model
            'max_passengers': 5000,  # Cap on number of simulated passengers
            'min_headway': 5  # Minimum headway in minutes
        }
    )

    # 初始化列车位置
    sim._initialize_train_positions()

    return sim


# 运行优化的后台线程
def run_optimization(optimize_config, callback=None):
    global optimization_status, optimizer

    optimization_status['status'] = 'running'
    optimization_status['is_running'] = True
    optimization_status['progress'] = 0
    optimization_status['message'] = 'Optimization started'
    optimization_status['start_time'] = time.time()

    # 每次迭代的回调函数
    def optimization_callback(generation, best_solution, best_score, *args, **kwargs):
        global optimization_status

        # 更新状态
        progress = (generation + 1) / optimize_config['generations'] * 100
        optimization_status['progress'] = min(100, progress)
        optimization_status['generation'] = generation
        optimization_status['best_score'] = best_score

        # 检查是否应该停止
        if not optimization_status['is_running']:
            return False  # 返回False会停止优化

        return True

    try:
        # 运行优化
        result = optimizer.optimize(callback=optimization_callback)

        # 处理完成后的状态更新
        if optimization_status['is_running']:  # 只有在未被手动停止的情况下
            optimization_status['status'] = 'completed'
            optimization_status['progress'] = 100
            optimization_status['message'] = 'Optimization completed successfully'
            optimization_status['result'] = result
            optimization_status['end_time'] = time.time()
            optimization_status['execution_time'] = optimization_status['end_time'] - optimization_status['start_time']

    except Exception as e:
        app.logger.error(f"Optimization error: {str(e)}")
        optimization_status['status'] = 'error'
        optimization_status['message'] = f"Error during optimization: {str(e)}"

    finally:
        optimization_status['is_running'] = False
        if callback:
            callback(optimization_status)


# 初始化全局模拟器
simulator = initialize_simulator()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/simulator/initialize', methods=['POST'])
def init_simulation():
    global simulator

    try:
        data = request.json

        # 重新初始化模拟器
        simulator = initialize_simulator()

        # 应用配置
        if data:
            num_trains = data.get('num_trains')
            if num_trains and int(num_trains) > 0:
                # 调整列车数量
                current_trains = len(simulator.trains)
                target_trains = int(num_trains)

                if current_trains > target_trains:
                    # 移除多余的列车
                    simulator.trains = simulator.trains[:target_trains]
                elif current_trains < target_trains:
                    # 添加额外的列车
                    for i in range(current_trains, target_trains):
                        new_train = Train(id=i + 1, capacity=200, max_speed=80)
                        simulator.trains.append(new_train)

            # 设置其他配置
            headway = data.get('headway')
            if headway:
                simulator.config['min_headway'] = float(headway)

            # 乘客流量设置
            passenger_flow = data.get('passenger_flow', 'medium')
            multipliers = {
                'low': 0.5,
                'medium': 1.0,
                'high': 1.5,
                'rush-hour': 2.0
            }

            if passenger_flow in multipliers:
                # 调整所有站点的乘客到达率
                for station in simulator.line.stations:
                    station.passenger_arrival_rate *= multipliers[passenger_flow]

            # 随机延迟设置
            random_delays = data.get('random_delays', True)
            simulator.config['random_delays'] = random_delays

            # 模拟速度
            sim_speed = data.get('sim_speed', 10)
            simulator.config['simulation_speed'] = float(sim_speed)

        # 重新初始化列车位置
        simulator._initialize_train_positions()

        return jsonify({
            'status': 'success',
            'message': 'Simulator initialized successfully',
            'config': simulator.config,
            'trains': len(simulator.trains),
            'stations': len(simulator.line.stations)
        })

    except Exception as e:
        app.logger.error(f"Initialization error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error initializing simulator: {str(e)}"
        }), 500


@app.route('/api/simulator/start', methods=['POST'])
def start_simulation():
    global simulator

    try:
        data = request.json
        duration = int(data.get('duration', 120))

        # 检查模拟器是否已初始化
        if simulator is None:
            simulator = initialize_simulator()

        # 开始模拟
        simulator.reset_simulation()  # 确保从干净状态开始

        # 运行模拟并获取结果
        results = simulator.run_simulation(duration=duration)

        # 提取关键统计数据
        stats = results.get('statistics', {})

        return jsonify({
            'status': 'success',
            'message': 'Simulation completed',
            'duration': duration,
            'statistics': stats,
            'passengers_served': stats.get('total_passengers_served', 0),
            'average_waiting_time': stats.get('average_waiting_time', 0),
            'total_energy_consumed': stats.get('total_energy_consumed', 0),
            'train_utilization': stats.get('train_utilization', 0)
        })

    except Exception as e:
        app.logger.error(f"Simulation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error running simulation: {str(e)}"
        }), 500


@app.route('/api/simulator/status', methods=['GET'])
def get_simulator_status():
    global simulator

    try:
        if simulator is None:
            return jsonify({
                'status': 'not_initialized',
                'message': 'Simulator not initialized'
            })

        # 获取系统状态
        system_state = simulator.get_system_state()

        # 提取列车状态信息
        train_status = []
        for train in simulator.trains:
            status = {
                'id': train.id,
                'status': train.status,
                'current_station': train.current_station,
                'next_station': train.next_station,
                'passengers': len(train.passengers),
                'capacity': train.capacity,
                'utilization': len(train.passengers) / train.capacity if train.capacity > 0 else 0,
                'delay': train.delay
            }
            train_status.append(status)

        # 提取站点状态信息
        station_status = []
        for station in simulator.line.stations:
            waiting = simulator.get_waiting_passengers(station.id)
            status = {
                'id': station.id,
                'name': station.name,
                'waiting_passengers': len(waiting) if waiting is not None else 0,
                'arrival_rate': station.passenger_arrival_rate
            }
            station_status.append(status)

        # 获取模拟系统时间
        sim_time = simulator.current_time
        hours = sim_time // 60
        minutes = sim_time % 60
        formatted_time = f"{hours:02d}:{minutes:02d}"

        return jsonify({
            'status': 'active',
            'simulation_time': formatted_time,
            'raw_time': sim_time,
            'is_running': simulator.is_running,
            'trains': train_status,
            'stations': station_status,
            'statistics': system_state.get('statistics', {}),
            'config': simulator.config
        })

    except Exception as e:
        app.logger.error(f"Status error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error getting simulator status: {str(e)}"
        }), 500


@app.route('/api/simulator/reset', methods=['POST'])
def reset_simulation():
    global simulator

    try:
        if simulator is None:
            simulator = initialize_simulator()
        else:
            simulator.reset_simulation()
            simulator._initialize_train_positions()

        return jsonify({
            'status': 'success',
            'message': 'Simulator reset successfully'
        })

    except Exception as e:
        app.logger.error(f"Reset error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error resetting simulator: {str(e)}"
        }), 500


@app.route('/api/simulator/pause', methods=['POST'])
def pause_simulation():
    global simulator

    try:
        if simulator and simulator.is_running:
            simulator.pause_simulation()
            return jsonify({
                'status': 'success',
                'message': 'Simulation paused'
            })
        else:
            return jsonify({
                'status': 'warning',
                'message': 'Simulation is not running'
            })

    except Exception as e:
        app.logger.error(f"Pause error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error pausing simulation: {str(e)}"
        }), 500


@app.route('/api/simulator/resume', methods=['POST'])
def resume_simulation():
    global simulator

    try:
        if simulator:
            simulator.resume_simulation()
            return jsonify({
                'status': 'success',
                'message': 'Simulation resumed'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Simulator not initialized'
            }), 400

    except Exception as e:
        app.logger.error(f"Resume error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error resuming simulation: {str(e)}"
        }), 500


@app.route('/api/optimizer/start', methods=['POST'])
def start_optimization():
    global optimizer, simulator, optimization_thread, optimization_status

    try:
        data = request.json

        # 检查模拟器是否已初始化
        if simulator is None:
            return jsonify({
                'status': 'error',
                'message': 'Simulator not initialized'
            }), 400

        # 确保没有其他优化任务正在运行
        if optimization_status.get('is_running', False):
            return jsonify({
                'status': 'error',
                'message': 'Another optimization is already running'
            }), 400

        # 从请求中提取配置参数
        algorithm = data.get('algorithm', 'genetic')
        objective = data.get('objective', 'balanced')

        # 创建优化器配置
        optimize_config = {
            'algorithm': algorithm,
            'objective': objective,
            'simulation_duration': int(data.get('opt_duration', 120)),
            'population_size': int(data.get('population_size', 20)),
            'generations': int(data.get('generations', 30)),
            'mutation_rate': float(data.get('mutation_rate', 0.2)),
            'crossover_rate': float(data.get('crossover_rate', 0.7)),
            'weights': {
                'passenger_wait_time': float(data.get('weight_wait', 0.4)),
                'energy_consumption': float(data.get('weight_energy', 0.3)),
                'capacity_utilization': float(data.get('weight_capacity', 0.3))
            },
            'constraints': {
                'min_headway': float(data.get('min_headway', 3)),
                'max_headway': float(data.get('max_headway', 15)),
                'min_trains': int(data.get('min_trains', 5)),
                'max_trains': int(data.get('max_trains', 15))
            }
        }

        # 初始化优化器
        optimizer = RailSystemOptimizer(simulator=simulator, config=optimize_config)

        # 在后台线程中运行优化
        optimization_thread = threading.Thread(
            target=run_optimization,
            args=(optimize_config,)
        )
        optimization_thread.daemon = True
        optimization_thread.start()

        return jsonify({
            'status': 'started',
            'message': 'Optimization started',
            'config': optimize_config,
            'algorithm': algorithm,
            'objective': objective
        })

    except Exception as e:
        app.logger.error(f"Optimization start error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error starting optimization: {str(e)}"
        }), 500


@app.route('/api/optimizer/progress', methods=['GET'])
def get_optimization_progress():
    global optimization_status, optimizer

    try:
        if optimizer is None:
            return jsonify({
                'status': 'not_started',
                'progress': 0,
                'message': 'Optimizer not initialized'
            })

        # 计算运行时间
        if 'start_time' in optimization_status:
            current_time = time.time()
            elapsed_seconds = current_time - optimization_status['start_time']
            hours = int(elapsed_seconds // 3600)
            minutes = int((elapsed_seconds % 3600) // 60)
            seconds = int(elapsed_seconds % 60)
            elapsed_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            optimization_status['elapsed_time'] = elapsed_time

        # 获取优化历史
        if optimizer.history:
            history_data = []
            for entry in optimizer.history:
                # 转换历史条目为可序列化格式
                serializable_entry = {}
                for key, value in entry.items():
                    if key != 'best_solution':  # 排除复杂对象
                        serializable_entry[key] = value
                history_data.append(serializable_entry)
            optimization_status['history'] = history_data

        # 获取当前最佳解方案的详细信息
        if optimizer.best_solution:
            best_solution = optimizer.best_solution

            # 提取关键参数
            solution_details = {
                'num_trains': best_solution.get('num_trains', 0),
                'global_headway': best_solution.get('global_headway', 0),
                'global_speed_factor': best_solution.get('global_speed_factor', 1.0)
            }
            optimization_status['best_solution_details'] = solution_details

        return jsonify(optimization_status)

    except Exception as e:
        app.logger.error(f"Progress error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error getting optimization progress: {str(e)}"
        }), 500


@app.route('/api/optimizer/stop', methods=['POST'])
def stop_optimization():
    global optimization_status

    try:
        if not optimization_status.get('is_running', False):
            return jsonify({
                'status': 'warning',
                'message': 'No optimization is currently running'
            })

        # 设置停止标志
        optimization_status['is_running'] = False
        optimization_status['status'] = 'stopped'
        optimization_status['message'] = 'Optimization stopped by user'

        return jsonify({
            'status': 'success',
            'message': 'Stop signal sent to optimizer'
        })

    except Exception as e:
        app.logger.error(f"Stop error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error stopping optimization: {str(e)}"
        }), 500


@app.route('/api/optimizer/apply', methods=['POST'])
def apply_optimization():
    global optimizer, simulator

    try:
        if optimizer is None or optimizer.best_solution is None:
            return jsonify({
                'status': 'error',
                'message': 'No optimization solution available to apply'
            }), 400

        # 保存当前模拟器配置以便比较
        prev_num_trains = len(simulator.trains)
        prev_headway = simulator.config.get('min_headway', 0)

        # 应用最佳解决方案到模拟器
        simulator = optimizer.apply_best_solution(simulator)

        # 计算变化
        new_num_trains = len(simulator.trains)
        new_headway = simulator.config.get('min_headway', 0)

        changes = {
            'num_trains': {
                'before': prev_num_trains,
                'after': new_num_trains,
                'change': new_num_trains - prev_num_trains
            },
            'headway': {
                'before': prev_headway,
                'after': new_headway,
                'change': new_headway - prev_headway
            }
        }

        return jsonify({
            'status': 'success',
            'message': 'Optimization solution applied successfully',
            'changes': changes,
            'solution': optimizer.best_solution
        })

    except Exception as e:
        app.logger.error(f"Apply error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error applying optimization: {str(e)}"
        }), 500


@app.route('/api/optimizer/results', methods=['GET'])
def get_optimization_results():
    global optimizer

    try:
        if optimizer is None or optimizer.best_solution is None:
            return jsonify({
                'status': 'error',
                'message': 'No optimization results available'
            }), 404

        # 获取完整的优化结果
        results = optimizer._format_result()

        # 转换为可序列化格式
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            elif hasattr(obj, '__dict__'):
                return make_serializable(obj.__dict__)
            else:
                try:
                    # 尝试转换为JSON可序列化类型
                    json.dumps(obj)
                    return obj
                except (TypeError, OverflowError):
                    return str(obj)

        serializable_results = make_serializable(results)

        return jsonify({
            'status': 'success',
            'results': serializable_results
        })

    except Exception as e:
        app.logger.error(f"Results error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error getting optimization results: {str(e)}"
        }), 500


@app.route('/api/optimizer/export', methods=['GET'])
def export_optimization_results():
    global optimizer

    try:
        if optimizer is None or optimizer.best_solution is None:
            return jsonify({
                'status': 'error',
                'message': 'No optimization results available to export'
            }), 404

        # 保存优化结果
        filename = optimizer.save_results()

        return jsonify({
            'status': 'success',
            'message': 'Results exported successfully',
            'filename': filename
        })

    except Exception as e:
        app.logger.error(f"Export error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error exporting results: {str(e)}"
        }), 500


@app.route('/api/system/info', methods=['GET'])
def get_system_info():
    global simulator

    try:
        if simulator is None:
            return jsonify({
                'status': 'not_initialized',
                'message': 'System not initialized'
            })

        # 获取线路信息
        line_info = {
            'id': simulator.line.id,
            'name': simulator.line.name,
            'stations': [
                {
                    'id': station.id,
                    'name': station.name,
                    'x': station.x,
                    'y': station.y,
                    'passenger_arrival_rate': station.passenger_arrival_rate
                }
                for station in simulator.line.stations
            ],
            'connections': [
                {
                    'from_id': conn.from_station,
                    'to_id': conn.to_station,
                    'travel_time': conn.travel_time
                }
                for conn in simulator.line.connections
            ],
            'total_length': simulator.line.calculate_total_length()
        }

        # 获取列车信息
        train_info = {
            'total': len(simulator.trains),
            'active': sum(1 for train in simulator.trains if train.status != train.STATUS_INACTIVE),
            'trains': [
                {
                    'id': train.id,
                    'capacity': train.capacity,
                    'max_speed': train.max_speed,
                    'status': train.status
                }
                for train in simulator.trains
            ]
        }

        # 获取模拟器信息
        simulator_info = {
            'status': 'running' if simulator.is_running else 'stopped',
            'current_time': simulator.current_time,
            'formatted_time': f"{simulator.current_time // 60:02d}:{simulator.current_time % 60:02d}",
            'config': simulator.config
        }

        # 获取统计信息
        statistics = simulator.get_statistics()

        return jsonify({
            'status': 'success',
            'line': line_info,
            'trains': train_info,
            'simulator': simulator_info,
            'statistics': statistics,
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        app.logger.error(f"System info error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error getting system information: {str(e)}"
        }), 500


@app.route('/api/reports/generate', methods=['POST'])
def generate_report():
    global simulator, optimizer

    try:
        data = request.json
        report_type = data.get('type', 'system_performance')

        if simulator is None:
            return jsonify({
                'status': 'error',
                'message': 'Simulator not initialized'
            }), 400

        # 获取系统状态和统计信息
        system_state = simulator.get_system_state()
        statistics = simulator.get_statistics()

        # 准备报告数据
        report_data = {
            'title': 'System Performance Report',
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'line_name': simulator.line.name,
            'stations': len(simulator.line.stations),
            'trains': len(simulator.trains),
            'statistics': statistics
        }

        # 添加优化结果（如果有）
        if optimizer and optimizer.best_solution:
            report_data['optimization'] = {
                'algorithm': optimizer.config['algorithm'],
                'objective': optimizer.config['objective'],
                'best_score': optimizer.best_score,
                'best_solution': optimizer.best_solution,
                'iterations': len(optimizer.history)
            }

        # 根据报告类型添加特定数据
        if report_type == 'energy_consumption':
            report_data['title'] = 'Energy Consumption Analysis'
            # 添加更多能耗分析数据
            energy_by_train = {}
            for train in simulator.trains:
                energy_by_train[train.id] = train.energy_consumed
            report_data['energy_by_train'] = energy_by_train

        elif report_type == 'passenger_flow':
            report_data['title'] = 'Passenger Flow Study'
            # 添加乘客流量数据
            passenger_data = {}
            for station in simulator.line.stations:
                waiting = simulator.get_waiting_passengers(station.id)
                passenger_data[station.id] = {
                    'name': station.name,
                    'waiting': len(waiting) if waiting is not None else 0,
                    'arrival_rate': station.passenger_arrival_rate
                }
            report_data['passenger_data'] = passenger_data

        elif report_type == 'optimization_results':
            report_data['title'] = 'Optimization Results'
            # 优化结果已在前面添加

        elif report_type == 'delay_analysis':
            report_data['title'] = 'Delay Analysis'
            # 添加延迟分析数据
            delay_data = {}
            for train in simulator.trains:
                delay_data[train.id] = {
                    'current_delay': train.delay,
                    'total_delays': train.total_delays
                }
            report_data['delay_data'] = delay_data

        return jsonify({
            'status': 'success',
            'report': report_data
        })

    except Exception as e:
        app.logger.error(f"Report generation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error generating report: {str(e)}"
        }), 500


@app.route('/api/data/export', methods=['POST'])
def export_data():
    global simulator, optimizer

    try:
        data = request.json
        format_type = data.get('format', 'json')

        if simulator is None:
            return jsonify({
                'status': 'error',
                'message': 'Simulator not initialized'
            }), 400

        # 获取系统状态和统计信息
        system_state = simulator.get_system_state()
        statistics = simulator.get_statistics()

        # 准备导出数据
        export_data = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system_state': system_state,
            'statistics': statistics
        }

        # 添加优化结果（如果有）
        if optimizer and optimizer.best_solution:
            export_data['optimization'] = {
                'algorithm': optimizer.config['algorithm'],
                'objective': optimizer.config['objective'],
                'best_score': optimizer.best_score,
                'best_solution': optimizer.best_solution
            }

        # 根据格式类型导出
        if format_type == 'json':
            # 转换为JSON字符串
            json_data = json.dumps(export_data, indent=2)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"smartrailcloud_export_{timestamp}.json"

            # 创建内存文件对象
            file_obj = io.BytesIO(json_data.encode('utf-8'))

            # 返回文件下载响应
            return send_file(
                file_obj,
                mimetype='application/json',
                as_attachment=True,
                download_name=filename
            )

        elif format_type == 'csv':
            # 实现CSV格式导出
            import csv
            import zipfile
            from io import StringIO, BytesIO

            # 创建一个ZIP文件来存储多个CSV
            memory_file = BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 1. 系统概况CSV
                system_info = StringIO()
                system_writer = csv.writer(system_info)
                system_writer.writerow(['Property', 'Value'])
                system_writer.writerow(['Export Time', export_data['generated_at']])
                system_writer.writerow(['Line Name', simulator.line.name])
                system_writer.writerow(['Number of Stations', len(simulator.line.stations)])
                system_writer.writerow(['Number of Trains', len(simulator.trains)])
                system_writer.writerow(
                    ['Active Trains', sum(1 for t in simulator.trains if t.status != t.STATUS_INACTIVE)])
                system_writer.writerow(['Current Simulation Time',
                                        f"{simulator.current_time // 60:02d}:{simulator.current_time % 60:02d}"])
                system_writer.writerow(['Simulation Status', 'Running' if simulator.is_running else 'Stopped'])
                zf.writestr('system_overview.csv', system_info.getvalue())

                # 2. 站点数据CSV
                stations_info = StringIO()
                stations_writer = csv.writer(stations_info)
                stations_writer.writerow(
                    ['Station ID', 'Name', 'X', 'Y', 'Passenger Arrival Rate', 'Waiting Passengers'])
                for station in simulator.line.stations:
                    waiting = simulator.get_waiting_passengers(station.id)
                    stations_writer.writerow([
                        station.id,
                        station.name,
                        station.x,
                        station.y,
                        station.passenger_arrival_rate,
                        len(waiting) if waiting is not None else 0
                    ])
                zf.writestr('stations_data.csv', stations_info.getvalue())

                # 3. 列车数据CSV
                trains_info = StringIO()
                trains_writer = csv.writer(trains_info)
                trains_writer.writerow(['Train ID', 'Capacity', 'Current Status', 'Current Station',
                                        'Next Station', 'Passengers', 'Utilization', 'Delay', 'Energy Consumed'])
                for train in simulator.trains:
                    utilization = len(train.passengers) / train.capacity if train.capacity > 0 else 0
                    trains_writer.writerow([
                        train.id,
                        train.capacity,
                        train.status,
                        train.current_station,
                        train.next_station,
                        len(train.passengers),
                        f"{utilization:.2f}",
                        train.delay,
                        train.energy_consumed if hasattr(train, 'energy_consumed') else 0
                    ])
                zf.writestr('trains_data.csv', trains_info.getvalue())

                # 4. 统计数据CSV
                stats_info = StringIO()
                stats_writer = csv.writer(stats_info)
                stats_writer.writerow(['Metric', 'Value'])
                for key, value in statistics.items():
                    stats_writer.writerow([key.replace('_', ' ').title(), value])
                zf.writestr('statistics.csv', stats_info.getvalue())

                # 5. 优化结果CSV（如果有）
                if optimizer and optimizer.best_solution:
                    opt_info = StringIO()
                    opt_writer = csv.writer(opt_info)
                    opt_writer.writerow(['Parameter', 'Value'])
                    opt_writer.writerow(['Algorithm', optimizer.config['algorithm']])
                    opt_writer.writerow(['Objective', optimizer.config['objective']])
                    opt_writer.writerow(['Best Score', optimizer.best_score])

                    # 添加最佳解决方案的参数
                    for key, value in optimizer.best_solution.items():
                        if isinstance(value, (int, float, str, bool)):  # 只包含简单类型
                            opt_writer.writerow([key.replace('_', ' ').title(), value])

                    zf.writestr('optimization_results.csv', opt_info.getvalue())

            # 准备文件名和内存文件下载
            memory_file.seek(0)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return send_file(
                memory_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'smartrailcloud_export_{timestamp}.zip'
            )

        else:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported export format: {format_type}'
            }), 400

    except Exception as e:
        app.logger.error(f"Data export error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error exporting data: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
