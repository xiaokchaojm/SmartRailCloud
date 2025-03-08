// static/js/charts.js - Chart functionality for SmartRailCloud

const ChartModule = {
    // Store chart instances
    charts: {
        systemPerformance: null,
        passengerFlow: null,
        energy: null,
        fitness: null,
        solutionMetrics: null,
        stationAnalysis: null
    },
    
    // Initialize all charts
    initializeCharts: function() {
        console.log('Initializing charts...');
        
        // Initialize dashboard charts
        this.initializeSystemPerformanceChart();
        
        // Initialize simulator charts
        this.initializePassengerFlowChart();
        this.initializeEnergyChart();
        
        // Initialize optimizer charts
        this.initializeFitnessChart();
        this.initializeSolutionMetricsChart();
        
        // Initialize report charts
        this.initializeStationAnalysisChart();
        
        console.log('Charts initialized.');
    },
    
    // Initialize system performance chart
    initializeSystemPerformanceChart: function() {
        const ctx = document.getElementById('system-performance-chart');
        if (!ctx) return;
        
        // Demo data
        const labels = ['06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', 
                         '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00'];
        const passengerData = [120, 340, 580, 520, 380, 320, 340, 360, 300, 320, 440, 580, 520, 380];
        const waitTimeData = [2, 5, 8, 6, 4, 3, 3, 3, 3, 3, 5, 7, 5, 4];
        const energyData = [250, 320, 380, 370, 330, 310, 320, 325, 310, 320, 350, 380, 370, 330];
        
        this.charts.systemPerformance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Passenger Count',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    data: passengerData,
                    yAxisID: 'y',
                    fill: true
                }, {
                    label: 'Wait Time (min)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    data: waitTimeData,
                    yAxisID: 'y1',
                    fill: false
                }, {
                    label: 'Energy Use (kWh)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    data: energyData,
                    yAxisID: 'y2',
                    fill: false
                }]
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Passengers'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {
                            drawOnChartArea: false,
                        },
                        title: {
                            display: true,
                            text: 'Wait Time (min)'
                        }
                    },
                    y2: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {
                            drawOnChartArea: false,
                        },
                        title: {
                            display: true,
                            text: 'Energy (kWh)'
                        }
                    },
                }
            }
        });
    },
    
    // Initialize passenger flow chart
    initializePassengerFlowChart: function() {
        const ctx = document.getElementById('passenger-flow-chart');
        if (!ctx) return;
        
        // Demo data
        const labels = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'];
        const boardingData = [65, 59, 80, 81, 56, 55, 40, 28];
        const alightingData = [28, 48, 40, 19, 86, 27, 90, 42];
        
        this.charts.passengerFlow = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Boarding',
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    data: boardingData
                }, {
                    label: 'Alighting',
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    data: alightingData
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Passenger Flow by Station'
                    },
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    },
    
    // Initialize energy chart
    initializeEnergyChart: function() {
        const ctx = document.getElementById('energy-chart');
        if (!ctx) return;
        
        // Demo data
        const labels = ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00'];
        const energyData = [250, 380, 330, 320, 310, 380, 370, 280];
        
        this.charts.energy = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Energy (kWh)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    fill: true,
                    data: energyData
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Energy Consumption Over Time'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Energy (kWh)'
                        }
                    }
                }
            }
        });
    },
    
    // Initialize fitness chart
    initializeFitnessChart: function() {
        const ctx = document.getElementById('fitness-chart');
        if (!ctx) return;
        
        // Demo data - this will be updated during optimization
        const labels = [1, 2, 3, 4, 5];
        const bestFitnessData = [0.2, 0.3, 0.35, 0.4, 0.45];
        const avgFitnessData = [0.1, 0.15, 0.2, 0.25, 0.3];
        
        this.charts.fitness = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Best Fitness',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    data: bestFitnessData,
                    tension: 0.1
                }, {
                    label: 'Average Fitness',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    data: avgFitnessData,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Generation'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Fitness Score'
                        },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    },
    
    // Initialize solution metrics chart
    initializeSolutionMetricsChart: function() {
        const ctx = document.getElementById('solution-metrics-chart');
        if (!ctx) return;
        
        // Demo data
        const labels = ['Wait Time', 'Energy Efficiency', 'Capacity Utilization', 'Headway Compliance', 'Passenger Satisfaction'];
        const currentData = [0.65, 0.6, 0.7, 0.7, 0.6];
        const optimizedData = [0.65, 0.6, 0.7, 0.7, 0.6]; // Start same as current
        
        this.charts.solutionMetrics = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Current',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    pointBackgroundColor: 'rgba(255, 99, 132, 1)',
                    data: currentData
                }, {
                    label: 'Optimized',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    pointBackgroundColor: 'rgba(54, 162, 235, 1)',
                    data: optimizedData
                }]
            },
            options: {
                responsive: true,
                scales: {
                    r: {
                        min: 0,
                        max: 1,
                        ticks: {
                            stepSize: 0.2
                        }
                    }
                }
            }
        });
    },
    
    // Initialize station analysis chart
    initializeStationAnalysisChart: function() {
        const ctx = document.getElementById('station-analysis-chart');
        if (!ctx) return;
        
        // Demo data
        const labels = ['Station 1', 'Station 2', 'Station 3', 'Station 4', 'Station 5', 'Station 6', 'Station 7', 'Station 8', 'Station 9', 'Station 10'];
        const volumeData = [1250, 1480, 1820, 2100, 3200, 1750, 1400, 1600, 1950, 2250];
        const waitTimeData = [3.2, 3.5, 4.1, 4.5, 5.8, 4.2, 3.8, 3.9, 4.3, 4.7];
        
        this.charts.stationAnalysis = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Passenger Volume',
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    data: volumeData,
                    yAxisID: 'y'
                }, {
                    label: 'Avg. Wait Time (min)',
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    data: waitTimeData,
                    type: 'line',
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Passenger Volume'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {
                            drawOnChartArea: false,
                        },
                        title: {
                            display: true,
                            text: 'Wait Time (min)'
                        },
                        min: 0
                    }
                }
            }
        });
    },
    
    // Update passenger flow chart with new data
    updatePassengerFlowChart: function(data) {
        if (!this.charts.passengerFlow) {
            this.initializePassengerFlowChart();
            if (!this.charts.passengerFlow) return;
        }
        
        // Extract data from array of objects
        const stations = data.map(item => item.station);
        const boarding = data.map(item => item.boarding);
        const alighting = data.map(item => item.alighting);
        
        // Update chart data
        this.charts.passengerFlow.data.labels = stations;
        this.charts.passengerFlow.data.datasets[0].data = boarding;
        this.charts.passengerFlow.data.datasets[1].data = alighting;
        
        // Update chart
        this.charts.passengerFlow.update();
    },
    
    // Update energy chart with new data
    updateEnergyChart: function(data) {
        if (!this.charts.energy) {
            this.initializeEnergyChart();
            if (!this.charts.energy) return;
        }
        
        // Extract data from array of objects
        const times = data.map(item => item.time);
        const energy = data.map(item => item.energy);
        
        // Update chart data
        this.charts.energy.data.labels = times;
        this.charts.energy.data.datasets[0].data = energy;
        
        // Update chart
        this.charts.energy.update();
    },
    
    // Update fitness chart with new data
    updateFitnessChart: function(labels, bestFitness, avgFitness) {
        if (!this.charts.fitness) {
            this.initializeFitnessChart();
            if (!this.charts.fitness) return;
        }
        
        // Update chart data
        this.charts.fitness.data.labels = labels;
        this.charts.fitness.data.datasets[0].data = bestFitness;
        this.charts.fitness.data.datasets[1].data = avgFitness;
        
        // Update chart
        this.charts.fitness.update();
    },
    
    // Update solution metrics chart with new data
    updateSolutionMetricsChart: function(currentMetrics, optimizedMetrics) {
        if (!this.charts.solutionMetrics) {
            this.initializeSolutionMetricsChart();
            if (!this.charts.solutionMetrics) return;
        }
        
        // Update chart data
        this.charts.solutionMetrics.data.datasets[0].data = currentMetrics;
        this.charts.solutionMetrics.data.datasets[1].data = optimizedMetrics;
        
        // Update chart
        this.charts.solutionMetrics.update();
    },
    
    // Initialize special charts for reports
    initializeReportCharts: function() {
        // Re-initialize station analysis chart if needed
        if (!this.charts.stationAnalysis) {
            this.initializeStationAnalysisChart();
        }
    }
};
