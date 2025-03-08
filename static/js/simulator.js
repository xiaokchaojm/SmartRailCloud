// static/js/simulator.js - Simulator functionality for SmartRailCloud

const SimulatorModule = {
    // Simulator state
    state: {
        running: false,
        paused: false,
        simulationTime: 0,
        simulationSpeed: 10,
        trains: [],
        stations: [],
        passengerData: [],
        energyData: [],
        configuredSettings: {
            numTrains: 8,
            headway: 5,
            passengerFlow: 'medium',
            randomDelays: true,
            duration: 120
        }
    },
    
    // Initialize the simulator module
    init: function() {
        console.log('Initializing simulator module...');
        
        // Set up visualization
        this.setupVisualization();
        
        // Load initial data
        this.loadInitialData();
        
        // Initialize form with current settings
        this.initializeForm();
        
        console.log('Simulator module initialized.');
    },
    
    // Set up the line visualization
    setupVisualization: function() {
        const vizContainer = document.getElementById('line-visualization');
        
        // Clear any existing content
        vizContainer.innerHTML = '';
        
        // Create track line
        const trackLine = document.createElement('div');
        trackLine.className = 'track-line';
        trackLine.style.width = '90%';
        trackLine.style.left = '5%';
        trackLine.style.top = '50%';
        vizContainer.appendChild(trackLine);
        
        // For demonstration purposes, create some sample stations
        const stationPositions = [
            { id: 1, name: 'Terminal 1', x: 10 },
            { id: 2, name: 'Downtown', x: 25 },
            { id: 3, name: 'Central Square', x: 40 },
            { id: 4, name: 'University', x: 55 },
            { id: 5, name: 'Business District', x: 70 },
            { id: 6, name: 'Terminal 2', x: 85 }
        ];
        
        // Create station markers
        stationPositions.forEach(station => {
            const stationMarker = document.createElement('div');
            stationMarker.className = 'station-marker';
            stationMarker.dataset.stationId = station.id;
            stationMarker.dataset.stationName = station.name;
            stationMarker.style.left = `${station.x}%`;
            stationMarker.style.top = '50%';
            
            // Add tooltip functionality
            stationMarker.addEventListener('mouseenter', (e) => {
                this.showTooltip(e.target, `${station.name}<br>Waiting: 45 passengers`);
            });
            
            stationMarker.addEventListener('mouseleave', () => {
                this.hideTooltip();
            });
            
            vizContainer.appendChild(stationMarker);
        });
        
        // Save station data
        this.state.stations = stationPositions;
        
        // Create sample trains
        for (let i = 0; i < 4; i++) {
            const trainMarker = document.createElement('div');
            trainMarker.className = 'train-marker';
            trainMarker.dataset.trainId = i + 1;
            
            // Position trains along the line
            const position = 10 + (i * 25);
            trainMarker.style.left = `${position}%`;
            trainMarker.style.top = '50%';
            
            // Add tooltip functionality
            trainMarker.addEventListener('mouseenter', (e) => {
                this.showTooltip(e.target, `Train ${i+1}<br>Passengers: 120/200<br>Status: On Time`);
            });
            
            trainMarker.addEventListener('mouseleave', () => {
                this.hideTooltip();
            });
            
            vizContainer.appendChild(trainMarker);
            
            // Save train data
            this.state.trains.push({
                id: i + 1,
                position: position,
                status: 'running',
                passengers: 120,
                capacity: 200,
                element: trainMarker
            });
        }
    },
    
    // Show tooltip near an element
    showTooltip: function(element, content) {
        let tooltip = document.querySelector('.custom-tooltip');
        
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.className = 'custom-tooltip';
            document.body.appendChild(tooltip);
        }
        
        tooltip.innerHTML = content;
        
        const rect = element.getBoundingClientRect();
        tooltip.style.left = `${rect.left + rect.width/2}px`;
        tooltip.style.top = `${rect.top - tooltip.offsetHeight - 5}px`;
        tooltip.style.display = 'block';
    },
    
    // Hide the tooltip
    hideTooltip: function() {
        const tooltip = document.querySelector('.custom-tooltip');
        if (tooltip) {
            tooltip.style.display = 'none';
        }
    },
    
    // Load initial simulation data
    loadInitialData: function() {
        // This would typically fetch data from the backend
        // For now, we'll use placeholder data
        
        // Generate some sample passenger flow data
        this.state.passengerData = [];
        const stations = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'];
        
        stations.forEach(station => {
            this.state.passengerData.push({
                station: station,
                boarding: Math.floor(Math.random() * 100),
                alighting: Math.floor(Math.random() * 100)
            });
        });
        
        // Generate some sample energy data
        this.state.energyData = [];
        for (let i = 0; i < 8; i++) {
            const hour = 6 + i * 2;
            this.state.energyData.push({
                time: `${hour}:00`,
                energy: 250 + Math.floor(Math.random() * 150)
            });
        }
        
        // Update charts with this data
        if (typeof ChartModule !== 'undefined') {
            ChartModule.updatePassengerFlowChart(this.state.passengerData);
            ChartModule.updateEnergyChart(this.state.energyData);
        }
        
        // Update train status list
        this.updateTrainStatusList();
    },
    
    // Initialize the simulation settings form
    initializeForm: function() {
        const settings = this.state.configuredSettings;
        
        document.getElementById('sim-duration').value = settings.duration;
        document.getElementById('num-trains').value = settings.numTrains;
        document.getElementById('headway').value = settings.headway;
        document.getElementById('passenger-flow').value = settings.passengerFlow;
        document.getElementById('random-delays').checked = settings.randomDelays;
    },
    
    // Start the simulation
    startSimulation: function() {
        if (this.state.paused) {
            // Resume from pause
            this.state.paused = false;
            console.log('Resuming simulation...');
        } else {
            // Start new simulation
            this.state.running = true;
            this.state.simulationTime = 0;
            console.log('Starting simulation...');
            
            // Reset simulation time display
            document.getElementById('sim-time').textContent = '00:00';
            
            // If this were a real simulator, we'd connect to a backend API here
            // Instead, we'll simulate with a simple timer
        }
        
        // Start simulation timer
        this.simulationTimer = setInterval(() => {
            this.simulationStep();
        }, 1000 / this.state.simulationSpeed);
    },
    
    // Pause the simulation
    pauseSimulation: function() {
        if (this.state.running) {
            this.state.paused = true;
            clearInterval(this.simulationTimer);
            console.log('Simulation paused.');
        }
    },
    
    // Stop the simulation
    stopSimulation: function() {
        this.state.running = false;
        this.state.paused = false;
        clearInterval(this.simulationTimer);
        console.log('Simulation stopped.');
    },
    
    // Reset the simulation
    resetSimulation: function() {
        this.stopSimulation();
        this.state.simulationTime = 0;
        document.getElementById('sim-time').textContent = '00:00';
        
        // Reset train positions
        this.setupVisualization();
        
        // Reset data
        this.loadInitialData();
        
        console.log('Simulation reset.');
    },
    
    // Perform a single simulation step
    simulationStep: function() {
        // Increment simulation time (1 minute per step)
        this.state.simulationTime += 1;
        
        // Update simulation time display
        const hours = Math.floor(this.state.simulationTime / 60);
        const minutes = this.state.simulationTime % 60;
        document.getElementById('sim-time').textContent = 
            `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
        
        // Update train positions
        this.updateTrainPositions();
        
        // Update passenger data
        this.updatePassengerData();
        
        // Update energy data
        this.updateEnergyData();
        
        // Update train status list
        this.updateTrainStatusList();
        
        // Check if simulation should end
        if (this.state.simulationTime >= this.state.configuredSettings.duration) {
            this.stopSimulation();
            console.log('Simulation completed.');
            
            // If this were connected to a backend, we'd fetch final results here
            if (typeof SmartRailCloud !== 'undefined') {
                SmartRailCloud.showNotification('Simulation completed');
                SmartRailCloud.updateSimulationStatus('Completed');
            }
        }
    },
    
    // Update train positions during simulation
    updateTrainPositions: function() {
        // In a real implementation, this would use actual simulation data
        // For demonstration, we'll just move trains along the track
        
        this.state.trains.forEach(train => {
            // Move train
            train.position += 0.5;
            
            // Loop back to start when reaching the end
            if (train.position > 90) {
                train.position = 10;
            }
            
            // Update train element position
            if (train.element) {
                train.element.style.left = `${train.position}%`;
                
                // Randomly apply delay status for demo
                if (this.state.configuredSettings.randomDelays && Math.random() < 0.05) {
                    train.element.classList.toggle('delayed');
                }
            }
        });
    },
    
    // Update passenger data during simulation
    updatePassengerData: function() {
        // In a real implementation, this would use actual simulation data
        // For demonstration, we'll adjust values slightly
        
        if (this.state.simulationTime % 10 === 0) { // Update every 10 minutes
            this.state.passengerData.forEach(stationData => {
                stationData.boarding = Math.max(0, stationData.boarding + Math.floor(Math.random() * 21) - 10);
                stationData.alighting = Math.max(0, stationData.alighting + Math.floor(Math.random() * 21) - 10);
            });
            
            // Update chart
            if (typeof ChartModule !== 'undefined') {
                ChartModule.updatePassengerFlowChart(this.state.passengerData);
            }
        }
    },
    
    // Update energy data during simulation
    updateEnergyData: function() {
        // In a real implementation, this would use actual simulation data
        // For demonstration, we'll add a point every hour
        
        if (this.state.simulationTime % 60 === 0) { // Update every hour
            const hour = Math.floor(this.state.simulationTime / 60) + 6; // Starting at 6:00
            this.state.energyData.push({
                time: `${hour}:00`,
                energy: 250 + Math.floor(Math.random() * 150)
            });
            
            // Keep only the last 8 points
            if (this.state.energyData.length > 8) {
                this.state.energyData.shift();
            }
            
            // Update chart
            if (typeof ChartModule !== 'undefined') {
                ChartModule.updateEnergyChart(this.state.energyData);
            }
        }
    },
    
    // Update train status list
    updateTrainStatusList: function() {
        const statusList = document.getElementById('train-status-list');
        statusList.innerHTML = '';
        
        this.state.trains.forEach(train => {
            // Determine train status
            let status = 'On Time';
            let badgeClass = 'bg-success';
            
            if (train.element && train.element.classList.contains('delayed')) {
                status = 'Delayed 2m';
                badgeClass = 'bg-warning';
            }
            
            // Determine if train is at a station or moving
            let location = 'Moving';
            
            // Find closest station
            const closestStation = this.state.stations.reduce((closest, station) => {
                const distance = Math.abs(station.x - train.position);
                return distance < Math.abs(closest.x - train.position) ? station : closest;
            }, { x: 999 });
            
            // If very close to a station, consider it at that station
            if (Math.abs(closestStation.x - train.position) < 2) {
                location = `At ${closestStation.name}`;
            } else {
                // Find the next station in the direction of travel
                const nextStation = this.state.stations.find(s => s.x > train.position) || this.state.stations[0];
                location = `Moving to ${nextStation.name}`;
            }
            
            // Create status item
            const item = document.createElement('div');
            item.className = 'list-group-item';
            item.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div><strong>Train ${train.id}</strong>: ${location}</div>
                    <span class="badge ${badgeClass}">${status}</span>
                </div>
                <div class="small text-muted">Passengers: ${train.passengers}/${train.capacity}</div>
            `;
            
            statusList.appendChild(item);
        });
    },
    
    // Set simulation speed
    setSimulationSpeed: function(speed) {
        this.state.simulationSpeed = speed;
        console.log(`Simulation speed set to ${speed}x`);
        
        // If simulation is running, restart timer with new speed
        if (this.state.running && !this.state.paused) {
            clearInterval(this.simulationTimer);
            this.simulationTimer = setInterval(() => {
                this.simulationStep();
            }, 1000 / this.state.simulationSpeed);
        }
    },
    
    // Apply settings from the form
    applySettings: function() {
        // Get values from form
        const duration = parseInt(document.getElementById('sim-duration').value);
        const numTrains = parseInt(document.getElementById('num-trains').value);
        const headway = parseFloat(document.getElementById('headway').value);
        const passengerFlow = document.getElementById('passenger-flow').value;
        const randomDelays = document.getElementById('random-delays').checked;
        
        // Update settings
        this.state.configuredSettings = {
            duration,
            numTrains,
            headway,
            passengerFlow,
            randomDelays
        };
        
        console.log('Applied new settings:', this.state.configuredSettings);
        
        // If the number of trains changed, reset visualization
        if (numTrains !== this.state.trains.length) {
            // In a real implementation, this would adjust train count properly
            // For demo purposes, we'll just reset the visualization
            this.resetSimulation();
        }
    },
    
    // Refresh the simulator view (called when switching to simulator tab)
    refreshView: function() {
        console.log('Refreshing simulator view...');
        
        // Update visualization if needed
        if (this.state.stations.length === 0) {
            this.setupVisualization();
        }
        
        // Update charts
        if (typeof ChartModule !== 'undefined') {
            ChartModule.updatePassengerFlowChart(this.state.passengerData);
            ChartModule.updateEnergyChart(this.state.energyData);
        }
        
        // Update train status list
        this.updateTrainStatusList();
    }
};
