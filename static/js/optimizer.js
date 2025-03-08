// static/js/optimizer.js - Optimizer functionality for SmartRailCloud

const OptimizerModule = {
    // Optimizer state
    state: {
        running: false,
        algorithm: 'genetic',
        objective: 'balanced',
        currentGeneration: 0,
        totalGenerations: 30,
        bestScore: 0,
        startTime: null,
        elapsedTime: 0,
        fitnessHistory: [],
        bestSolution: null,
        settings: {
            populationSize: 20,
            generations: 30,
            mutationRate: 0.2,
            crossoverRate: 0.7,
            initialTemperature: 100,
            coolingRate: 0.95,
            maxIterations: 100,
            weights: {
                passengerWaitTime: 0.4,
                energyConsumption: 0.3,
                capacityUtilization: 0.3
            }
        },
        constraints: {
            minHeadway: 3,
            maxHeadway: 15,
            minTrains: 5,
            maxTrains: 15,
            maxEnergyConsumption: null
        }
    },
    
    // Initialize the optimizer module
    init: function() {
        console.log('Initializing optimizer module...');
        
        // Initialize form with current settings
        this.initializeForm();
        
        // Set up event handlers for algorithm-specific fields
        this.setupAlgorithmSpecificFields();
        
        console.log('Optimizer module initialized.');
    },
    
    // Initialize the optimizer settings form
    initializeForm: function() {
        const settings = this.state.settings;
        const constraints = this.state.constraints;
        
        // Algorithm selection
        document.getElementById('opt-algorithm').value = this.state.algorithm;
        
        // Objective selection
        document.getElementById('opt-objective').value = this.state.objective;
        
        // Genetic algorithm parameters
        document.getElementById('population-size').value = settings.populationSize;
        document.getElementById('generations').value = settings.generations;
        document.getElementById('mutation-rate').value = settings.mutationRate;
        document.getElementById('mutation-rate-value').textContent = settings.mutationRate.toFixed(2);
        
        // Evaluation duration
        document.getElementById('opt-duration').value = 120;
        
        // Objective weights
        document.getElementById('weight-wait').value = settings.weights.passengerWaitTime;
        document.getElementById('weight-energy').value = settings.weights.energyConsumption;
        document.getElementById('weight-capacity').value = settings.weights.capacityUtilization;
        
        // Constraints
        document.getElementById('min-headway').value = constraints.minHeadway;
        document.getElementById('max-headway').value = constraints.maxHeadway;
        document.getElementById('min-trains').value = constraints.minTrains;
        document.getElementById('max-trains').value = constraints.maxTrains;
    },
    
    // Set up visibility of algorithm-specific fields
    setupAlgorithmSpecificFields: function() {
        const algorithmSelect = document.getElementById('opt-algorithm');
        const geneticParams = document.getElementById('genetic-params');
        
        // Show/hide fields based on initial algorithm
        if (this.state.algorithm === 'genetic') {
            geneticParams.style.display = 'block';
        } else {
            geneticParams.style.display = 'none';
        }
        
        // Add change handler
        algorithmSelect.addEventListener('change', (e) => {
            if (e.target.value === 'genetic') {
                geneticParams.style.display = 'block';
            } else {
                geneticParams.style.display = 'none';
            }
        });
    },
    
    // Start the optimization process
    startOptimization: function() {
        if (this.state.running) {
            console.log('Optimization already running.');
            return;
        }
        
        console.log('Starting optimization...');
        
        // Reset state
        this.state.running = true;
        this.state.currentGeneration = 0;
        this.state.bestScore = 0;
        this.state.fitnessHistory = [];
        this.state.startTime = new Date();
        this.state.elapsedTime = 0;
        
        // Update UI
        document.getElementById('opt-status-text').textContent = 'Initializing...';
        document.getElementById('opt-progress').style.width = '0%';
        document.getElementById('best-score').textContent = '0.000';
        document.getElementById('opt-time').textContent = '00:00:00';
        
        // In a real implementation, this would connect to a backend API
        // For demonstration, we'll simulate the optimization process
        this.simulateOptimization();
    },
    
    // Stop the optimization process
    stopOptimization: function() {
        if (!this.state.running) {
            return;
        }
        
        console.log('Stopping optimization...');
        this.state.running = false;
        
        // Clear any running timers
        if (this.optimizationTimer) {
            clearTimeout(this.optimizationTimer);
        }
        
        document.getElementById('opt-status-text').textContent = 'Stopped';
    },
    
    // Apply the best solution found
    applyOptimization: function() {
        if (!this.state.bestSolution) {
            console.warn('No optimization solution available to apply');
            return false;
        }
        
        console.log('Applying optimization solution:', this.state.bestSolution);
        
        // In a real application, this would apply the solution to the simulator
        // For demonstration, we'll just update UI elements
        
        // Update the simulator settings if possible
        if (typeof SimulatorModule !== 'undefined') {
            // Set number of trains
            SimulatorModule.state.configuredSettings.numTrains = this.state.bestSolution.numTrains;
            
            // Set headway
            SimulatorModule.state.configuredSettings.headway = this.state.bestSolution.headway;
            
            // Update simulator form
            document.getElementById('num-trains').value = this.state.bestSolution.numTrains;
            document.getElementById('headway').value = this.state.bestSolution.headway;
        }
        
        return true;
    },
    
    // Simulate the optimization process
    simulateOptimization: function() {
        if (!this.state.running) {
            return;
        }
        
        // Get total iterations/generations based on algorithm
        let totalIterations = 30; // default
        
        if (this.state.algorithm === 'genetic') {
            totalIterations = this.state.settings.generations;
        } else if (this.state.algorithm === 'simulated_annealing' || this.state.algorithm === 'hill_climbing') {
            totalIterations = this.state.settings.maxIterations;
        }
        
        // Progress tracking
        const iterationInterval = 500; // milliseconds between iterations
        const progressInterval = 1000; // milliseconds between progress updates
        
        // Function to update progress
        const updateProgress = () => {
            if (!this.state.running) return;
            
            // Update elapsed time
            const now = new Date();
            this.state.elapsedTime = (now - this.state.startTime) / 1000;
            
            // Format elapsed time as HH:MM:SS
            const hours = Math.floor(this.state.elapsedTime / 3600);
            const minutes = Math.floor((this.state.elapsedTime % 3600) / 60);
            const seconds = Math.floor(this.state.elapsedTime % 60);
            document.getElementById('opt-time').textContent = 
                `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            
            // Schedule next update if still running
            if (this.state.running) {
                setTimeout(updateProgress, progressInterval);
            }
        };
        
        // Start progress updates
        updateProgress();
        
        // Function to process an iteration
        const processIteration = () => {
            if (!this.state.running) return;
            
            // Increment generation/iteration
            this.state.currentGeneration++;
            
            // Calculate progress
            const progress = (this.state.currentGeneration / totalIterations) * 100;
            document.getElementById('opt-progress').style.width = `${progress}%`;
            
            // Update status text
            let statusText = '';
            if (this.state.algorithm === 'genetic') {
                statusText = `Generation ${this.state.currentGeneration} of ${totalIterations}`;
            } else {
                statusText = `Iteration ${this.state.currentGeneration} of ${totalIterations}`;
            }
            document.getElementById('opt-status-text').textContent = statusText;
            
            // For demonstration, generate a simulated fitness improvement
            // In reality, this would come from the actual optimization algorithm
            const newBestScore = this.simulateFitnessImprovement();
            this.state.bestScore = newBestScore;
            document.getElementById('best-score').textContent = newBestScore.toFixed(3);
            
            // Record history for chart
            this.state.fitnessHistory.push({
                iteration: this.state.currentGeneration,
                bestFitness: newBestScore,
                avgFitness: newBestScore * 0.8 // Just an approximation for demo
            });
            
            // Update fitness chart
            this.updateFitnessChart();
            
            // Generate a simulated solution when we reach a certain point
            if (this.state.currentGeneration === Math.floor(totalIterations / 2)) {
                this.generateSimulatedSolution();
                this.updateSolutionMetricsChart();
            }
            
            // Check if we're done
            if (this.state.currentGeneration >= totalIterations) {
                this.state.running = false;
                document.getElementById('opt-status-text').textContent = 'Completed';
                
                // Final solution update
                this.generateFinalSolution();
                this.updateSolutionMetricsChart();
                
                if (typeof SmartRailCloud !== 'undefined') {
                    SmartRailCloud.showNotification('Optimization completed successfully');
                    SmartRailCloud.updateOptimizationStatus('Completed');
                }
                
                return;
            }
            
            // Schedule next iteration
            this.optimizationTimer = setTimeout(processIteration, iterationInterval);
        };
        
        // Start the first iteration
        processIteration();
    },
    
    // Simulate fitness improvement curve
    simulateFitnessImprovement: function() {
        // Generate an exponential improvement curve that plateaus
        // Starting with low fitness, rapidly improving, then slowing down
        
        // Maximum possible fitness (asymptote)
        const maxFitness = 0.85;
        
        // Rate of improvement - higher means slower improvement
        const rate = 8;
        
        // Calculate fitness based on current iteration
        const progress = this.state.currentGeneration / this.state.settings.generations;
        const fitness = maxFitness * (1 - Math.exp(-rate * progress));
        
        // Add a small random variation
        const variation = (Math.random() * 0.04) - 0.02;
        
        return Math.min(maxFitness, fitness + variation);
    },
    
    // Generate a simulated solution (mid-optimization)
    generateSimulatedSolution: function() {
        // For demonstration, create a plausible solution
        this.state.bestSolution = {
            numTrains: this.state.constraints.minTrains + 3,
            headway: this.state.constraints.minHeadway + 1.5,
            speedFactor: 1.05,
            waitTimeReduction: 18,
            energyReduction: 8
        };
        
        // Update solution details in UI
        document.getElementById('opt-num-trains').textContent = this.state.bestSolution.numTrains;
        document.getElementById('opt-headway').textContent = `${this.state.bestSolution.headway.toFixed(1)} minutes`;
        document.getElementById('opt-speed').textContent = this.state.bestSolution.speedFactor.toFixed(2);
        document.getElementById('opt-wait-time').textContent = `${this.state.bestSolution.waitTimeReduction}%`;
        document.getElementById('opt-energy').textContent = `${this.state.bestSolution.energyReduction}% reduction`;
    },
    
    // Generate the final solution
    generateFinalSolution: function() {
        // For demonstration, create a better final solution
        this.state.bestSolution = {
            numTrains: 10,
            headway: 4.2,
            speedFactor: 1.05,
            waitTimeReduction: 24,
            energyReduction: 12
        };
        
        // Update solution details in UI
        document.getElementById('opt-num-trains').textContent = this.state.bestSolution.numTrains;
        document.getElementById('opt-headway').textContent = `${this.state.bestSolution.headway.toFixed(1)} minutes`;
        document.getElementById('opt-speed').textContent = this.state.bestSolution.speedFactor.toFixed(2);
        document.getElementById('opt-wait-time').textContent = `${this.state.bestSolution.waitTimeReduction}%`;
        document.getElementById('opt-energy').textContent = `${this.state.bestSolution.energyReduction}% reduction`;
    },
    
    // Update the fitness evolution chart
    updateFitnessChart: function() {
        if (typeof ChartModule !== 'undefined' && this.state.fitnessHistory.length > 0) {
            const labels = this.state.fitnessHistory.map(item => item.iteration);
            const bestFitness = this.state.fitnessHistory.map(item => item.bestFitness);
            const avgFitness = this.state.fitnessHistory.map(item => item.avgFitness);
            
            ChartModule.updateFitnessChart(labels, bestFitness, avgFitness);
        }
    },
    
    // Update the solution metrics chart
    updateSolutionMetricsChart: function() {
        if (typeof ChartModule !== 'undefined' && this.state.bestSolution) {
            // For demonstration, generate some metrics comparing current vs optimized
            // In reality, these would come from actual solution evaluation
            
            const currentMetrics = [0.65, 0.6, 0.7, 0.7, 0.6]; // Placeholder values
            
            // Calculate optimized metrics (improved from current)
            const optimizedMetrics = [
                Math.min(0.95, currentMetrics[0] * (1 + this.state.bestSolution.waitTimeReduction/100)),
                Math.min(0.95, currentMetrics[1] * (1 + this.state.bestSolution.energyReduction/100)),
                Math.min(0.95, currentMetrics[2] * 1.15), // capacity utilization
                Math.min(0.95, currentMetrics[3] * 1.05), // headway compliance
                Math.min(0.95, currentMetrics[4] * 1.3)  // passenger satisfaction
            ];
            
            ChartModule.updateSolutionMetricsChart(currentMetrics, optimizedMetrics);
        }
    },
    
    // Apply settings from the form
    applySettings: function() {
        // Get values from form
        const algorithm = document.getElementById('opt-algorithm').value;
        const objective = document.getElementById('opt-objective').value;
        const populationSize = parseInt(document.getElementById('population-size').value);
        const generations = parseInt(document.getElementById('generations').value);
        const mutationRate = parseFloat(document.getElementById('mutation-rate').value);
        const duration = parseInt(document.getElementById('opt-duration').value);
        
        // Get weights
        const waitWeight = parseFloat(document.getElementById('weight-wait').value);
        const energyWeight = parseFloat(document.getElementById('weight-energy').value);
        const capacityWeight = parseFloat(document.getElementById('weight-capacity').value);
        
        // Update settings
        this.state.algorithm = algorithm;
        this.state.objective = objective;
        this.state.settings.populationSize = populationSize;
        this.state.settings.generations = generations;
        this.state.settings.mutationRate = mutationRate;
        this.state.settings.weights = {
            passengerWaitTime: waitWeight,
            energyConsumption: energyWeight,
            capacityUtilization: capacityWeight
        };
        
        console.log('Applied new optimizer settings:', this.state);
    },
    
    // Apply constraints from the form
    applyConstraints: function() {
        // Get values from form
        const minHeadway = parseFloat(document.getElementById('min-headway').value);
        const maxHeadway = parseFloat(document.getElementById('max-headway').value);
        const minTrains = parseInt(document.getElementById('min-trains').value);
        const maxTrains = parseInt(document.getElementById('max-trains').value);
        const enforceMaxEnergy = document.getElementById('enforce-max-energy').checked;
        
        // Update constraints
        this.state.constraints = {
            minHeadway,
            maxHeadway,
            minTrains,
            maxTrains,
            maxEnergyConsumption: enforceMaxEnergy ? 5000 : null // Example value
        };
        
        console.log('Applied new constraints:', this.state.constraints);
    },
    
    // Refresh the optimizer view (called when switching to optimizer tab)
    refreshView: function() {
        console.log('Refreshing optimizer view...');
        
        // Update fitness chart if we have data
        if (this.state.fitnessHistory.length > 0) {
            this.updateFitnessChart();
        }
        
        // Update solution metrics chart if we have a solution
        if (this.state.bestSolution) {
            this.updateSolutionMetricsChart();
        }
    }
};
