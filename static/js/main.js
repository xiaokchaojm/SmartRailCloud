// static/js/main.js - Main JavaScript functionality for SmartRailCloud

// Main application object
const SmartRailCloud = {
    // Application state
    state: {
        currentView: 'dashboard',
        simulationRunning: false,
        optimizationRunning: false,
        currentTime: new Date(),
        systemData: {
            activeTrains: 0,
            passengersServed: 0,
            avgWaitTime: 0,
            energyEfficiency: 0
        }
    },
    
    // Initialize the application
    init: function() {
        console.log('Initializing SmartRailCloud application...');
        
        // Set up navigation
        this.setupNavigation();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Start clock
        this.startClock();
        
        // Initialize components
        this.initializeComponents();
        
        console.log('SmartRailCloud initialization complete.');
    },
    
    // Set up navigation between sections
    setupNavigation: function() {
        const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
        
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Get target section id from href
                const targetId = link.getAttribute('href').substring(1);
                
                // Hide all sections
                document.querySelectorAll('section').forEach(section => {
                    section.style.display = 'none';
                });
                
                // Show target section
                const targetSection = document.getElementById(targetId);
                if (targetSection) {
                    targetSection.style.display = 'block';
                    
                    // Update active nav link
                    navLinks.forEach(l => l.classList.remove('active'));
                    link.classList.add('active');
                    
                    // Update current view
                    this.state.currentView = targetId;
                    
                    // Trigger view-specific initialization if needed
                    this.handleViewChange(targetId);
                }
            });
        });
        
        // Show dashboard by default
        document.querySelectorAll('section').forEach(section => {
            if (section.id !== 'dashboard') {
                section.style.display = 'none';
            }
        });
    },
    
    // Handle view change
    handleViewChange: function(viewId) {
        switch(viewId) {
            case 'simulator':
                // Initialize simulator view if needed
                SimulatorModule.refreshView();
                break;
            case 'optimizer':
                // Initialize optimizer view if needed
                OptimizerModule.refreshView();
                break;
            case 'reports':
                // Load reports data if needed
                this.loadReportsData();
                break;
            default:
                // Dashboard - update with latest data
                this.updateDashboard();
                break;
        }
    },
    
    // Set up global event listeners
    setupEventListeners: function() {
        // Settings button
        document.querySelector('a[href="#settings"]').addEventListener('click', (e) => {
            e.preventDefault();
            this.showSettingsModal();
        });
        
        // Start simulation button
        document.getElementById('start-sim').addEventListener('click', () => {
            SimulatorModule.startSimulation();
            this.state.simulationRunning = true;
            this.updateSimulationStatus('Running');
        });
        
        // Pause simulation button
        document.getElementById('pause-sim').addEventListener('click', () => {
            SimulatorModule.pauseSimulation();
            this.updateSimulationStatus('Paused');
        });
        
        // Stop simulation button
        document.getElementById('stop-sim').addEventListener('click', () => {
            SimulatorModule.stopSimulation();
            this.state.simulationRunning = false;
            this.updateSimulationStatus('Stopped');
        });
        
        // Reset simulation button
        document.getElementById('reset-sim').addEventListener('click', () => {
            SimulatorModule.resetSimulation();
            this.updateSimulationStatus('Ready');
        });
        
        // Start optimization button
        document.getElementById('start-opt').addEventListener('click', () => {
            OptimizerModule.startOptimization();
            this.state.optimizationRunning = true;
            this.updateOptimizationStatus('Running');
        });
        
        // Stop optimization button
        document.getElementById('stop-opt').addEventListener('click', () => {
            OptimizerModule.stopOptimization();
            this.state.optimizationRunning = false;
            this.updateOptimizationStatus('Stopped');
        });
        
        // Apply optimization button
        document.getElementById('apply-opt').addEventListener('click', () => {
            const result = OptimizerModule.applyOptimization();
            if (result) {
                this.showNotification('Optimization applied successfully');
            }
        });
        
        // Simulation speed change
        document.getElementById('sim-speed').addEventListener('change', (e) => {
            SimulatorModule.setSimulationSpeed(parseInt(e.target.value));
        });
        
        // Apply simulation settings
        document.getElementById('apply-sim-settings').addEventListener('click', () => {
            SimulatorModule.applySettings();
            this.showNotification('Simulation settings applied');
        });
        
        // Apply optimization settings
        document.getElementById('apply-opt-settings').addEventListener('click', () => {
            OptimizerModule.applySettings();
            this.showNotification('Optimization settings applied');
        });
        
        // Apply constraints
        document.getElementById('apply-constraints').addEventListener('click', () => {
            OptimizerModule.applyConstraints();
            this.showNotification('Constraints applied');
        });
        
        // Generate report button
        document.getElementById('generate-report').addEventListener('click', () => {
            this.generateReport();
        });
        
        // Export buttons
        document.getElementById('export-csv').addEventListener('click', () => {
            this.exportData('csv');
        });
        
        document.getElementById('export-json').addEventListener('click', () => {
            this.exportData('json');
        });
    },
    
    // Start a clock to update current time
    startClock: function() {
        setInterval(() => {
            this.state.currentTime = new Date();
            const timeStr = this.state.currentTime.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            document.getElementById('current-time').textContent = timeStr;
        }, 1000);
    },
    
    // Initialize application components
    initializeComponents: function() {
        // Initialize simulator module
        if (typeof SimulatorModule !== 'undefined') {
            SimulatorModule.init();
        }
        
        // Initialize optimizer module
        if (typeof OptimizerModule !== 'undefined') {
            OptimizerModule.init();
        }
        
        // Initialize charts
        if (typeof ChartModule !== 'undefined') {
            ChartModule.initializeCharts();
        }
        
        // Load initial dashboard data
        this.updateDashboard();
    },
    
    // Update dashboard with latest data
    updateDashboard: function() {
        // In a real application, this would fetch data from the backend
        // For now, we'll use placeholder data
        
        // Update statistics
        document.getElementById('active-trains').textContent = this.state.systemData.activeTrains || 8;
        document.getElementById('passengers-served').textContent = this.formatNumber(this.state.systemData.passengersServed || 4521);
        document.getElementById('avg-wait-time').textContent = `${this.state.systemData.avgWaitTime || 4.2} min`;
        document.getElementById('energy-efficiency').textContent = `${this.state.systemData.energyEfficiency || 86}%`;
        
        // System status could be updated here
    },
    
    // Update simulation status display
    updateSimulationStatus: function(status) {
        const statusElement = document.getElementById('sim-status');
        let badgeClass = 'bg-secondary';
        
        switch(status) {
            case 'Running':
                badgeClass = 'bg-success';
                break;
            case 'Paused':
                badgeClass = 'bg-warning';
                break;
            case 'Stopped':
                badgeClass = 'bg-danger';
                break;
            case 'Ready':
                badgeClass = 'bg-secondary';
                break;
        }
        
        statusElement.innerHTML = `<span class="badge ${badgeClass}">${status}</span>`;
    },
    
    // Update optimization status display
    updateOptimizationStatus: function(status) {
        const statusElement = document.getElementById('opt-status');
        let badgeClass = 'bg-secondary';
        
        switch(status) {
            case 'Running':
                badgeClass = 'bg-primary';
                break;
            case 'Completed':
                badgeClass = 'bg-success';
                break;
            case 'Stopped':
                badgeClass = 'bg-danger';
                break;
            case 'Ready':
                badgeClass = 'bg-secondary';
                break;
        }
        
        statusElement.innerHTML = `<span class="badge ${badgeClass}">${status}</span>`;
    },
    
    // Load reports data
    loadReportsData: function() {
        // This would typically fetch data from the backend
        console.log('Loading reports data...');
        
        // For demo purposes, we'll just ensure the charts are initialized
        if (typeof ChartModule !== 'undefined') {
            ChartModule.initializeReportCharts();
        }
    },
    
    // Show settings modal
    showSettingsModal: function() {
        // In a real application, this would display a settings modal
        alert('Settings functionality would be shown here.');
    },
    
    // Generate a report
    generateReport: function() {
        // In a real application, this would generate a detailed report
        console.log('Generating report...');
        this.showNotification('Report generated successfully');
    },
    
    // Export data in specified format
    exportData: function(format) {
        // In a real application, this would export data in the specified format
        console.log(`Exporting data in ${format} format...`);
        this.showNotification(`Data exported in ${format.toUpperCase()} format`);
    },
    
    // Show a notification message
    showNotification: function(message) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'alert alert-info alert-dismissible fade show';
        notification.style.position = 'fixed';
        notification.style.top = '20px';
        notification.style.right = '20px';
        notification.style.zIndex = '9999';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Add to document
        document.body.appendChild(notification);
        
        // Remove after delay
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 150);
        }, 3000);
    },
    
    // Format number with commas
    formatNumber: function(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }
};

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    SmartRailCloud.init();
});
