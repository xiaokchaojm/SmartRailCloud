class Train:
    """
    Represents a train in the rail transit system.
    Each train has a unique ID, passenger capacity, and keeps track of
    its current location, status, and passenger count.
    """

    # Define train status constants
    STATUS_AT_STATION = "AT_STATION"
    STATUS_MOVING = "MOVING"
    STATUS_IDLE = "IDLE"

    def __init__(self, id, capacity=200, max_speed=80):
        """
        Initialize a new train.

        Args:
            id (int): Unique identifier for the train
            capacity (int): Maximum number of passengers the train can hold
            max_speed (float): Maximum speed of the train in km/h
        """
        self.id = id
        self.capacity = capacity
        self.max_speed = max_speed

        # Current state
        self.passengers = 0
        self.current_station = 0  # Index of current station or segment
        self.direction = 1  # 1 for forward, -1 for reverse
        self.status = self.STATUS_AT_STATION

        # Operational metrics
        self.dwell_time = 0  # Time remaining at station in minutes
        self.remaining_travel_time = 0  # Time to reach next station in minutes
        self.energy_consumed = 0  # Total energy consumed in kWh
        self.distance_traveled = 0  # Total distance traveled in km
        self.passengers_served = 0  # Total passengers that have ridden the train

        # Performance tracking
        self.stops_made = 0
        self.total_delay = 0  # Total minutes of delay
        self.current_speed = 0  # Current speed in km/h

    def board_passengers(self, station):
        """
        Board passengers from a station onto the train.

        Args:
            station: The station object where boarding occurs

        Returns:
            int: Number of passengers that boarded
        """
        # The station's board_passengers method handles the logic
        return station.board_passengers(self)

    def alight_passengers(self, station, alight_probability=0.3):
        """
        Alight passengers from the train at a station.

        Args:
            station: The station object where alighting occurs
            alight_probability (float): Probability of passengers alighting

        Returns:
            int: Number of passengers that alighted
        """
        # The station's alight_passengers method handles the logic
        passengers_alighted = station.alight_passengers(self, alight_probability)
        self.passengers_served += passengers_alighted
        return passengers_alighted

    def start_moving(self, travel_time):
        """
        Start moving to the next station.

        Args:
            travel_time (int): Time required to reach the next station in minutes
        """
        self.status = self.STATUS_MOVING
        self.remaining_travel_time = travel_time
        self.current_speed = self.calculate_speed(travel_time)

    def calculate_speed(self, travel_time, segment_distance=None):
        """
        Calculate the current speed based on travel time and distance.

        Args:
            travel_time (int): Travel time in minutes
            segment_distance (float, optional): Distance of current segment in km

        Returns:
            float: Speed in km/h
        """
        # If segment_distance is not provided, use a default value based on max_speed
        if segment_distance is None:
            # Assuming average segment is traversed at max_speed
            segment_distance = (self.max_speed / 60) * travel_time

        # Calculate speed in km/h (distance / time in hours)
        if travel_time > 0:
            speed = (segment_distance / travel_time) * 60  # Convert minutes to hours
            return min(speed, self.max_speed)  # Ensure we don't exceed max_speed
        return 0

    def arrive_at_station(self, station_idx, dwell_time=1):
        """
        Arrive at a station.

        Args:
            station_idx (int): Index of the station
            dwell_time (int): Time to dwell at the station in minutes
        """
        self.current_station = station_idx
        self.status = self.STATUS_AT_STATION
        self.dwell_time = dwell_time
        self.current_speed = 0
        self.stops_made += 1

    def update_position(self, time_delta, line):
        """
        Update the train's position based on elapsed time.

        Args:
            time_delta (int): Time elapsed since last update in minutes
            line: The Line object the train is operating on

        Returns:
            dict: Status update information
        """
        energy_used = 0
        distance_moved = 0

        if self.status == self.STATUS_MOVING:
            # Train is moving between stations
            time_moved = min(time_delta, self.remaining_travel_time)
            self.remaining_travel_time -= time_moved

            # Calculate energy consumption and distance
            # Simplified energy model: energy ~ distance * (1 + passengers/capacity)
            distance_per_minute = self.current_speed / 60  # km per minute
            distance_moved = distance_per_minute * time_moved

            # Energy in kWh, assuming 5 kWh per km baseline
            passenger_factor = 1 + (self.passengers / self.capacity * 0.5)  # Heavier trains use more energy
            energy_used = distance_moved * 5 * passenger_factor

            self.energy_consumed += energy_used
            self.distance_traveled += distance_moved

            # Check if we've arrived at the next station
            if self.remaining_travel_time <= 0:
                # Determine next station based on direction
                next_station_idx = line.get_next_station_idx(self.current_station, self.direction)

                # If we've reached the end of the line, reverse direction
                if next_station_idx is None:
                    self.direction *= -1  # Reverse direction
                    next_station_idx = line.get_next_station_idx(self.current_station, self.direction)

                # Arrive at the station
                standard_dwell_time = 1 + (self.passengers / 50)  # More passengers = longer dwell time
                self.arrive_at_station(next_station_idx, dwell_time=standard_dwell_time)

                return {
                    'status': 'arrived',
                    'station': next_station_idx,
                    'energy_used': energy_used,
                    'distance_moved': distance_moved
                }

        elif self.status == self.STATUS_AT_STATION:
            # Train is at a station
            time_dwelled = min(time_delta, self.dwell_time)
            self.dwell_time -= time_dwelled

            # Minimal energy use while at station (HVAC, lights, etc.)
            energy_used = time_dwelled * 0.5  # 0.5 kWh per minute at station
            self.energy_consumed += energy_used

            # Check if we're ready to depart
            if self.dwell_time <= 0:
                # Find the next station
                next_station_idx = line.get_next_station_idx(self.current_station, self.direction)

                # If we've reached the end of the line, reverse direction
                if next_station_idx is None:
                    self.direction *= -1
                    next_station_idx = line.get_next_station_idx(self.current_station, self.direction)

                # Calculate travel time to next station
                travel_time = line.calculate_travel_time(
                    self.current_station,
                    next_station_idx,
                    avg_speed=self.max_speed * 0.8  # Operating at 80% of max speed
                )

                # Start moving
                self.start_moving(travel_time)

                return {
                    'status': 'departed',
                    'from_station': self.current_station,
                    'next_station': next_station_idx,
                    'travel_time': travel_time,
                    'energy_used': energy_used
                }

        # Return status update
        return {
            'status': self.status,
            'current_position': self.current_station,
            'energy_used': energy_used,
            'distance_moved': distance_moved
        }

    def get_statistics(self):
        """
        Get train statistics.

        Returns:
            dict: Dictionary containing train statistics
        """
        return {
            'id': self.id,
            'passengers': self.passengers,
            'capacity': self.capacity,
            'capacity_utilization': (self.passengers / self.capacity) if self.capacity > 0 else 0,
            'energy_consumed': self.energy_consumed,
            'distance_traveled': self.distance_traveled,
            'passengers_served': self.passengers_served,
            'stops_made': self.stops_made,
            'total_delay': self.total_delay,
            'efficiency': (self.passengers_served / self.energy_consumed) if self.energy_consumed > 0 else 0
        }

    def reset_statistics(self):
        """Reset train statistics for a new simulation run."""
        self.passengers = 0
        self.energy_consumed = 0
        self.distance_traveled = 0
        self.passengers_served = 0
        self.stops_made = 0
        self.total_delay = 0

    def __str__(self):
        status_str = {
            self.STATUS_AT_STATION: f"at station {self.current_station}",
            self.STATUS_MOVING: f"moving to station {self.current_station + self.direction}",
            self.STATUS_IDLE: "idle"
        }

        return (f"Train {self.id}: {status_str.get(self.status, 'unknown')} "
                f"with {self.passengers}/{self.capacity} passengers")