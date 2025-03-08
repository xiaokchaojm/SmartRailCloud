class Line:
    """
    Represents a transit line in the rail system.
    A line consists of a series of stations and the distances between them.
    """

    def __init__(self, stations, distances):
        """
        Initialize a new transit line.

        Args:
            stations (list): List of Station objects along this line
            distances (list): List of distances between adjacent stations (in km)
                              Length should be len(stations) - 1
        """
        self.stations = stations
        self.distances = distances

        # Validate input
        if len(distances) != len(stations) - 1:
            raise ValueError("Number of distances must be one less than number of stations")

    def get_station_by_id(self, station_id):
        """
        Get a station by its ID.

        Args:
            station_id (int): The ID of the station to find

        Returns:
            Station: The station with the given ID, or None if not found
        """
        for station in self.stations:
            if station.id == station_id:
                return station
        return None

    def get_distance(self, from_station_idx, to_station_idx):
        """
        Get the distance between two stations.

        Args:
            from_station_idx (int): Index of the starting station
            to_station_idx (int): Index of the destination station

        Returns:
            float: Distance between the stations in km
        """
        # Ensure indices are valid
        if from_station_idx < 0 or from_station_idx >= len(self.stations) or \
                to_station_idx < 0 or to_station_idx >= len(self.stations):
            raise ValueError("Invalid station indices")

        # Calculate distance
        distance = 0
        if from_station_idx < to_station_idx:
            # Forward direction
            for i in range(from_station_idx, to_station_idx):
                distance += self.distances[i]
        else:
            # Reverse direction
            for i in range(to_station_idx, from_station_idx):
                distance += self.distances[i]

        return distance

    def calculate_travel_time(self, from_station_idx, to_station_idx, avg_speed=40):
        """
        Calculate the travel time between two stations.

        Args:
            from_station_idx (int): Index of the starting station
            to_station_idx (int): Index of the destination station
            avg_speed (float): Average train speed in km/h (default: 40 km/h)

        Returns:
            int: Travel time in minutes
        """
        # Get distance between stations
        distance = self.get_distance(from_station_idx, to_station_idx)

        # Calculate travel time (distance / speed * 60 minutes)
        travel_time = int((distance / avg_speed) * 60)

        return travel_time

    def get_next_station_idx(self, current_idx, direction=1):
        """
        Get the index of the next station in the specified direction.

        Args:
            current_idx (int): Current station index
            direction (int): 1 for forward, -1 for backward

        Returns:
            int: Index of the next station, or None if at the end of line
        """
        next_idx = current_idx + direction

        if next_idx < 0 or next_idx >= len(self.stations):
            return None

        return next_idx

    def get_total_line_length(self):
        """
        Calculate the total length of the line.

        Returns:
            float: Total length in kilometers
        """
        return sum(self.distances)

    def update_all_stations(self, time_delta):
        """
        Update passenger counts at all stations.

        Args:
            time_delta (int): Time elapsed since last update in minutes

        Returns:
            dict: Dictionary with station IDs as keys and new passenger counts as values
        """
        results = {}
        for station in self.stations:
            new_passengers = station.update_passengers(time_delta)
            results[station.id] = new_passengers
        return results

    def get_line_statistics(self):
        """
        Get statistics for the entire line.

        Returns:
            dict: Dictionary containing line statistics
        """
        total_waiting = 0
        total_served = 0
        total_waiting_time = 0

        # Gather statistics from all stations
        station_stats = []
        for station in self.stations:
            stats = station.get_statistics()
            station_stats.append(stats)

            total_waiting += stats['waiting_passengers']
            total_served += stats['passengers_served']
            total_waiting_time += stats['avg_waiting_time'] * stats['passengers_served']

        # Calculate line-wide average waiting time
        avg_waiting_time = 0
        if total_served > 0:
            avg_waiting_time = total_waiting_time / total_served

        return {
            'station_statistics': station_stats,
            'total_waiting_passengers': total_waiting,
            'total_passengers_served': total_served,
            'average_waiting_time': avg_waiting_time,
            'total_length': self.get_total_line_length()
        }

    def reset_all_stations(self):
        """Reset statistics for all stations in the line."""
        for station in self.stations:
            station.reset_statistics()

    def __str__(self):
        return f"Line with {len(self.stations)} stations, total length: {self.get_total_line_length()} km"