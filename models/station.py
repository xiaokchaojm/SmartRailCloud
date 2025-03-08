class Station:
    """
    Represents a station in the rail transit system.
    Each station has a unique ID, name, passenger arrival rate,
    and keeps track of waiting passengers.
    """

    def __init__(self, id, name, passenger_rate):
        """
        Initialize a new station.

        Args:
            id (int): Unique identifier for the station
            name (str): Human-readable name of the station
            passenger_rate (float): Average number of passengers arriving per minute
        """
        self.id = id
        self.name = name
        self.passenger_rate = passenger_rate  # passengers per minute
        self.waiting_passengers = 0
        self.total_waiting_time = 0
        self.passengers_served = 0

    class Passenger:
        def __init__(self, origin_time, origin_station, destination):
            self.wait_time = 0  # 累计等待时间（分钟）
            self.origin_time = origin_time  # 生成时间
            self.origin_station = origin_station
            self.destination = destination

        def update_wait(self, time_step):
            self.wait_time += time_step

    def update_passengers(self, time_delta):
        """
        Update the number of waiting passengers based on the arrival rate.

        Args:
            time_delta (int): Time elapsed since last update in minutes

        Returns:
            int: Number of new passengers that arrived
        """
        # Calculate how many new passengers arrived during this time period
        # We use Poisson distribution to model randomness in passenger arrivals
        import random
        from math import exp

        # Simple approximation of Poisson distribution
        lambda_val = self.passenger_rate * time_delta
        new_passengers = int(random.gauss(lambda_val, lambda_val ** 0.5))
        new_passengers = max(0, new_passengers)  # Ensure non-negative

        self.waiting_passengers += new_passengers
        self.total_waiting_time += self.waiting_passengers * time_delta  # Accumulate waiting time

        return new_passengers

    def board_passengers(self, train):
        """
        Move passengers from the station to the train.

        Args:
            train: The train object that is boarding passengers

        Returns:
            int: Number of passengers that boarded the train
        """
        # Calculate how many passengers can board based on train capacity
        available_capacity = train.capacity - train.passengers
        passengers_to_board = min(self.waiting_passengers, available_capacity)

        # Update station state
        self.waiting_passengers -= passengers_to_board
        self.passengers_served += passengers_to_board

        # Update train state
        train.passengers += passengers_to_board

        return passengers_to_board

    def alight_passengers(self, train, alight_probability=0.3):
        """
        Move passengers from the train to the station (alighting).

        Args:
            train: The train object that is alighting passengers
            alight_probability (float): Probability of a passenger alighting at this station

        Returns:
            int: Number of passengers that alighted from the train
        """
        # Calculate how many passengers will alight at this station
        passengers_to_alight = int(train.passengers * alight_probability)

        # Update train state
        train.passengers -= passengers_to_alight

        return passengers_to_alight

    def get_statistics(self):
        """
        Get station statistics.

        Returns:
            dict: Dictionary containing station statistics
        """
        avg_waiting_time = 0
        if self.passengers_served > 0:
            avg_waiting_time = self.total_waiting_time / self.passengers_served

        return {
            'id': self.id,
            'name': self.name,
            'waiting_passengers': self.waiting_passengers,
            'passengers_served': self.passengers_served,
            'avg_waiting_time': avg_waiting_time
        }

    def reset_statistics(self):
        """Reset station statistics for a new simulation run."""
        self.waiting_passengers = 0
        self.total_waiting_time = 0
        self.passengers_served = 0

    def __str__(self):
        return f"Station {self.id}: {self.name} (Waiting: {self.waiting_passengers})"