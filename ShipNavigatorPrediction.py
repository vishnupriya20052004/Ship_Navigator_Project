import requests
from geopy.geocoders import Nominatim
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
import logging
import random
import time
from functools import lru_cache
from groq import Groq
import threading
from math import radians, sin, cos, sqrt, atan2
import tkinter as tk
from tkinter import messagebox
import folium

# Replace these with your actual API keys
WEATHER_API_KEY = "1ebf275de5f049e4f85f859659046525"
GROQ_API_KEY = "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"

# Configure logging
logging.basicConfig(filename='marine_optimization.log', level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class WeatherData:
    location: str
    temperature: float
    feels_like: float
    description: str
    wind_speed: float
    wind_direction: float
    humidity: int
    pressure: float
    visibility: float
    timestamp: datetime

@dataclass
class Ship:
    ship_type: str
    max_speed: float
    fuel_consumption: float
    safety_rating: float
    fuel_capacity: float  # in liters or tons
    vessel_weight: float  # in tons

class MarineWeatherAnalyzer:
    def _init_(self):
        """Initialize the analyzer with API clients and caching"""
        self.weather_api_key = WEATHER_API_KEY
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.geolocator = Nominatim(
            user_agent="marine_optimization_v1.0",
            timeout=5
        )
        
    @lru_cache(maxsize=100)
    def get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates with caching for faster repeated lookups"""
        location = location.lower().strip()

        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logging.info(f"Attempting to geocode location: {location} (attempt {attempt + 1})")
                    loc = self.geolocator.geocode(location)
                    if loc:
                        coords = (loc.latitude, loc.longitude)
                        logging.info(f"Successfully geocoded location: {location}")
                        return coords
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"Failed to geocode after {max_retries} attempts: {e}")
                        raise
                    logging.warning(f"Geocoding attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
            logging.warning(f"Could not find coordinates for location: {location}")
            return None
        except Exception as e:
            logging.error(f"Error fetching coordinates: {e}")
            return None

    def fetch_weather_data(self, lat: float, lon: float) -> Optional[WeatherData]:
        """Fetch and parse marine weather data with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = (
                    f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"
                )
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                location = data.get('name', 'Open Water')
                if not location or location == '':
                    location = f"Coordinates: {lat:.2f}°N, {lon:.2f}°E"
                
                weather_data = WeatherData(
                    location=location,
                    temperature=data['main']['temp'],
                    feels_like=data['main']['feels_like'],
                    description=data['weather'][0]['description'],
                    wind_speed=data['wind'].get('speed', 0),
                    wind_direction=data['wind'].get('deg', 0),
                    humidity=data['main']['humidity'],
                    pressure=data['main']['pressure'],
                    visibility=data.get('visibility', 0) / 1000,  # Convert to km
                    timestamp=datetime.fromtimestamp(data['dt'])
                )
                logging.info(f"Successfully fetched weather data for {location}")
                return weather_data

            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to fetch weather data after {max_retries} attempts: {e}")
                    return None
                logging.warning(f"Weather data fetch attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            except KeyError as e:
                logging.error(f"Error parsing weather data: {e}")
                return None

    def ant_colony_optimization(self, graph, start, end, ship: Ship, weather_data: WeatherData, num_ants=10, max_iterations=100, alpha=1, beta=2, evaporation_rate=0.5):
        """Implement Ant Colony Optimization with fuel, weight, and weather considerations"""
        pheromones = np.ones_like(graph, dtype=float)
        best_route = None
        best_cost = float('inf')
        
        convergence_count = 0
        prev_best_cost = float('inf')
        
        for iteration in range(max_iterations):
            routes = []
            for ant in range(num_ants):
                route = self.construct_route(graph, start, end, pheromones, alpha, beta, ship, weather_data)
                cost = self.calculate_cost(route, graph, ship, weather_data)
                routes.append((route, cost))

                if cost < best_cost:
                    best_route = route
                    best_cost = cost
                    convergence_count = 0
                elif cost == prev_best_cost:
                    convergence_count += 1
                    
                if convergence_count >= 10:
                    logging.info(f"ACO converged after {iteration + 1} iterations")
                    return best_route, best_cost

            prev_best_cost = best_cost
            self.update_pheromones(pheromones, routes, evaporation_rate)

        return best_route, best_cost

    def construct_route(self, graph, start, end, pheromones, alpha, beta, ship: Ship, weather_data: WeatherData):
        """Construct a route for an ant considering fuel, weight, and weather"""
        route = [start]
        current = start
        visited = {start}

        while current != end:
            neighbors = [n for n in range(len(graph)) if graph[current][n] != 0 and n not in visited]
            if not neighbors:
                break
                
            probabilities = self.calculate_probabilities(current, neighbors, pheromones, graph, alpha, beta, ship, weather_data)
            next_node = random.choices(neighbors, probabilities)[0]
            route.append(next_node)
            visited.add(next_node)
            current = next_node

        return route

    def calculate_probabilities(self, current, neighbors, pheromones, graph, alpha, beta, ship: Ship, weather_data: WeatherData):
        """Calculate transition probabilities considering fuel, weight, and weather"""
        probabilities = []
        total = 0

        for neighbor in neighbors:
            pheromone = pheromones[current][neighbor] ** alpha
            heuristic = (1 / self.calculate_edge_cost(current, neighbor, graph, ship, weather_data)) ** beta
            probability = pheromone * heuristic
            probabilities.append(probability)
            total += probability

        return [p / total if total > 0 else 1 / len(neighbors) for p in probabilities]

    def calculate_edge_cost(self, current, neighbor, graph, ship: Ship, weather_data: WeatherData):
        """Calculate the cost of an edge considering fuel, weight, and weather"""
        base_cost = graph[current][neighbor]
        
        # Adjust cost based on weather conditions
        weather_factor = 1 + (weather_data.wind_speed / 10)  # Higher wind speed increases cost
        if weather_data.visibility < 5:  # Low visibility increases cost
            weather_factor *= 1.5
        
        # Adjust cost based on fuel consumption and vessel weight
        fuel_factor = (ship.fuel_consumption * base_cost) / ship.fuel_capacity
        weight_factor = ship.vessel_weight / 1000  # Heavier vessels have higher costs
        
        return base_cost * weather_factor * (1 + fuel_factor + weight_factor)

    def calculate_cost(self, route, graph, ship: Ship, weather_data: WeatherData):
        """Calculate route cost considering fuel, weight, and weather"""
        return sum(self.calculate_edge_cost(route[i], route[i + 1], graph, ship, weather_data) for i in range(len(route) - 1))

    def update_pheromones(self, pheromones, routes, evaporation_rate):
        """Update pheromone levels"""
        pheromones *= (1 - evaporation_rate)
        for route, cost in routes:
            deposit = 1.0 / cost if cost > 0 else 1.0
            for i in range(len(route) - 1):
                pheromones[route[i]][route[i + 1]] += deposit

    def generate_route_summary(self, route, weather_data: WeatherData, ship: Ship) -> str:
        """Generate a concise route summary with key details"""
        try:
            optimized_distance = len(route) - 1  # Assuming distance is based on route length
            summary = f"""
            *Optimized Route Summary*
            - Start: {route[0]}
            - End: {route[-1]}
            - Total Distance: {optimized_distance} nautical miles
            
            *Vessel Details*
            - Fuel Capacity: {ship.fuel_capacity} liters/tons
            - Vessel Weight: {ship.vessel_weight} tons
            
            *Recommendations*
            - Maintain optimal speed for fuel efficiency.
            - Monitor fuel consumption closely.
            - Be prepared for potential weather changes.
            """
            return summary.strip()
        except Exception as e:
            logging.error(f"Error generating route summary: {e}")
            return "Error generating route summary. Please check the logs for details."

    def check_weather_and_optimize(self, current_coords, ship: Ship, optimized_route, initial_fuel_capacity):
        """Check weather every minute, optimize path if necessary, and update fuel capacity."""
        distance_traveled = 0  # Initialize distance traveled
        while True:
            time.sleep(60)  # Wait for 1 minute
            weather_data = self.fetch_weather_data(current_coords[0], current_coords[1])
            
            if not weather_data:
                logging.warning("Could not fetch weather data during the journey.")
                continue
            
            # Check for dangerous weather conditions
            if weather_data.wind_speed > 10 or weather_data.visibility < 2:  # Example danger conditions
                logging.warning("Dangerous weather conditions detected! Finding an alternative route...")
                print("Finding an alternative optimized path...")
            else:
                print("Weather is safe. Continue on the current path.")
            
            # Recalculate the optimal route based on the new weather data
            graph = np.array([  # Sample graph, replace with real data
                [0, 1, 2, 0],
                [1, 0, 3, 4],
                [2, 3, 0, 5],
                [0, 4, 5, 0]
            ])
            optimized_route, cost = self.ant_colony_optimization(
                graph, start=0, end=3, ship=ship, weather_data=weather_data
            )
            print(f"New Optimized Route: {optimized_route} with cost: {cost:.2f} nautical miles")
            
            # Update distance traveled (assuming a constant speed for simplicity)
            distance_traveled += ship.max_speed / 60  # Speed in nautical miles per minute
            
            # Update fuel capacity
            remaining_fuel = self.update_fuel_capacity(initial_fuel_capacity, distance_traveled, ship)
            print(f"Remaining Fuel Capacity: {remaining_fuel:.2f} liters/tons")
            
            # Check if fuel is below a certain threshold
            if remaining_fuel <= 0:
                print("Warning: Fuel capacity is critically low!")
                break  # Exit the loop or implement further logic

    def haversine_distance(self, coord1, coord2):
        """Calculate the great-circle distance between two points on the Earth."""
        R = 6371.0  # Radius of the Earth in kilometers
        lat1, lon1 = radians(coord1[0]), radians(coord1[1])
        lat2, lon2 = radians(coord2[0]), radians(coord2[1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)*2 + cos(lat1) * cos(lat2) * sin(dlon / 2)*2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c  # Distance in kilometers

    def calculate_eta(self, start_coords, end_coords, ship: Ship) -> float:
        """Calculate the estimated time of arrival based on the start and end coordinates and ship conditions."""
        distance_km = self.haversine_distance(start_coords, end_coords)  # Get distance in kilometers
        
        # Convert distance to nautical miles (1 km = 0.539957 nautical miles)
        distance_nautical_miles = distance_km * 0.539957
        
        # Calculate speed adjustment based on wind speed
        wind_adjustment = 1 + (ship.max_speed * 0.1 * (ship.fuel_consumption / ship.fuel_capacity))  # Example adjustment
        effective_speed = ship.max_speed / wind_adjustment  # Adjusted speed considering wind and fuel

        # Calculate time in hours
        time_hours = distance_nautical_miles / effective_speed if effective_speed > 0 else float('inf')
        return time_hours

    def calculate_fuel_consumption(self, distance_nautical_miles: float, ship: Ship) -> float:
        """Calculate the fuel consumed based on distance and ship specifications."""
        return distance_nautical_miles * ship.fuel_consumption

    def update_fuel_capacity(self, initial_fuel_capacity: float, distance_nautical_miles: float, ship: Ship) -> float:
        """Update the fuel capacity based on distance traveled."""
        fuel_consumed = self.calculate_fuel_consumption(distance_nautical_miles, ship)
        return max(0, initial_fuel_capacity - fuel_consumed)  # Ensure fuel doesn't go below 0

    def generate_report(self, start_location, end_location, optimized_route, weather_data: WeatherData, eta_hours, remaining_fuel, vessel_details):
        """Generate a report of the voyage and save it to a text file."""
        report_content = f"""
        === Marine Voyage Report ===
        
        Start Location: {start_location}
        End Location: {end_location}
        
        Vessel Details:
        - Ship Type: {vessel_details.ship_type}
        - Max Speed: {vessel_details.max_speed} knots
        - Fuel Capacity: {vessel_details.fuel_capacity} liters/tons
        - Vessel Weight: {vessel_details.vessel_weight} tons
        
        Optimized Route: {optimized_route}
        
        Weather Conditions:
        - Location: {weather_data.location}
        - Temperature: {weather_data.temperature}°C
        - Wind Speed: {weather_data.wind_speed} m/s
        - Visibility: {weather_data.visibility} km
        
        Estimated Time of Arrival: {eta_hours:.2f} hours
        Remaining Fuel Capacity: {remaining_fuel:.2f} liters/tons
        
        Thank you for using the Marine Route Optimizer!
        """
        
        with open("marine_voyage_report.txt", "w") as report_file:
            report_file.write(report_content.strip())
        print("Report saved as 'marine_voyage_report.txt'.")

    def create_map(self, start_coords, end_coords, sea_route_waypoints):
        """Create a map with the optimized sea route and save it as an HTML file."""
        # Create a map centered around the start location
        m = folium.Map(location=start_coords, zoom_start=7)

        # Add markers for start and end locations
        folium.Marker(start_coords, tooltip='Start Location', icon=folium.Icon(color='green')).add_to(m)
        folium.Marker(end_coords, tooltip='End Location', icon=folium.Icon(color='red')).add_to(m)

        # Add the sea route as a polyline
        folium.PolyLine(locations=sea_route_waypoints, color='blue', weight=5, opacity=0.7).add_to(m)

        # Optionally, add weather markers along the route
        for waypoint in sea_route_waypoints:
            folium.Marker(waypoint, tooltip='Waypoint', icon=folium.Icon(color='orange')).add_to(m)

        # Save the map to an HTML file
        m.save("marine_route_map.html")
        print("Map saved as 'marine_route_map.html'.")

def submit():
    start_location = start_entry.get()
    end_location = end_entry.get()
    fuel_capacity = fuel_entry.get()
    vessel_weight = weight_entry.get()
    
    # Call the main function with these parameters
    main(start_location, end_location, fuel_capacity, vessel_weight)

def main(start_location, end_location, fuel_capacity, vessel_weight):
    ship = None  # Initialize ship variable
    try:
        print("Initializing MarineWeatherAnalyzer...")
        analyzer = MarineWeatherAnalyzer()
        
        print("Fetching coordinates...")
        start_coords = analyzer.get_coordinates(start_location)
        end_coords = analyzer.get_coordinates(end_location)

        if not start_coords or not end_coords:
            print("Error: Could not find coordinates for the specified locations.")
            return

        print("Fetching weather data for the starting location...")
        weather_data = analyzer.fetch_weather_data(start_coords[0], start_coords[1])
        if not weather_data:
            print("Error: Could not fetch weather data.")
            return

        print("Calculating optimal route...")
        # Create a simple distance-based graph (replace with real maritime data)
        graph = np.array([  # Sample graph, replace with real data
            [0, 1, 2, 0],
            [1, 0, 3, 4],
            [2, 3, 0, 5],
            [0, 4, 5, 0]
        ])

        # Collect additional ship details
        ship = Ship(
            ship_type="Cargo",
            max_speed=20,
            fuel_consumption=0.1,
            safety_rating=0.9,
            fuel_capacity=float(fuel_capacity),
            vessel_weight=float(vessel_weight)
        )

        # Pass ship and weather data to ACO
        optimized_route, cost = analyzer.ant_colony_optimization(
            graph, start=0, end=3, ship=ship, weather_data=weather_data
        )
        
        print("\n=== Results ===")
        print(f"Start Location: {start_location} {start_coords}")
        print(f"End Location: {end_location} {end_coords}")
        print(f"Optimized Route: {optimized_route}")
        print(f"Total Cost: {cost:.2f} nautical miles")

        # Calculate ETA based on the coordinates
        eta_hours = analyzer.calculate_eta(start_coords, end_coords, ship)
        print(f"Estimated Time of Arrival: {eta_hours:.2f} hours")

        # Start a thread to check weather and optimize path
        current_coords = start_coords  # Update this as the vessel moves
        weather_thread = threading.Thread(target=analyzer.check_weather_and_optimize, args=(current_coords, ship, optimized_route, ship.fuel_capacity))
        weather_thread.start()

        print("\n=== Weather Conditions ===")
        print(f"Location: {weather_data.location}")
        print(f"Temperature: {weather_data.temperature}°C")
        print(f"Wind Speed: {weather_data.wind_speed} m/s")
        print(f"Visibility: {weather_data.visibility} km")

        print("\n=== Generating Route Analysis ===")
        route_summary = analyzer.generate_route_summary(optimized_route, weather_data, ship)
        print("\n🌊 Route Summary and Recommendations:")
        print(route_summary)

        # Define waypoints for the sea route
        sea_route_waypoints = [
            (34.0522, -118.2437),  # Los Angeles
            (34.5, -118.5),        # Waypoint 1
            (35.0, -119.0),        # Waypoint 2
            (35.5, -119.5),        # Waypoint 3
            (36.1699, -115.1398)   # Las Vegas
        ]

        # Create and save the map visualization
        analyzer.create_map(start_coords, end_coords, sea_route_waypoints)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")
        print(f"\nAn unexpected error occurred. Please check the logs for details.")
    finally:
        if ship:  # Check if ship is defined before accessing it
            remaining_fuel = ship.fuel_capacity  # You may want to track this more accurately
            analyzer.generate_report(start_location, end_location, optimized_route, weather_data, eta_hours, remaining_fuel, ship)

# Create the main window for the GUI
root = tk.Tk()
root.title("Marine Route Optimizer")

# Create and place labels and entries
tk.Label(root, text="Start Location:").grid(row=0)
tk.Label(root, text="End Location:").grid(row=1)
tk.Label(root, text="Fuel Capacity (liters/tons):").grid(row=2)
tk.Label(root, text="Vessel Weight (tons):").grid(row=3)

start_entry = tk.Entry(root)
end_entry = tk.Entry(root)
fuel_entry = tk.Entry(root)
weight_entry = tk.Entry(root)

start_entry.grid(row=0, column=1)
end_entry.grid(row=1, column=1)
fuel_entry.grid(row=2, column=1)
weight_entry.grid(row=3, column=1)

# Submit button
submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.grid(row=4, columnspan=2)

# Run the application
root.mainloop()

if __name__ == "_main_":
    # Test with hardcoded values
    main("Los Angeles", "Las Vegas", "1000", "50")

test_start_coords = (34.0522, -118.2437)  # Los Angeles
test_end_coords = (36.1699, -115.1398)  # Las Vegas
test_route = [0, 1, 2]  # Example indices or points
analyzer = MarineWeatherAnalyzer() 
analyzer.create_map(test_start_coords, test_end_coords, test_route)

# Example waypoints for a maritime route (latitude, longitude)
sea_route_waypoints = [
    (34.0522, -118.2437),  # Los Angeles
    (34.5, -118.5),        # Waypoint 1
    (35.0, -119.0),        # Waypoint 2
    (35.5, -119.5),        # Waypoint 3
    (36.1699, -115.1398)   # Las Vegas
]