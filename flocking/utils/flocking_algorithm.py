#!/usr/bin/env python
"""
| File: nonlinear_controller.py
| Author: Marcelo Jacinto and Joao Pinto (marcelo.jacinto@tecnico.ulisboa.pt, joao.s.pinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to use the control backends API to create a custom controller 
for the vehicle from scratch and use it to perform a simulation, without using PX4 nor ROS. In this controller, we
provide a quick way of following a given trajectory specified in csv files or track an hard-coded trajectory based
on exponentials! NOTE: This is just an example, to demonstrate the potential of the API. A much more flexible solution
can be achieved
"""

# Imports to be able to log to the terminal with fancy colors
import carb

# Imports from the Pegasus library
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends import Backend

# Auxiliary scipy and numpy modules
import numpy as np
import csv
from scipy.spatial.transform import Rotation
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points

from utils.rotor_control_algorithm import VelocityControl

class FlockingAlgorithm(Backend):
    """A nonlinear controller class. It implements a nonlinear controller that allows a vehicle to track
    aggressive trajectories. This controlers is well described in the papers
    
    [1] J. Pinto, B. J. Guerreiro and R. Cunha, "Planning Parcel Relay Manoeuvres for Quadrotors," 
    2021 International Conference on Unmanned Aircraft Systems (ICUAS), Athens, Greece, 2021, 
    pp. 137-145, doi: 10.1109/ICUAS51884.2021.9476757.
    [2] D. Mellinger and V. Kumar, "Minimum snap trajectory generation and control for quadrotors," 
    2011 IEEE International Conference on Robotics and Automation, Shanghai, China, 2011, 
    pp. 2520-2525, doi: 10.1109/ICRA.2011.5980409.
    """

    def __init__(self, 
        id,
        agents_info,
        Kp=[10.0, 10.0, 10.0],
        Kd=[8.5, 8.5, 8.5],
        Ki=[1.50, 1.50, 1.50],
        Kr=[3.5, 3.5, 3.5],
        Kw=[0.5, 0.5, 0.5]):

        self.velocity_control = VelocityControl(id)
        
        self.id = id
        self.data_storage_file = 'flocking/results/FlockingData/Drone_' + str(id) + '.csv'
        self.agents_info = agents_info

        # The current rotor references [rad/s]
        self.input_ref = [0, 0, 0, 0]

        # The current state of the vehicle expressed in the inertial frame (in ENU)
        self.p = np.zeros((3,))                   # The vehicle position
        self.R: Rotation = Rotation.identity()    # The vehicle attitude
        # self.w = np.zeros((3,))                   # The angular velocity of the vehicle
        self.v = np.zeros((3,))                   # The linear velocity of the vehicle in the inertial frame
        self.a = np.zeros((3,))                   # The linear acceleration of the vehicle in the inertial frame

        # Define the control gains matrix for the outer-loop
        # self.Kp = np.diag(Kp)
        # self.Kd = np.diag(Kd)
        # self.Ki = np.diag(Ki)
        # self.Kr = np.diag(Kr)
        # self.Kw = np.diag(Kw)

        # self.int = np.array([0.0, 0.0, 0.0])

        # Define the dynamic parameters for the vehicle
        self.m = 1.50        # Mass in Kg
        self.g = 9.81       # The gravity acceleration ms^-2

        # Auxiliar variable, so that we only start sending motor commands once we get the state of the vehicle
        self.reveived_first_state = False

        # Lists used for analysing performance statistics
        self.total_time = 0.0
        self.min_distance_to_obstacle = np.inf
        # self.time_vector = []
        # self.desired_position_over_time = []
        # self.position_over_time = []
        # self.position_error_over_time = []
        # self.velocity_error_over_time = []
        # self.atittude_error_over_time = []
        # self.attitude_rate_error_over_time = []

        self.field_names = ["time", "min_distance_to_obstacle", "min_distance_to_other_agent", "max_distance_to_other_agent", "position_x", "position_y", "velocity_x", "velocity_y", "migration_x", "migration_y", "cohesion_x", "cohesion_y", "repulsion_x", "repulsion_y", "friction_x", "friction_y", "obstacle_x", "obstacle_y", "desired_x", "desired_y"]
        with open(self.data_storage_file, 'w+') as csv_file:
            self.csv_writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
            self.csv_writer.writeheader()

    def start(self):
        """
        Reset the control and trajectory index
        """
        

    def stop(self):
        """
        Stopping the controller. Saving the statistics data for plotting later
        """

    def update_sensor(self, sensor_type: str, data):
        """
        Do nothing. For now ignore all the sensor data and just use the state directly for demonstration purposes. 
        This is a callback that is called at every physics step.

        Args:
            sensor_type (str): The name of the sensor providing the data
            data (dict): A dictionary that contains the data produced by the sensor
        """
        pass

    def update_state(self, state: State):
        """
        Method that updates the current state of the vehicle. This is a callback that is called at every physics step

        Args:
            state (State): The current state of the vehicle.
        """
        self.agents_info.set_agent_position(self.id, state.position[0:2])
        self.agents_info.set_agent_velocity(self.id, state.linear_velocity[0:2])

        self.p = state.position
        self.R = Rotation.from_quat(state.attitude)
        # self.w = state.angular_velocity
        self.v = state.linear_velocity

        self.reveived_first_state = True

    def input_reference(self):
        """
        Method that is used to return the latest target angular velocities to be applied to the vehicle

        Returns:
            A list with the target angular velocities for each individual rotor of the vehicle
        """
        # print('out: ',self.input_ref)
        return self.input_ref

    def update(self, dt: float):
        """Method that implements the nonlinear control law and updates the target angular velocities for each rotor. 
        This method will be called by the simulation on every physics step

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """
        
        if self.reveived_first_state == False:
            return

        # -------------------------------------------------
        # Update the references for the controller to track
        # -------------------------------------------------
        euler_angles = self.R.as_euler('zyx', degrees=True)
        yaw = euler_angles[0]
        pitch = euler_angles[1]
        roll = euler_angles[2]
        altitude = self.p[2]
        # print('wys: ',altitude)
        # print("xy velocity: ", self.v[0:2])

        if (self.total_time < 15):
            desired_velocity = [0, 0]
        else:
            desired_velocity = self.compute_desired_velocity()
        
        self.input_ref = self.velocity_control.compute(desired_velocity,0,5,self.v[0:2],yaw,roll,pitch,altitude,dt)

        self.total_time += dt

    def compute_desired_velocity(self):
        my_position = self.agents_info.get_my_position(self.id)
        others_position = self.agents_info.get_other_agents_position(self.id)
        

        my_velocity = self.agents_info.get_my_velocity(self.id)
        others_velocity = self.agents_info.get_other_agents_velocity(self.id)
        
        #PARAMETERS
        v_max = 1.2

        #Migration:
        v_ref = 0.5
        u_ref = np.array([1, -1])
        #----------

        v_mig_term = v_ref * u_ref
        v_coh_term = self.compute_v_coh(my_position, others_position)
        v_rep_term = self.compute_v_rep(my_position, others_position)
        v_frict_term = self.compute_v_frict(my_position, my_velocity, others_position, others_velocity)
        v_obstacle_term = self.compute_v_obstacle_term(my_position, my_velocity)

        
    
        desired_velocity = v_mig_term + v_coh_term + v_rep_term + v_frict_term + v_obstacle_term
        desired_velocity_norm = np.linalg.norm(desired_velocity)

        if (desired_velocity_norm == 0.0):
            desired_velocity_unit_vector = np.array([0, 0])
        else:
            desired_velocity_unit_vector = (desired_velocity / desired_velocity_norm)

        desired_velocity = desired_velocity_unit_vector * np.minimum(desired_velocity_norm, v_max)

        with open(self.data_storage_file, 'a') as csv_file:
            self.csv_writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
            
            row = {
                "time" : self.total_time,
                "min_distance_to_obstacle" : self.min_distance_to_obstacle,
                "min_distance_to_other_agent" : self.agents_info.get_min_distance_to_other_agent(self.id),
                "max_distance_to_other_agent" : self.agents_info.get_max_distance_to_other_agent(self.id),
                "position_x" : my_position[0],
                "position_y" : my_position[1],
                "velocity_x" : my_velocity[0],
                "velocity_y" : my_velocity[1],
                "migration_x" : v_mig_term[0],
                "migration_y" : v_mig_term[1],
                "cohesion_x" : v_coh_term[0],
                "cohesion_y" : v_coh_term[1],
                "repulsion_x" : v_rep_term[0],
                "repulsion_y" : v_rep_term[1],
                "friction_x" : v_frict_term[0],
                "friction_y" : v_frict_term[1],
                "obstacle_x" : v_obstacle_term[0],
                "obstacle_y" : v_obstacle_term[1],
                "desired_x" : desired_velocity[0],
                "desired_y" : desired_velocity[1]
            }

            self.csv_writer.writerow(row)

        return desired_velocity.tolist()
    
    @staticmethod
    def compute_v_coh(ri, others_position):
        #PARAMETERS
        r0_coh = 8
        p_coh = 0.6
        #----------

        vi_coh = np.array([0, 0])
        for rj in others_position.values():
            # print('other_pos', rj)
            rij = np.linalg.norm(ri - rj)

            if (rij > r0_coh):
                vij_coh = p_coh * (r0_coh - rij) * (ri - rj)/rij
            else:
                vij_coh = 0

            vi_coh = vi_coh + vij_coh

        # print('cohesion', vi_coh)
        return vi_coh
    
    @staticmethod
    def compute_v_rep(ri, others_position):
        #PARAMETERS
        r0_rep = 2
        p_rep = 0.8
        #----------

        vi_rep = np.array([0, 0])
        for rj in others_position.values():
            # print('other_pos', rj)
            rij = np.linalg.norm(ri - rj)

            if (rij < r0_rep):
                vij_rep = p_rep * (r0_rep - rij) * (ri - rj)/rij
            else:
                vij_rep = 0

            vi_rep = vi_rep + vij_rep

        # print('repuls', vi_rep)
        return vi_rep
    
    @staticmethod
    def D(r, a, p):
        if(r <= 0):
            return 0
        elif(0 < r*p and r*p < a/p):
            return r*p
        else:
            return np.sqrt(2*a*r - (a**2)/(p**2))

    @staticmethod
    def compute_v_frict(ri, vi, others_position, others_velocity):
        #PARAMETERS
        v_frict = 0.2
        r0_frict = 3
        p_frict = 2
        a_frict = 0.1
        C_frict = 0.2
        #----------
        
        vi_frict = np.array([0, 0])
        for id in others_position.keys():
            rj = others_position[id]
            vj = others_velocity[id]
            rij = np.linalg.norm(ri - rj)
            vij = np.linalg.norm(vi - vj)

            vij_frictmax = np.maximum(v_frict, __class__.D(rij - r0_frict, a_frict, p_frict))
            if(vij > vij_frictmax):
                vij_frict = C_frict * (vij - vij_frictmax) * (vi - vj)/vij
            else:
                vij_frict = 0
            
            vi_frict = vi_frict + vij_frict

        return vi_frict
    
    def compute_v_obstacle_term(self, ri, vi):
        #PARAMETERS
        v_shill = 0.6
        r0_shill = 1
        a_shill = 0.4
        p_shill = 1
        #----------
    
        obstacles_position_and_velocities = self.agents_info.get_obstacles_position_and_velocity(ri, v_shill)

        vi_obstacle = np.array([0, 0])
        min_distance_to_obstacle = np.inf
        for rs, vs in obstacles_position_and_velocities:
            # print('virtuar_agent_vel: ',vs)
            ris = np.linalg.norm(ri - rs)
            if (ris < min_distance_to_obstacle):
                min_distance_to_obstacle = ris
            vis = np.linalg.norm(vi - vs)

            vis_shillmax = self.D(ris - r0_shill, a_shill, p_shill)
            if(vis > vis_shillmax):
                vis_obstacle = (vis - vis_shillmax) * (vs - vi)/vis
            else:
                vis_obstacle = 0

            vi_obstacle = vi_obstacle + vis_obstacle
        
        self.min_distance_to_obstacle = min_distance_to_obstacle
        return vi_obstacle
    

class AgentsInfo:
    def __init__(self, agent_ids):
        self.positions = dict.fromkeys(agent_ids)
        self.velocities = dict.fromkeys(agent_ids)
        self.nearest_obstacle_distance = dict.fromkeys(agent_ids)

        self.obstacles = [
            # self.create_regular_polygon(101, 89, 10, 1),
            # self.create_regular_polygon(106, 94, 10, 1),
            # self.create_regular_polygon(111, 99, 10, 1),
            # self.create_regular_polygon(107.8, 87.2, 10, 1),
            # self.create_regular_polygon(112.8, 92.2, 10, 1),
            # self.create_regular_polygon(109.6, 80.4, 10, 1),
            # self.create_regular_polygon(114.6, 85.4, 10, 1),
            # self.create_regular_polygon(119.6, 90.4, 10, 1),
            # Polygon([(106, 95), (117, 95), (117, 84), (123, 84), (123, 101), (106, 101), (106, 95)]),
            Polygon([(106, 93),(113, 93), (121, 101), (106, 101), (106, 93)]),
            Polygon([(115, 91), (115, 84), (123, 84), (123, 99), (115, 91)]),
            Polygon([(99, 87), (99, 77), (109, 77), (109, 87), (99, 87)]),
        ]

        obstacle_field_names = ["id", "x", "y"]
        with open('flocking/results/ObstaclesData/Obstacles.csv', 'w+') as csv_file:
            obstacles_writer = csv.DictWriter(csv_file, fieldnames=obstacle_field_names)
            obstacles_writer.writeheader()

        with open('flocking/results/ObstaclesData/Obstacles.csv', 'a') as csv_file:
            obstacles_writer = csv.DictWriter(csv_file, fieldnames=obstacle_field_names)
            i = 0

            for polygon in self.obstacles:
                x, y = polygon.exterior.xy
                for x, y in zip(x, y):
                    row = {
                        "id" : i,
                        "x" : x,
                        "y" : y
                    }
                    obstacles_writer.writerow(row)
    
                i += 1

        self.field_names = ["position_x", "position_y",  "virtual_vel_x", "virtual_vel_y"]
        with open('flocking/results/ObstaclesData/VirtualVelocities.csv', 'w+') as csv_file:
            self.csv_writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
            self.csv_writer.writeheader()

    def set_agent_position(self, id, position):
        self.positions[id] = position
    
    def set_agent_velocity(self, id, velocity):
        self.velocities[id] = velocity

    def get_my_position(self, id):
        return self.positions[id]

    def get_my_velocity(self, id):
        return self.velocities[id]

    def get_other_agents_position(self, id):
        return {i:self.positions[i] for i in self.positions if i!=id}
    
    def get_other_agents_velocity(self, id):
        return {i:self.velocities[i] for i in self.velocities if i!=id}
    
    def get_min_distance_to_other_agent(self, id):
        my_position = self.get_my_position(id)

        return np.min([np.linalg.norm(my_position - self.positions[i]) for i in self.positions if i!=id])

    def get_max_distance_to_other_agent(self, id):
        my_position = self.get_my_position(id)

        return np.max([np.linalg.norm(my_position - self.positions[i]) for i in self.positions if i!=id])

    @staticmethod
    def create_regular_polygon(center_x, center_y, num_points, radius):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        vertices_x = center_x + radius * np.cos(angles)
        vertices_y = center_y + radius * np.sin(angles)

        vertices = [(vertices_x[i], vertices_y[i]) for i in range(num_points)]

        polygon = Polygon(vertices)
        return polygon
    
    def get_obstacles_position_and_velocity(self, my_position, v_shill):
        my_point = Point(my_position[0], my_position[1])
        result = []

        for polygon in self.obstacles:
            nearest_point_on_polygon, _ = nearest_points(polygon.exterior, my_point)
            polygon_center = np.array(polygon.centroid.coords[0])

            nearest_point = None
            for i in range(len(polygon.exterior.coords) - 1):
                point = Point([polygon.exterior.coords[i]])
                if point.contains(nearest_point_on_polygon):
                    nearest_point = point
                    break

            if (nearest_point is not None):
                unit_normal_vector = np.array([nearest_point.x - polygon_center[0], nearest_point.y - polygon_center[1]])
                unit_normal_vector = (unit_normal_vector / np.linalg.norm(unit_normal_vector)) * v_shill

                result.append((np.array([nearest_point_on_polygon.x, nearest_point_on_polygon.y]), unit_normal_vector))
                continue


            nearest_edge = None
            for i in range(len(polygon.exterior.coords) - 1):
                edge = LineString([polygon.exterior.coords[i], polygon.exterior.coords[i + 1]])
                if nearest_point_on_polygon.distance(edge) < 1e-8:
                    nearest_edge = edge
                    break

            nearest_edge_first_point_x, nearest_edge_first_point_y = nearest_edge.coords[0]

            vector_edge_to_point = np.array([nearest_point_on_polygon.x - nearest_edge_first_point_x, nearest_point_on_polygon.y - nearest_edge_first_point_y])
            vector_perpendicular_to_edge = np.array([-vector_edge_to_point[1], vector_edge_to_point[0]])

            vector_polygon_center_to_point = np.array([nearest_point_on_polygon.x - polygon_center[0], nearest_point_on_polygon.y - polygon_center[1]])

            dot_product = np.dot(vector_perpendicular_to_edge, vector_polygon_center_to_point)

            if dot_product > 0:
                unit_normal_vector = vector_perpendicular_to_edge / np.linalg.norm(vector_perpendicular_to_edge)
            else:
                unit_normal_vector = -vector_perpendicular_to_edge / np.linalg.norm(vector_perpendicular_to_edge)
            unit_normal_vector = unit_normal_vector * v_shill

            result.append((np.array([nearest_point_on_polygon.x, nearest_point_on_polygon.y]), unit_normal_vector))
            continue
        
        self.write_data(result)
        return result
    

    def write_data(self, position_and_velocity):
        for position, velocity in position_and_velocity:
            with open('flocking/results/ObstaclesData/VirtualVelocities.csv', 'a') as csv_file:
                self.csv_writer = csv.DictWriter(csv_file, fieldnames=self.field_names)

                row = {
                    "position_x" : position[0],
                    "position_y" : position[1],
                    "virtual_vel_x" : velocity[0],
                    "virtual_vel_y" : velocity[1]
                    }
                
                self.csv_writer.writerow(row)
    
