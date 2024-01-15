#!/usr/bin/env python
import csv

class VelocityControl:
    def __init__(self, id):
        self.roll_PID = PID(10, 0.6, 0.45) #X ZN: 11.12893, 3.4
        self.pitch_PID = PID(4, 0.6, 0.45) #Y ZN: 11.12893, 3.1
        self.rotor_control = RotorControl()

        self.total_time = 0
        self.data_storage_file = 'flocking/results/RotorControlData/Drone_' + str(id) + '.csv'
        
        self.field_names = ["time", "roll", "pitch", "yaw", "altitude", "linear_velocity_x", "linear_velocity_y", "desired_roll", "desired_pitch", "desired_yaw", "desired_altitude", "desired_linear_velocity_x", "desired_linear_velocity_y"]
        with open(self.data_storage_file, 'w+') as csv_file:
            self.csv_writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
            self.csv_writer.writeheader()
    
    def compute(self, 
            desired_xy_velocity, 
            desired_yaw, 
            desired_altitude,
            xy_velocity, 
            yaw, 
            roll,
            pitch,
            altitude, 
            dt):
        
        self.total_time += dt

        velocity_x_error = desired_xy_velocity[0] - xy_velocity[0]
        velocity_y_error = desired_xy_velocity[1] - xy_velocity[1]
        # print("x_error: ", velocity_x_error)
        # print("y_error: ", velocity_y_error)

        pitch_to_apply = self.pitch_PID.compute(velocity_y_error, dt)
        roll_to_apply = self.roll_PID.compute(velocity_x_error, dt)

        if (pitch_to_apply >= 20):
            pitch_to_apply = 20
        if (pitch_to_apply <= -20):
            pitch_to_apply = -20
        if (roll_to_apply >= 20):
            roll_to_apply =20
        if (roll_to_apply <= -20):
            roll_to_apply = -20
        
        with open(self.data_storage_file, 'a') as csv_file:
            self.csv_writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
            
            row = {
                "time" : self.total_time,
                "roll" : roll,
                "pitch" : pitch,
                "yaw" : yaw,
                "altitude": altitude,
                "linear_velocity_x": xy_velocity[0],
                "linear_velocity_y": xy_velocity[1],
                "desired_roll" : roll_to_apply,
                "desired_pitch" : pitch_to_apply,
                "desired_yaw" : desired_yaw,
                "desired_altitude" : desired_altitude,
                "desired_linear_velocity_x" : desired_xy_velocity[0],
                "desired_linear_velocity_y" : desired_xy_velocity[1]
            }

            self.csv_writer.writerow(row)

        return self.rotor_control.compute(
            roll_to_apply, 
            pitch_to_apply, 
            desired_yaw,
            desired_altitude,
            roll,
            pitch,
            yaw,
            altitude,
            dt)

class RotorControl:
    def __init__(self):
        self.MMA_roll_PID = PID(10, 4, 4) #OK (10, 4, 4)
        self.MMA_pitch_PID = PID(10, 5, 6) #OK (10, 5, 5)
        self.MMA_yaw_PID = PID(10, 3, 6) #ZN: 0.53, 3.93 OK: (10, 3, 6)
        self.MMA_thrust_PID = PID(120, 68.181818, 52.8) # ZN: 200, 3.52 PID(120, 68.181818, 52.8)

    @staticmethod
    def MMA(roll, pitch, yaw, thrust):
        # print('roll: ', roll, 'pitch: ', pitch, 'yaw: ', yaw, 'thrust: ', thrust)
        motor_1 = thrust - pitch - yaw - roll
        motor_2 = thrust + pitch - yaw + roll
        motor_3 = thrust - pitch + yaw + roll
        motor_4 = thrust + pitch + yaw - roll

        return [motor_1, motor_2, motor_3, motor_4]

    def compute(self, 
            desired_roll, 
            desired_pitch, 
            desired_yaw, 
            desired_altitude, 
            roll,
            pitch,
            yaw, 
            altitude, 
            dt):

        roll_error = desired_roll - roll
        pitch_error = desired_pitch - pitch
        yaw_error = desired_yaw - yaw
        thrust_error = desired_altitude - altitude

        output_roll = self.MMA_roll_PID.compute(roll_error, dt)
        output_pitch = self.MMA_pitch_PID.compute(pitch_error, dt)
        output_yaw = self.MMA_yaw_PID.compute(yaw_error, dt)
        output_thrust = self.MMA_thrust_PID.compute(thrust_error, dt)

        return self.MMA(output_roll, output_pitch, output_yaw, output_thrust)


class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt != 0 else 0

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.prev_error = error

        return output

