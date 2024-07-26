
import os
import matplotlib.pyplot as plt
import numpy as np
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import *
from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Trigger


from eufs_msgs.msg import CanState
from eufs_msgs.srv import SetCanState
from eufs_msgs.msg import ConeArrayWithCovariance
from eufs_msgs.msg import CarState

from .submodules_ppc.trajectory_packages import *

import scipy
from scipy.optimize import minimize

# from test_controls.skully import *

PERIOD = 1


class eufs_car(Node):
    def __init__(self):

        super().__init__('main')

        # Publihsers
        self.publish_cmd = self.create_publisher(AckermannDriveStamped, '/cmd', 5)
        self.cones_groundtruth = self.create_subscription(ConeArrayWithCovariance, '/ground_truth/track',self.get_map, 1)
        self.carstate_groundtruth = self.create_subscription(CarState, '/ground_truth/state',self.get_carState, 1)
        # self.cones_groundtruth.destroy()
        
        self.states = {CanState.AS_OFF: "OFF",
                       CanState.AS_READY: "READY",
                       CanState.AS_DRIVING: "DRIVING",
                       CanState.AS_EMERGENCY_BRAKE: "EMERGENCY",
                       CanState.AS_FINISHED: "FINISHED"}

        # Autonomous missions
        self.missions = {CanState.AMI_NOT_SELECTED: "NOT_SELECTED",
                         CanState.AMI_ACCELERATION: "ACCELERATION",
                         CanState.AMI_SKIDPAD: "SKIDPAD",
                         CanState.AMI_AUTOCROSS: "AUTOCROSS",
                         CanState.AMI_TRACK_DRIVE: "TRACK_DRIVE",
                         CanState.AMI_AUTONOMOUS_DEMO: "AUTONOMOUS_DEMO",
                         CanState.AMI_ADS_INSPECTION: "ADS_INSPECTION",
                         CanState.AMI_ADS_EBS: "ADS_EBS",
                         CanState.AMI_DDT_INSPECTION_A: "DDT_INSPECTION_A",
                         CanState.AMI_DDT_INSPECTION_B: "DDT_INSPECTION_B",
                         CanState.AMI_JOYSTICK: "JOYSTICK",
                         }

        # Services
        self.ebs_srv = self.create_client(Trigger, "/ros_can/ebs")
        self.reset_srv = self.create_client(Trigger, "/ros_can/reset")
        self.set_mission_cli = self.create_client(SetCanState, "/ros_can/set_mission")
        self.reset_vehicle_pos_srv = self.create_client(Trigger,
                                                             "/ros_can/reset_vehicle_pos")
        self.reset_cone_pos_srv = self.create_client(Trigger,
                                                          "/ros_can/reset_cone_pos")

        # Timers
        self.timer = self.create_timer(PERIOD, self.control_callback)
        
        # Misc
        self.setManualDriving()

        # Attributes
        self.t_runtime = 60
        self.track_available = False
        self.t1 = time.time() - 1
        self.integral = 0
        self.error=0
        self.prev_error = 0
        self.vel_error = 0
        
    def get_map(self,data):
        '''
        store the map of all the cones.
        Track, blue cones, yellow cones:
        small_track, 37, 30
        hairpin_increasing_difficulty, 490, 490 --equal number of cones
        its_a_mess, 60, 72
        garden_light, 101, 99
        peanut, 54, 64
        boa_constrictor, 44, 51
        rectangle, 40, 52
        comp_2021, 156, 156  --equal number of cones
        '''
        self.get_logger().info(f'Blue:{len(data.blue_cones)} Yellow:{len(data.yellow_cones)}')
        if len(data.blue_cones) != len(data.yellow_cones):
            self.get_logger().info(f'Exiting, for now Raceline generation require same number of cone on each side')
        blue_cones = []
        yellow_cones = []
        for cone in data.blue_cones:
            blue_cones.append([cone.point.x, cone.point.y])
        for cone in data.yellow_cones:
            yellow_cones.append([cone.point.x, cone.point.y])
        
        self.blue_cones = np.array(blue_cones)
        self.yellow_cones = np.array(yellow_cones)
        self.cones_groundtruth.destroy()  # So that /ground/track is subscribed only once

        self.get_waypoints()  #to generate the optimized trajectory - the raceline

        return None
    def get_carState(self, data):
        self.carState = data
    
    def timer_callback(self):
        control_msg = AckermannDriveStamped()
        # control_msg.header.stamp = Node.get_clock(self).now().to_msg()
        # control_msg.header.
        control_msg.drive.steering_angle = 0.0
        control_msg.drive.steering_angle_velocity = 0.0
        control_msg.drive.speed = 0.0
        control_msg.drive.acceleration = 0.5
        control_msg.drive.jerk = 0.0
        self.publish_cmd.publish(control_msg)

    def control_callback(self):

        if self.track_available == False:
            self.t_start = time.time()
            return


        if time.time() < self.t_start + self.t_runtime:
            self.get_logger().info('Enter Control loop')

            pos_x = self.carState.pose.pose.position.x
            pos_y = self.carState.pose.pose.position.y
            q = self.carState.pose.pose.orientation
            v_curr=np.sqrt(self.carState.twist.twist.linear.x**2 + self.carState.twist.twist.linear.y**2)

            car_yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)

            kp =  0.4
            ki =  0.00
            kd =  0.02
            dt_vel = time.time() - self.t1   #Used to obtain the time difference for PID control.
            self.t1 = time.time()
            
            closest_waypoint_index=np.argmin((pos_x-self.x)**2+(pos_y-self.y)**2)
            
            [throttle,brake,self.integral,self.vel_error,diffn ] = vel_controller2(kp=kp, ki=ki, kd=kd,
                                                        v_curr=v_curr, v_ref=self.v[closest_waypoint_index],
                                                        dt=dt_vel, prev_integral=self.integral, prev_vel_error=self.vel_error)
            # self.get_logger().info('close_index',closest_waypoint_index)
            # self.get_logger().info('no. of midpoints',self.midpoints.shape)
            # self.get_logger().info('paired',self.paired_indexes)

            [steer_pp, x_p, y_p] = pure_pursuit(x=self.x, y=self.y, vf=v_curr, pos_x=pos_x, pos_y=pos_y, veh_head=car_yaw,pos=closest_waypoint_index)


            self.vizulaize_pp_waypoint(x_pp=x_p,y_pp=y_p)
            #self.get_logger().info('following',x_p,y_p)
            #self.get_logger().info('position',pos_x,pos_y)
            #self.get_logger().info('steer',steer_pp,'yaw',car_yaw)

                # carControlsmsg.throttle = throttle
                # carControlsmsg.brake = brake
                # carControlsmsg.steering = steer_pp

                # carControls.publish(carControlsmsg)

            self.get_logger().info(f'Steer:{steer_pp}, Acceleration:{throttle - brake}')
            control_msg = AckermannDriveStamped()
            # control_msg.header.stamp = Node.get_clock(self).now().to_msg()
            # control_msg.header.
            control_msg.drive.steering_angle = float(steer_pp)
            control_msg.drive.steering_angle_velocity = 0.0
            control_msg.drive.speed = 0.0
            control_msg.drive.acceleration = float(throttle - brake)
            control_msg.drive.jerk = 0.0
            self.publish_cmd.publish(control_msg)

    def get_waypoints(self):
        '''
        Takes in and stores the Track information provided by fsds through a latched (only publishes once) ros topic. 
        '''
        self.get_logger().info('Generating optimal waypoints')
        bounds_left=evaluate_bezier(self.blue_cones,3)
        bounds_right=evaluate_bezier(self.yellow_cones,3)
        # bounds_right=bounds_right[:,0:2]
        # bounds_left=bounds_left[:,0:2]

        def totalcurvature(params,bounds_left=bounds_left,bounds_right=bounds_right):
            [x,y]=genpath2(bounds_left=bounds_left,bounds_right=bounds_right,params=params)
            # plt.plot(x,y)
            # plt.pause(0.05)
            dx_dt = np.gradient(x)
            dy_dt = np.gradient(y)
            
            d2x_dt2 = np.gradient(dx_dt)
            d2y_dt2 = np.gradient(dy_dt)

            curvature = ((d2x_dt2 * dy_dt - dx_dt * d2y_dt2) /(dx_dt * dx_dt + dy_dt * dy_dt)**1.5)
            objective=0
            for c in curvature:
                objective=objective+c**2
            
            # self.get_logger().info(objective)
            return objective



        params=np.zeros(len(bounds_left))

        for i in range(len(params)):
            params[i]=0.5

        # Calculate time for midline trajectory
        [x_mid,y_mid]=genpath2(bounds_left=bounds_left,bounds_right=bounds_right,params=params)
        [k_mid, r_mid] = curvature(x_mid, y_mid)
        _, midline_expected_lap_time,_,_,_ = vel_find3(x_mid,y_mid, mu = 0.6, m = 230, g = 9.8, k =k_mid, r = r_mid)


        result=minimize(fun=totalcurvature,x0=params,bounds=scipy.optimize.Bounds(lb=0.0, ub=1.00, keep_feasible=False))
        params=result.x
        self.get_logger().info(f'Result: Success:{result.success} Fun:{result.fun}')

        # self.get_logger().info(params)
        [x,y]=genpath2(bounds_left=bounds_left,bounds_right=bounds_right,params=params)
   
        self.x=np.array(x)
        self.y=np.array(y)
        # self.get_logger().info(self.x,self.y)

        [self.k, self.r] = curvature(x, y)
    
        v2, raceline_expected_lap_time,v1,r,self.v = vel_find3(self.x, self.y, mu = 0.6, m = 230, g = 9.8, k = self.k, r = self.r)

        self.get_logger().info(f'Raceline Time :{raceline_expected_lap_time:.4f} Midline Time:{midline_expected_lap_time:.4f}')

        self.track_available = True

        plt.plot(bounds_left[:,0],bounds_left[:,1], c = 'blue')
        plt.plot(bounds_right[:,0],bounds_right[:,1], c = 'yellow')
        plt.plot(self.x, self.y, c = 'green')
        plt.show()

        plt.plot(v2, c = 'green' )
        plt.plot(v1, c = 'black' )
        plt.plot(self.v, c = 'blue' )
        plt.show()
        
        if raceline_expected_lap_time > midline_expected_lap_time:
            self.get_logger().info(f'Raising SystemExit, Optimization not improving time')
            raise SystemExit

        self.t_start = time.time()


        return None

    def sendRequest(self, mission_ami_state):
        """Sends a mission request to the simulated ros_can
        The mission request is of message type eufs_msgs/srv/SetCanState
        where only the ami_state field is used.
        """
        if self.set_mission_cli.wait_for_service(timeout_sec=1):
            request = SetCanState.Request()
            request.ami_state = mission_ami_state
            result = self.set_mission_cli.call_async(request)
            # self.node.get_logger().debug("Mission request sent successfully")
            # self.node.get_logger().debug(result)
        else:
            # self.node.get_logger().warn(
            #     "/ros_can/set_mission service is not available")
            self.get_logger().warn(
                "/ros_can/set_mission service is not available")

    def setMission(self, mission):
        """Requests ros_can to set mission"""
        # mission = self._widget.findChild(
        #     QComboBox, "MissionSelectMenu").currentText()

        # self.node.get_logger().debug(
        #     "Sending mission request for " + str(mission))

        # create message to be sent
        mission_msg = CanState()

        # find enumerated mission and set
        for enum, mission_name in self.missions.items():
            if mission_name == mission:
                mission_msg.ami_state = enum
                break
        # mission_msg.ami_state = CanState.AMI_SKIDPAD
        self.sendRequest(mission_msg.ami_state)

    def setManualDriving(self):
        self.get_logger().debug("Sending manual mission request")
        mission_msg = CanState()
        mission_msg.ami_state = CanState.AMI_MANUAL
        self.sendRequest(mission_msg.ami_state)

    def resetState(self):
        """Requests state_machine reset"""
        self.node.get_logger().debug("Requesting state_machine reset")

        if self.reset_srv.wait_for_service(timeout_sec=1):
            request = Trigger.Request()
            result = self.reset_srv.call_async(request)
            self.node.get_logger().debug("state reset successful")
            self.node.get_logger().debug(result)
        else:
            self.node.get_logger().warn(
                "/ros_can/reset service is not available")

    def resetVehiclePos(self):
        """Requests race car model position reset"""
        self.node.get_logger().debug(
            "Requesting race_car_model position reset")

        if self.reset_vehicle_pos_srv.wait_for_service(timeout_sec=1):
            request = Trigger.Request()
            result = self.reset_vehicle_pos_srv.call_async(request)
            self.node.get_logger().debug("Vehicle position reset successful")
            self.node.get_logger().debug(result)
        else:
            self.node.get_logger().warn(
                "/ros_can/reset_vehicle_pos service is not available")

    def resetConePos(self):
        """Requests gazebo_cone_ground_truth to reset cone position"""
        self.node.get_logger().debug(
            "Requesting gazebo_cone_ground_truth cone position reset")

        if self.reset_cone_pos_srv.wait_for_service(timeout_sec=1):
            request = Trigger.Request()
            result = self.reset_cone_pos_srv.call_async(request)
            self.node.get_logger().debug("Cone position reset successful")
            self.node.get_logger().debug(result)
        else:
            self.node.get_logger().warn(
                "/ros_can/reset_cone_pos service is not available")

    def resetSim(self):
        """Requests state machine, vehicle position and cone position reset"""
        self.node.get_logger().debug("Requesting Simulation Reset")

        # Reset State Machine
        self.resetState()

        # Reset Vehicle Position
        self.resetVehiclePos()

        # Reset Cone Position
        self.resetConePos()

    def requestEBS(self):
        """Requests ros_can to go into EMERGENCY_BRAKE state"""
        self.node.get_logger().debug("Requesting EBS")

        if self.ebs_srv.wait_for_service(timeout_sec=1):
            request = Trigger.Request()
            result = self.ebs_srv.call_async(request)
            self.node.get_logger().debug("EBS successful")
            self.node.get_logger().debug(result)
        else:
            self.node.get_logger().warn(
                "/ros_can/ebs service is not available")

    def stateCallback(self, msg):
        """Reads the robot state from the message
        and displays it within the GUI

        Args:
            msg (eufs_msgs/CanState): state of race car
        """
        if msg.ami_state == CanState.AMI_MANUAL:
            self._widget.findChild(QLabel, "StateDisplay").setText(
                "Manual Driving")
            self._widget.findChild(QLabel, "MissionDisplay").setText(
                "MANUAL")
        else:
            self._widget.findChild(QLabel, "StateDisplay").setText(
                self.states[msg.as_state])
            self._widget.findChild(QLabel, "MissionDisplay").setText(
                self.missions[msg.ami_state])

    def shutdown_plugin(self):
        """stop all publisher, subscriber and services
        necessary for clean shutdown"""
        assert (self.node.destroy_client(
            self.set_mission_cli)), "Mission client could not be destroyed"
        assert (self.node.destroy_subscription(
            self.state_sub)), "State subscriber could not be destroyed"
        assert (self.node.destroy_client(
            self.ebs_srv)), "EBS client could not be destroyed"
        assert (self.node.destroy_client(
            self.reset_srv)), "State reset client could not be destroyed"
        # Note: do not destroy the node in shutdown_plugin as this could
        # cause errors for the Robot Steering GUI. Let ROS 2 clean up nodes
    

def main(args=None):

    rclpy.init(args = args)
    car4_56 = eufs_car()
    rclpy.spin(car4_56)

    car4_56.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()