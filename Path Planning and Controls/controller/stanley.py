import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import *
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose, TwistWithCovarianceStamped, Point
from visualization_msgs.msg import Marker,MarkerArray
# from rclpy.duration import Duration
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration


from eufs_msgs.msg import CanState
from eufs_msgs.srv import SetCanState
from eufs_msgs.msg import ConeArrayWithCovariance
from eufs_msgs.msg import CarState
from dv_msgs.msg import Track

from trajectory.submodules_ppc.trajectory_packages import *
from scipy.spatial import Delaunay

# from test_controls.skully import *

PERIOD = 0.05 #20Hz


class eufs_car(Node):
    def __init__(self):

        super().__init__('main')

        # Publihsers
        self.publish_cmd = self.create_publisher(AckermannDriveStamped, '/cmd', 5)
        self.pp_waypoint = self.create_publisher(Marker, '/waypoint', 5)
        # self.viz_cones = self.create_publisher(MarkerArray, '/viz_cones', 1)
        # self.delaunay_viz = self.create_publisher(MarkerArray, '/delaunay', 1)
        self.cones_groundtruth = self.create_subscription(MarkerArray, '/waypoint_array',self.store_waypoints, 1)

        # self.cones_groundtruth = self.create_subscription(ConeArrayWithCovariance, '/ground_truth/track',self.get_map, 1)
        # self.cones_groundtruth = self.create_subscription(ConeArrayWithCovariance, '/ground_truth/cones',self.get_map, 1)
        # self.cones_perception = self.create_subscription(Track, '/perception/cones',self.get_map, 1)
        

        self.carstate_groundtruth = self.create_subscription(CarState, '/ground_truth/state',self.get_carState, 1)
        
        
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
        self.t_start = time.time()
        self.t_runtime = 1000
        self.t_start = time.time()
        self.track_available = False
        self.CarState_available = False
        self.t1 = time.time() - 1
        self.integral = 0
        self.error=0
        self.prev_error = 0
        self.vel_error = 0
        self.id = 0
        self.pp_id = 0
        self.id_line = 0
        self.cube_scale=0.1

        self.current_waypoints = None
        self.store_path_taken = np.array([[0,0]])
        self.waypoints_available = False

        self.carState = CarState()
    
    def store_waypoints(self, msg):

        self.current_waypoints = np.array([[0,0]])
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            self.current_waypoints = np.append(self.current_waypoints, [[x,y]], axis=0)
        self.current_waypoints = self.current_waypoints[1:]

        self.waypoints_available = True

        return None

 
    def get_carState(self, data):
        # print('Car state updated')
        self.carState = data
        self.CarState_available = True
        return None        
    
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
        return None

    def control_callback(self):
       
       # Check track availablility, return if not present
        if (self.waypoints_available and self.CarState_available) == False:
            self.t_start = time.time()
            self.get_logger().info(f'Waypoints Available:{self.waypoints_available} CarState available:{self.CarState_available}')
            return

        # Run Node for limited time 
        if time.time() < self.t_start + self.t_runtime :
            # print('Enter Control loop')

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # Just for storing path, car ground truth not required #
            pos_x = self.carState.pose.pose.position.x
            pos_y = self.carState.pose.pose.position.y
            self.store_path_taken = np.append(self.store_path_taken, [[pos_x,pos_y]], axis = 0)
            q = self.carState.pose.pose.orientation
            car_yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

            v_curr=np.sqrt(self.carState.twist.twist.linear.x**2 + self.carState.twist.twist.linear.y**2)
            
            kp =  0.4
            ki =  0.001
            kd =  0.02
            dt_vel = time.time() - self.t1   #Used to obtain the time difference for PID control.
            self.t1 = time.time()
            
            # closest_waypoint_index=np.argmin((pos_x-self.x)**2+(pos_y-self.y)**2)
            
            [throttle,brake,self.integral,self.vel_error,diffn ] = vel_controller2(kp=kp, ki=ki, kd=kd,
                                                        v_curr=v_curr, v_ref=8,
                                                        dt=dt_vel, prev_integral=self.integral, prev_vel_error=self.vel_error)
            # print('close_index',closest_waypoint_index)
            # print('no. of midpoints',self.midpoints.shape)
            # print('paired',self.paired_indexes)


            [steer_pp, x_p, y_p] = stanley_steering(final_x=self.current_waypoints[:,0],final_y=self.current_waypoints[:,1],v_curr=v_curr,pos_x=pos_x,pos_y=pos_y,car_yaw=car_yaw)
            # print('waypoint',x_p,y_p)
            # self.visualize_cones()
            self.visualize_pp_waypoint(x_pp = x_p,y_pp = y_p)
            # self.visualize_car(x_coord = pos_x, y_coord = pos_y)
            #print('following',x_p,y_p)
            #print('position',pos_x,pos_y)
            #print('steer',steer_pp,'yaw',car_yaw)

                # carControlsmsg.throttle = throttle
                # carControlsmsg.brake = brake
                # carControlsmsg.steering = steer_pp

                # carControls.publish(carControlsmsg)

            # print(f'Steer:{steer_pp}, Accn:{throttle - brake}, Car Yaw:{car_yaw}, Car Pos:{pos_x, pos_y}, PP point:{x_p, y_p}')
            control_msg = AckermannDriveStamped()
            # control_msg.header.stamp = Node.get_clock(self).now().to_msg()
            # control_msg.header.
            print(steer_pp)
            control_msg.drive.steering_angle = steer_pp
            control_msg.drive.acceleration = float(throttle - brake)
            # control_msg.drive.acceleration = 0.05
            self.publish_cmd.publish(control_msg)

            self.get_logger().info(f'Speed:{v_curr:.4f} Accn:{float(throttle - brake):.4f} Steer:{float(-steer_pp):.4f} Time:{time.time() - self.t_start:.4f}')

        else:
            self.get_logger().info(f'Time Finished')
            raise SystemExit
            
        return None
    
    def visualize_pp_waypoint(self, x_pp, y_pp):
        data = Pose()
        data.position.x = float(x_pp)
        data.position.y = float(y_pp)
        pure_pursuit_waypoint_msg = Marker()
        pure_pursuit_waypoint_msg.header.frame_id = 'base_footprint'
        pure_pursuit_waypoint_msg.ns = "Way_ppoint"
        pure_pursuit_waypoint_msg.id = self.pp_id
        self.pp_id += 1
        pure_pursuit_waypoint_msg.type = 1
        pure_pursuit_waypoint_msg.action = 0
        pure_pursuit_waypoint_msg.pose = data
        pure_pursuit_waypoint_msg.scale.x = self.cube_scale
        pure_pursuit_waypoint_msg.scale.y = self.cube_scale
        pure_pursuit_waypoint_msg.scale.z = self.cube_scale
        pure_pursuit_waypoint_msg.color.r = 0.0
        pure_pursuit_waypoint_msg.color.g = 256.0
        pure_pursuit_waypoint_msg.color.b = 256.0
        pure_pursuit_waypoint_msg.color.a = 1.0
        Duration_of_marker = Duration()
        Duration_of_marker.sec = 0
        Duration_of_marker.nanosec = 100000000  #0.1 seconds
        pure_pursuit_waypoint_msg.lifetime = Duration_of_marker
        self.pp_waypoint.publish(pure_pursuit_waypoint_msg)
    
    def visualize_car(self, x_coord, y_coord):
        data = Pose()
        data.position.x = float(x_coord)
        data.position.y = float(y_coord)
        car_location_msg = Marker()
        car_location_msg.header.frame_id = 'base_footprint'
        car_location_msg.ns = "Car's Location"
        car_location_msg.id = self.pp_id
        self.pp_id += 1
        car_location_msg.type = 1
        car_location_msg.action = 0
        car_location_msg.pose = data
        car_location_msg.scale.x = self.cube_scale
        car_location_msg.scale.y = self.cube_scale
        car_location_msg.scale.z = self.cube_scale
        car_location_msg.color.r = 256.0
        car_location_msg.color.g = 256.0
        car_location_msg.color.b = 256.0
        car_location_msg.color.a = 1.0
        Duration_of_marker = Duration()
        Duration_of_marker.sec = 0
        Duration_of_marker.nanosec = 50000000 # .1 seconds
        car_location_msg.lifetime = Duration_of_marker
        self.car_location.publish(car_location_msg)

    def visualize_cones(self):
        
        # print('Viz cones')
        all_cones = MarkerArray()
        
        for cone in self.blue_cones:
            marker = Marker()
            marker.ns = "track"
            marker.type = 2
            marker.action = 0
      
            # if color==0:
            color_cone_r = 0.0
            color_cone_g = 0.0
            color_cone_b = 1.0
            # if color==1:
            #         color_cone_r = 1
            #         color_cone_g = 1
            #         color_cone_b = 0
            # if color==2:
            #         color_cone_r = 1
            #         color_cone_g = 0.27
            #         color_cone_b = 0 
            # if color==3:
            #         color_cone_r = 1
            #         color_cone_g = 0.647
            #         color_cone_b = 0.5
            # if color==4:
            #         color_cone_r = 1
            #         color_cone_g = 1 
            #         color_cone_b = 1
            
            marker.id = self.id
            marker.color.r = color_cone_r
            marker.color.g = color_cone_g
            marker.color.b = color_cone_b
            marker.color.a = 0.5
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.pose.position.x = float(cone[0]) #x
            marker.pose.position.y = float(cone[1]) #y
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            Duration_of_marker = Duration()
            Duration_of_marker.sec = 0
            Duration_of_marker.nanosec = 50000000
            marker.lifetime = Duration_of_marker
            marker.header.frame_id = 'base_footprint'
            all_cones.markers.append(marker)
            self.id += 1
        for cone in self.yellow_cones:
            marker = Marker()
            marker.ns = "track"
            marker.type = 2
            marker.action = 0
      
            # if color==0:
            color_cone_r = 1.0
            color_cone_g = 1.0
            color_cone_b = 0.0
            # if color==1:
            #         color_cone_r = 1
            #         color_cone_g = 1
            #         color_cone_b = 0
            # if color==2:
            #         color_cone_r = 1
            #         color_cone_g = 0.27
            #         color_cone_b = 0 
            # if color==3:
            #         color_cone_r = 1
            #         color_cone_g = 0.647
            #         color_cone_b = 0.5
            # if color==4:
            #         color_cone_r = 1
            #         color_cone_g = 1 
            #         color_cone_b = 1
            
            marker.id = self.id
            marker.color.r = color_cone_r
            marker.color.g = color_cone_g
            marker.color.b = color_cone_b
            marker.color.a = 0.5
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.pose.position.x = float(cone[0]) #x
            marker.pose.position.y = float(cone[1]) #y
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            Duration_of_marker = Duration()
            Duration_of_marker.sec = 0
            Duration_of_marker.nanosec = 50000000
            marker.lifetime = Duration_of_marker
            marker.header.frame_id = 'base_footprint'
            all_cones.markers.append(marker)
            self.id += 1

        # print('Publishing cones on /viz_cones')

        # rate = rclpy.Rate(10)

        
            
        self.viz_cones.publish(all_cones)
            # time.sleep(0.05)

    def visualize_line(self, line_list):

        markerArrayMsg = MarkerArray()
        for pair_of_points in line_list:
            # print(pair_of_points)
            x1,y1,x2,y2 = pair_of_points
            for i in range(0,2): 
                for j in range(i+1,3):    
                    marker = Marker()
                    marker.header.frame_id = "base_footprint"
                    marker.id = self.id_line
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.05  # Line width

                    # Set the points of the line
                    point_1 = Point()
                    point_1.x = x1
                    point_1.y = y1
                    point_1.z = 0.0

                    point_2 = Point()
                    point_2.x = x2
                    point_2.y = y2
                    point_2.z = 0.0
                    marker.points = [point_1,
                                     point_2
                                    ]
                    # print(marker.points)
                    # Set the color (red in this case)
                    marker.color.r = 1.0
                    marker.color.a = 1.0  # Alpha value
                    Duration_of_marker = Duration()
                    Duration_of_marker.sec = 0
                    Duration_of_marker.nanosec = 5000000
                    marker.lifetime = Duration_of_marker  # Permanent marker
                    markerArrayMsg.markers.append(marker)
                    self.id_line += 1
        # rate = rospy.Rate(10)
        # print('hi')
        # print(markerArrayMsg)
        # cones_viz =  rospy.Publisher('/track_lines', MarkerArray, queue_size=10)
        # while not rospy.is_shutdown():
        # self.get_logger().info(f"finsihed viz lines")
        self.delaunay_viz.publish(markerArrayMsg)

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
    ads_dv = eufs_car()
    rclpy.spin(ads_dv)

    ads_dv.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()