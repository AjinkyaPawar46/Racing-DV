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
from eufs_msgs.msg import PointArray

# from dv_msgs.msg import Track

from trajectory.submodules_ppc.trajectory_packages import *
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
# from test_controls.skully import *

PERIOD = 0.05 #20Hz

class path_planner(Node):
    def __init__(self):

        super().__init__('main')

        # Publihsers
        self.publish_cmd = self.create_publisher(AckermannDriveStamped, '/cmd', 5)
        self.pp_waypoint = self.create_publisher(Marker, '/waypoint', 5)
        self.car_location = self.create_publisher(Marker, '/Car_location', 5)
        self.viz_cones = self.create_publisher(MarkerArray, '/viz_cones', 1)
        self.delaunay_viz = self.create_publisher(MarkerArray, '/delaunay', 1)
        self.waypoint_path = self.create_publisher(Float32MultiArray, '/waypoint_array', 1)

        # self.cones_groundtruth = self.create_subscription(ConeArrayWithCovariance, '/ground_truth/track',self.get_map, 1)
        self.cones_groundtruth = self.create_subscription(ConeArrayWithCovariance, '/ground_truth/cones',self.get_map, 1)
        # self.cones_perception = self.create_subscription(Track, '/perception/cones',self.get_map, 1)
        self.timer = self.create_timer(PERIOD, self.send_waypoints)

        # self.carstate_groundtruth = self.create_subscription(CarState, '/ground_truth/state',self.get_carState, 1)

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
        # self.timer = self.create_timer(PERIOD, self.control_callback)
        
        # Misc
        self.setManualDriving()

        # Attributes
        self.t1=time.time()-1  #variable to be used to calculate dt loop. This is the actual dt after inducing 
        self.t_start=time.time()
        self.integral = 0
        self.blue_cone_IDs=[]
        self.yellow_cone_IDs=[]
        self.carPose = Pose()
        self.error=0
        self.prev_error = 0
        self.vel_error = 0
        self.cube_scale=0.1
        self.conearray = MarkerArray()

        self.speed = TwistWithCovarianceStamped()
        self.data_from_mrptstate=MarkerArray()
        self.paired_indexes=np.array([[0,0]])
        self.const_velocity= 2
        self.I = 0
        self.pp_id=0
        self.id = 0
        self.track_available = False
        self.all_detected_blue_cones=np.array([[0,0,0]])
        self.all_detected_yellow_cones=np.array([[0,0,0]])
        self.paired_indexes=np.array([[0,0]])
        self.midpoints=np.array([[0,0]])
        
        self.mid_x=[]
        self.mid_y=[]
        self.theta_p=[]
        self.final_x=[]
        self.final_y=[]
        self.steering_angle = 0
        self.v_curr = 0

        self.stop_mid_x=[]        
        self.stop_mid_y=[]
        self.orange_mid_x =[]
        self.orange_mid_y =[]
        self.str_x=[]
        self.str_y=[]
        self.lt_x=[]
        self.lt_y=[]
        self.rt_x=[]
        self.rt_y=[]

        self.f_straight_x =[]
        self.f_straight_y=[]        

        self.theta_s= []

        self.f_right_x=[]
        self.f_right_y=[]
        
        self.theta_r=[]        

        self.f_left_x=[]
        self.f_left_y=[]
        
        self.theta_l=[]        

        self.radius = 0

        self.final_stop=0

        self.flag_r1 = 0 
        self.flag_r2 = 0 
        self.flag_l1 = 0 
        self.flag_l2 = 0 


        self.switch = 0
        self.car_yaw = 0       
        self.pos_x = 0
        self.f_tire_x=0
        self.f_tire_y=0
        self.pos_y = 0
        self.v_curr = 0
        self.counter = 0
        self.counter2 = 0

        self.aa_gya = 0
        self.gss_data=None

    def store_gss_data(self, data):
        
        self.aa_gya = 1
        self.gss_data=data
        # print("Hello in gss mode")

    def get_speed(self, data):
        '''
        Takes in speed from the GSS sensor.
        '''
        self.speed = data

    def get_carpose(self,data):

        self.carPose = data.pose.pose

    def catmull(self, x, y, num_points=1000):
   
        t = np.linspace(0, 1, len(x))
        spline_x = interp1d(t, x, kind='cubic')
        spline_y = interp1d(t, y, kind='cubic')
        t_smooth = np.linspace(0, 1, num_points)
        x_smooth = spline_x(t_smooth)
        y_smooth = spline_y(t_smooth)
        return x_smooth, y_smooth
    
    def vizualize_cones(self):
        '''
        '''
        
           
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
            
            marker.header.frame_id = 'base_footprint'
            self.conearray.markers.append(marker)
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

            
            marker.header.frame_id = 'base_footprint'
            self.conearray.markers.append(marker)
            self.id += 1

        for cone in self.big_orange_cones:
            marker = Marker()
            marker.ns = "track"
            marker.type = 2
            marker.action = 0
      
            # if color==0:
            color_cone_r = 1.0
            color_cone_g = 0.5
            color_cone_b = 0.0
            
            marker.id = self.id
            marker.color.r = color_cone_r
            marker.color.g = color_cone_g
            marker.color.b = color_cone_b
            marker.color.a = 0.5
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.4

            marker.pose.position.x = float(cone[0]) #x
            marker.pose.position.y = float(cone[1]) #y
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.header.frame_id = 'base_footprint'
            self.conearray.markers.append(marker)
            self.id += 1
            
        for cone in self.orange_cones:
            marker = Marker()
            marker.ns = "track"
            marker.type = 2
            marker.action = 0
      
            # if color==0:
            color_cone_r = 1.0
            color_cone_g = 0.5
            color_cone_b = 0.0
            
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

            marker.header.frame_id = 'base_footprint'
            self.conearray.markers.append(marker)
            self.id += 1

        for cone1 in self.conearray.markers:
            if(cone1.color.g == 0):
                for cone2 in self.conearray.markers:
                    if(cone2.color.g == 1 and (math.sqrt((cone1.pose.position.y - cone2.pose.position.y)**2 + (cone1.pose.position.x - cone2.pose.position.x)**2)< 3.8)):
                        # print('mid')
                        midx = (cone1.pose.position.x + cone2.pose.position.x)/2
                        midy = (cone1.pose.position.y + cone2.pose.position.y)/2
                        self.mid_x.append(midx)
                        self.mid_y.append(midy)
            elif(cone1.color.g == 1):
                continue
            else:
                for cone2 in self.conearray.markers:
                    if(cone2.color.g == 0.5 and cone1.pose.position.x == cone2.pose.position.x and abs(cone1.pose.position.y - cone2.pose.position.y) > 2.5 ):
                        midx = (cone1.pose.position.x + cone2.pose.position.x)/2
                        midy = (cone1.pose.position.y + cone2.pose.position.y)/2
                        if (cone1.scale.x == 0.4):
                            self.orange_mid_x.append(midx)
                            self.orange_mid_y.append(midy)
                        else:
                            self.stop_mid_x.append(midx)
                            self.stop_mid_y.append(midy)
                        self.mid_x.append(midx)
                        self.mid_y.append(midy)

        for i in range(len(self.mid_x)):
            if(self.mid_y[i] == 0):
                self.str_x.append(self.mid_x[i])
                self.str_y.append(self.mid_y[i])


            elif(self.mid_y[i]> 0.5):
                self.lt_x.append(self.mid_x[i])
                self.lt_y.append(self.mid_y[i])
            elif(self.mid_y[i]< 0):
                self.rt_x.append(self.mid_x[i])
                self.rt_y.append(self.mid_y[i])
        print(self.str_x)
        q=1
        r=0
        max_dist = 0
        for t in range(len(self.rt_x)):
            dist=math.sqrt(((self.rt_x[q]-self.rt_x[t])**2+(self.rt_y[q]-self.rt_y[t])**2))
            if (dist > max_dist):
                max_dist = dist
                r=t
        
        center_x = (self.rt_x[q]+self.rt_x[r])/2   #center_x is (14.41105224609375,-9.2763330078125) and radius is 9.112088146035356
        center_y = (self.rt_y[q]+self.rt_y[r])/2
        self.radius= max_dist/2
        
        circle_r_x=[]
        circle_r_y=[]
        for i in range(0,360,1):
            circle_r_x.append(center_x + self.radius*math.cos(np.deg2rad(i)))
            circle_r_y.append(center_y + self.radius*math.sin(np.deg2rad(i)))
        
                
        q1=1
        r1=0
        max_dist1 = 0
        for t in range(len(self.lt_x)):
            dist=math.sqrt(((self.lt_x[q1]-self.lt_x[t])**2+(self.lt_y[q1]-self.lt_y[t])**2))
            if (dist > max_dist1):
                max_dist1 = dist
                r1=t
        center_x_g = (self.lt_x[q1]+self.lt_x[r1])/2   #center_x is (14.41105224609375,+9.2763330078125) and radius is 9.112088146035356
        center_y_g = (self.lt_y[q1]+self.lt_y[r1])/2
        radius_g= max_dist1/2
        # print(center_x_g,center_y_g,radius_g)
    
        circle_l_x=[]
        circle_l_y=[]
        for i in range(0,360,1):
            circle_l_x.append(center_x_g + radius_g*math.cos(np.deg2rad(i)))
            circle_l_y.append(center_y_g + radius_g*math.sin(np.deg2rad(i)))
        


        self.f_straight_x,self.f_straight_y = self.catmull(self.str_x,self.str_y)
        self.f_left_x = circle_l_x
        self.f_left_y = circle_l_y
        self.f_right_x = circle_r_x
        self.f_right_y = circle_r_y
        
        def slope(x1, y1, x2, y2):
            m = (y2-y1)/(x2-x1)
            return np.arctan(m)

        for i in range(len(self.f_straight_y)-1):
            
            v=slope(self.f_straight_x[i], self.f_straight_y[i], self.f_straight_x[i+1], self.f_straight_y[i+1])
            self.theta_s.append(v)
        self.theta_s.append(slope(self.f_straight_x[-1],self.f_straight_y[-1],self.f_straight_x[0],self.f_straight_y[0]))

        for i in range(len(self.f_left_y)-1):
            
            v=slope(self.f_left_x[i], self.f_left_y[i], self.f_left_x[i+1], self.f_left_y[i+1])
            self.theta_l.append(v)
        self.theta_l.append(slope(self.f_left_x[-1], self.f_left_y[-1], self.f_left_x[0], self.f_left_y[0]))

        for i in range(len(self.f_right_y)-1):
            
            v=slope(self.f_right_x[i], self.f_right_y[i], self.f_right_x[i+1], self.f_right_y[i+1])
            self.theta_r.append(v)
        self.theta_r.append(slope(self.f_right_x[-1], self.f_right_y[-1], self.f_right_x[0], self.f_right_y[0]))
        
        for p in range (len(self.f_straight_y)):

            

            marker1 = Marker()
            marker1.header.frame_id = "map"
            marker1.header.stamp = rclpy.Time.now()

            marker1.ns = "basic_shapes"
            marker1.id =  self.id
            marker1.type = 0

            marker1.pose.position.x = self.f_straight_x[p]
            marker1.pose.position.y = self.f_straight_y[p]
            marker1.pose.position.z = 0
            marker1.pose.orientation.x = 0.0
            marker1.pose.orientation.y = 0.0
            marker1.pose.orientation.z = 0.0
            marker1.pose.orientation.w = 1.0
            
            marker1.scale.x = 0.1
            marker1.scale.y = 0.1
            marker1.scale.z = 0.1

            marker1.color.b = 1.0
            marker1.color.r = 1.0
            marker1.color.g = 0.5
            marker1.color.a = 1.0
            
            
            
            
            self.id=self.id+1

            self.conearray.markers.append(marker1)

        for p in range (len(self.f_right_y)):


            marker2 = Marker()
            marker2.header.frame_id = "map"
            marker2.header.stamp = rclpy.Time.now()

            marker2.ns = "basic_shapes"
            marker2.id =  self.id
            marker2.type = 0

            marker2.pose.position.x = self.f_right_x[p]
            marker2.pose.position.y = self.f_right_y[p]
            marker2.pose.position.z = 0
            marker2.pose.orientation.x = 0.0
            marker2.pose.orientation.y = 0.0
            marker2.pose.orientation.z = 0.0
            marker2.pose.orientation.w = 1.0
            
            marker2.scale.x = 0.1
            marker2.scale.y = 0.1
            marker2.scale.z = 0.1

            marker2.color.b = 0.0
            marker2.color.r = 1.0
            marker2.color.g = 0.0
            marker2.color.a = 1.0
            self.id=self.id+1
            self.conearray.markers.append(marker2)

        for p in range (len(self.f_left_y)):


            marker3 = Marker()
            marker3.header.frame_id = "map"
            marker3.header.stamp = rclpy.Time.now()

            marker3.ns = "basic_shapes"
            marker3.id =  self.id
            marker3.type = 0

            marker3.pose.position.x = self.f_left_x[p]
            marker3.pose.position.y = self.f_left_y[p]
            marker3.pose.position.z = 0
            marker3.pose.orientation.x = 0.0
            marker3.pose.orientation.y = 0.0
            marker3.pose.orientation.z = 0.0
            marker3.pose.orientation.w = 1.0
            
            marker3.scale.x = 0.1
            marker3.scale.y = 0.1
            marker3.scale.z = 0.1

            marker3.color.b = 0.0
            marker3.color.r = 0.0
            marker3.color.g = 1.0
            marker3.color.a = 1.0
            self.id=self.id+1
            self.conearray.markers.append(marker3)
            # print('Hello')
        for z in range (len(self.orange_mid_x)):
            marker4 = Marker()
            marker4.header.frame_id = "map"
            marker4.header.stamp = rclpy.Time.now()

            marker4.ns = "basic_shapes"
            marker4.id =  self.id
            marker4.type = 0

            marker4.pose.position.x = self.orange_mid_x[z]
            marker4.pose.position.y = self.orange_mid_y[z]
            marker4.pose.position.z = 0
            marker4.pose.orientation.x = 0.0
            marker4.pose.orientation.y = 0.0
            marker4.pose.orientation.z = 0.0
            marker4.pose.orientation.w = 1.0
            
            marker4.scale.x = 0.6
            marker4.scale.y = 0.6
            marker4.scale.z = 0.6

            marker4.color.b = 1.0
            marker4.color.r = 1.0
            marker4.color.g = 1.0
            marker4.color.a = 1.0
            self.id=self.id+1
            self.conearray.markers.append(marker4)

        
        marker4 = Marker()
        marker4.header.frame_id = "map"
        marker4.header.stamp = rclpy.Time.now()

        marker4.ns = "basic_shapes"
        marker4.id =  self.id
        marker4.type = 0

        marker4.pose.position.x = self.orange_mid_x[0]
        marker4.pose.position.y = (self.orange_mid_y[0]- 2*self.radius)
        marker4.pose.position.z = 0
        marker4.pose.orientation.x = 0.0
        marker4.pose.orientation.y = 0.0
        marker4.pose.orientation.z = 0.0
        marker4.pose.orientation.w = 1.0
        
        marker4.scale.x = 0.6
        marker4.scale.y = 0.6
        marker4.scale.z = 0.6

        marker4.color.b = 0.790
        marker4.color.r = 1.0
        marker4.color.g = 0.730
        marker4.color.a = 1.0
        self.id=self.id+1
        self.conearray.markers.append(marker4)

        marker4 = Marker()
        marker4.header.frame_id = "map"
        marker4.header.stamp = rclpy.Time.now()

        marker4.ns = "basic_shapes"
        marker4.id =  self.id
        marker4.type = 0

        marker4.pose.position.x = self.orange_mid_x[0]
        marker4.pose.position.y = (self.orange_mid_y[0] + 2*self.radius)
        marker4.pose.position.z = 0
        marker4.pose.orientation.x = 0.0
        marker4.pose.orientation.y = 0.0
        marker4.pose.orientation.z = 0.0
        marker4.pose.orientation.w = 1.0
        
        marker4.scale.x = 0.6
        marker4.scale.y = 0.6
        marker4.scale.z = 0.6

        marker4.color.b = 0.92
        marker4.color.r = 0.53
        marker4.color.g = 0.81
        marker4.color.a = 1.0
        self.id=self.id+1
        self.conearray.markers.append(marker4)

        marker4 = Marker()
        marker4.header.frame_id = "map"
        marker4.header.stamp = rclpy.Time.now()

        marker4.ns = "basic_shapes"
        marker4.id =  self.id
        marker4.type = 0

        marker4.pose.position.x = self.orange_mid_x[0]
        marker4.pose.position.y = (self.orange_mid_y[0]- 2*self.radius)
        marker4.pose.position.z = 0
        marker4.pose.orientation.x = 0.0
        marker4.pose.orientation.y = 0.0
        marker4.pose.orientation.z = 0.0
        marker4.pose.orientation.w = 1.0
        
        marker4.scale.x = 0.6
        marker4.scale.y = 0.6
        marker4.scale.z = 0.6

        marker4.color.b = 0.790
        marker4.color.r = 1.0
        marker4.color.g = 0.730
        marker4.color.a = 1.0
        self.id=self.id+1
        self.conearray.markers.append(marker4)

        marker4 = Marker()
        marker4.header.frame_id = "map"
        marker4.header.stamp = rclpy.Time.now()

        marker4.ns = "basic_shapes"
        marker4.id =  self.id
        marker4.type = 0

        marker4.pose.position.x = (self.orange_mid_x[0] - self.radius)
        marker4.pose.position.y = (self.orange_mid_y[0] + self.radius)
        marker4.pose.position.z = 0
        marker4.pose.orientation.x = 0.0
        marker4.pose.orientation.y = 0.0
        marker4.pose.orientation.z = 0.0
        marker4.pose.orientation.w = 1.0
        
        marker4.scale.x = 0.6
        marker4.scale.y = 0.6
        marker4.scale.z = 0.6

        marker4.color.b = 0.92
        marker4.color.r = 0.53
        marker4.color.g = 0.81
        marker4.color.a = 1.0
        self.id=self.id+1
        self.conearray.markers.append(marker4)

        marker4 = Marker()
        marker4.header.frame_id = "map"
        marker4.header.stamp = rclpy.Time.now()

        marker4.ns = "basic_shapes"
        marker4.id =  self.id
        marker4.type = 0

        marker4.pose.position.x = (self.orange_mid_x[0] - self.radius)
        marker4.pose.position.y = (self.orange_mid_y[0] - self.radius)
        marker4.pose.position.z = 0
        marker4.pose.orientation.x = 0.0
        marker4.pose.orientation.y = 0.0

        marker4.pose.orientation.z = 0.0
        marker4.pose.orientation.w = 1.0
        
        marker4.scale.x = 0.6
        marker4.scale.y = 0.6
        marker4.scale.z = 0.6

        marker4.color.b = 0.790
        marker4.color.r = 1.0
        marker4.color.g = 0.730
        marker4.color.a = 1.0
        self.id=self.id+1
        self.conearray.markers.append(marker4)

        rate = rcply.Rate(10)
        print('Lessgo')
        # print(markerArrayMsg)
        cones_viz =  rclpy.Publisher('/track_visualization', MarkerArray, queue_size=10)

        while True:
            cones_viz.publish(self.conearray)
            rate.sleep()

    def send_waypoints(self):
        '''
        Description in short pls.
        '''
        # if self.track_available == False:
        #     self.t_start = time.time()
        #     return None
        
            


        # if time.time() < self.t_start + self.t_runtime:
        #     # carControlsmsg = ControlCommand()

        #     pos_x = self.carPose.position.x
        #     pos_y = self.carPose.position.y

        #     q = self.carPose.orientation
        #     v_curr=np.sqrt(self.speed.twist.twist.linear.x**2 + self.speed.twist.twist.linear.y**2)
        #     car_yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
            
        #     # v_curr = np.sqrt(self.gss_data.twist.twist.linear.x**2+self.gss_data.twist.twist.linear.y**2)
        #     # car_yaw = self.carPose.orientation.x
        #     print(f"carpose_x: {pos_x}, carpose_y: {pos_y}, car_yaw : {car_yaw}")

        if(self.switch == 0):
            self.final_x = self.f_straight_x
            self.final_y = self.f_straight_y
            self.theta_final = self.theta_s
        elif(self.switch == 1):
            self.final_x = self.f_right_x
            self.final_y = self.f_right_y
            self.theta_final = self.theta_r
        else:
            self.final_x = self.f_left_x
            self.final_y = self.f_left_y
            self.theta_final = self.theta_l
        
        # print(self.final_x, self.final_y)

        # dt_vel = time.time() - self.t1   #Used to obtain the time difference for PID control.
        # self.t1 = time.time()

        # self.f_tire_x= self.pos_x + 1.7*cos(self.car_yaw)
        # self.f_tire_y= self.pos_y + 1.7*sin(self.car_yaw)
        # min_cte=10000
        # for i in range(len(self.final_x)):
        #     # dist = np.hypot((final_x[i] - f_tire_x),(final_y[i] - f_tire_y))
        #     dist = np.sqrt((self.final_x[i] - self.f_tire_x)**2 + (self.final_y[i] - self.f_tire_y)**2)
            
        #     if dist < min_cte:
        #         min_cte = dist
        #         closest_waypoint_index=i


        # [steer_pp, x_p, y_p] = pure_pursuit(x=self.final_x, y=self.final_y, vf=v_curr, pos_x=pos_x, pos_y=pos_y, veh_head=car_yaw,pos=closest_waypoint_index)

        # [steer_pp, x_p, y_p,closest_waypoint_index] = stanley_steering(final_x=self.final_x,final_y=self.final_y,v_curr=v_curr,pos_x=pos_x,pos_y=pos_y,car_yaw=car_yaw)
        min_cte = 10000
        i=1
        for i in range(len(self.final_x)):
            # dist = np.hypot((final_x[i] - f_tire_x),(final_y[i] - f_tire_y))
            dist = np.sqrt((self.final_x[i] - self.f_tire_x)**2 + (self.final_y[i] - self.f_tire_y)**2)
            
            if dist < min_cte:
                closest_waypoint_index = i
        
        msg = Float32MultiArray()
        # Define the layout of the multiarray
        msg.layout.dim = [MultiArrayDimension(label='rows', size=2, stride=len(self.final_x)),
                        MultiArrayDimension(label='cols', size=len(self.final_x), stride=1)]
        msg.layout.data_offset = 0
        msg.data = self.final_x + self.final_y  # Flatten the 2D array into 1D

        self.waypoint_path.publish(msg)


        self.trigger_x = self.orange_mid_x[0]
        self.trigger_y = self.orange_mid_y[0]
        
        self.trigger_red_x  = self.trigger_x 
        self.trigger_red_y = self.trigger_y - 2*self.radius
        self.trigger_red_x_2  = self.trigger_x - self.radius 
        self.trigger_red_y_2 = self.trigger_y - self.radius

        self.trigger_green_x  = self.trigger_x 
        self.trigger_green_y = self.trigger_y + 2*self.radius
        self.trigger_green_x_2  = self.trigger_x - self.radius 
        self.trigger_green_y_2 = self.trigger_y + self.radius
        # red_trigger_x = trigger_x + 2*radius
        # red_circ = 2*3.14*radius
        if(self.switch == 0 and (abs(self.final_y[closest_waypoint_index] - self.trigger_y) < 0.1) and (abs(self.final_x[closest_waypoint_index] - self.trigger_x) < 0.1)):
            self.switch = 1
            print("switched")
        if((abs(self.final_y[closest_waypoint_index] - self.trigger_red_y) < 1) and (abs(self.final_x[closest_waypoint_index] - self.trigger_red_x) < 1)):
            self.flag_r1=1
        if((abs(self.final_y[closest_waypoint_index] - self.trigger_red_y_2) < 1) and (abs(self.final_x[closest_waypoint_index] - self.trigger_red_x_2) < 1)):
            self.flag_r2=1

        if(self.flag_r1 == 1 and self.flag_r2 == 1 and self.counter <=1):
            self.counter=self.counter+1
            self.flag_r1 = 0
            self.flag_r2 = 0

        if(self.counter == 2 and (abs(self.final_y[closest_waypoint_index] - self.trigger_y) < 3) and (abs(self.final_x[closest_waypoint_index] - self.trigger_x) < 0.1)):
            self.switch = 2
            print("switched Again")
        
        if((abs(self.final_y[closest_waypoint_index] - self.trigger_green_y) < 1) and (abs(self.final_x[closest_waypoint_index] - self.trigger_green_x) < 1)):
            self.flag_l1 = 1
        if((abs(self.final_y[closest_waypoint_index] - self.trigger_green_y_2) < 1) and (abs(self.final_x[closest_waypoint_index] - self.trigger_green_x_2) < 1)):
            self.flag_l2 = 1
        if(self.flag_l1 == 1 and self.flag_l2 == 1 and self.counter2<=1):
            self.counter2=self.counter2+1
            self.flag_l1 = 0
            self.flag_l2 = 0    
        
        if(self.counter2 == 2 and (abs(self.final_y[closest_waypoint_index] - self.trigger_y) < 3) and (abs(self.final_x[closest_waypoint_index] - self.trigger_x) < 0.1)):
            self.switch = 0
            print("Switched Again")
                
        for s in range(len(self.stop_mid_x)):
            if((self.stop_mid_x[-3] <= self.stop_mid_x[s])and (abs(self.final_y[closest_waypoint_index] - self.stop_mid_y[s]) < 0.1) and (abs(self.final_x[closest_waypoint_index] - self.stop_mid_x[s]) < 0.1)):
                self.final_stop = 1
        # if(counter == 1 and   )
        print("Switch is ", self.switch)
        print("Counter is ", self.counter)
        print("Counter2 is ", self.counter2)

            

            # vel_err = v_curr - self.const_velocity
            # if vel_err<0:
            #     throttle=0.2
            #     brake=0
                
            # else:
            #     throttle=0
            #     brake=0.2
            
            # carControlsmsg.throttle = throttle
            # carControlsmsg.brake = brake
            # carControlsmsg.steering = steer_pp

            # if(self.final_stop == 1):
            #     carControlsmsg.throttle = 0
            #     carControlsmsg.brake = 0.6
            # carControls.publish(carControlsmsg)
        
        # else:
        #     '''
        #     do something. call reset service maybe?
        #     '''
        #     rclpy.wait_for_service('/fsds/reset')
        #     reset = rclpy.ServiceProxy('/fsds/reset', Reset)

        #     try:
        #         reset(True)
        #         print('Reset done...')
        #     except rclpy.ServiceException as exc:
        #         print("Service did not process request: " + self.str(exc))
        #     rclpy.signal_shutdown("Time khatam")
        #     pass
            
    def get_map(self, data):
        '''
        Store the map of all the cones.
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
        self.get_logger().info(f'Got data from simulated perception. Blue:{len(data.blue_cones)} Yellow:{len(data.yellow_cones)}')
        # self.get_logger().info(f"Got data from perception. No of cones - {len(data.track)}")

        blue_cones = []
        yellow_cones = []
        big_orange_cones = []
        orange_cones=[]
        # for /perception/cones
        # for cone in data.track:
        #     if cone.color == 0:
        #         blue_cones.append([(cone.location.x)*cos(cone.location.y), (cone.location.x)*sin(cone.location.y)])
        #     elif cone.color == 1:
        #         yellow_cones.append([(cone.location.x)*cos(cone.location.y), (cone.location.x)*sin(cone.location.y)])

        # for /ground_truth/cones       
        for cone in data.blue_cones:
            blue_cones.append([cone.point.x, cone.point.y])
        for cone in data.yellow_cones:
            yellow_cones.append([cone.point.x, cone.point.y])
        for cone in data.big_orange_cones:
            big_orange_cones.append([cone.point.x, cone.point.y])
        for cone in data.orange_cones:
            orange_cones.append([cone.point.x, cone.point.y])
        self.blue_cones = np.array(blue_cones)
        self.yellow_cones = np.array(yellow_cones)
        self.big_orange_cones = np.array(big_orange_cones)
        self.orange_cones = np.array(orange_cones)

        self.vizualize_cones()
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
    ads_dv = path_planner()
    rclpy.spin(ads_dv)

    ads_dv.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()