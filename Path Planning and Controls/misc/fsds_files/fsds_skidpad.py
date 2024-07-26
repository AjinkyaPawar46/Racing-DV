#!/usr/bin/env python3
import rospy,math
from dv_msgs.msg import Track
from dv_msgs.msg import ControlCommand

from dv_msgs.srv import Reset
from scipy.interpolate import interp1d
import numpy as np
import sympy as sp
from geometry_msgs.msg import TwistWithCovarianceStamped,Pose,TransformStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import time

from submodules_ppc.trajectory_packages import *



class dv_car():
    '''
    Description in short pls.
    '''
    
    
    dT=0.05
    update_carPose_in=0.01
    cubeScale=0.1
    # Decides how long the simulation runs (Units = seconds)
    t_runtime = 150
    
    def __init__(self) -> None:
        
        self.t1=time.time()-1  #variable to be used to calculate dt loop. This is the actual dt after inducing 
        self.t_start=time.time()
        self.integral = 0
        self.blue_cone_IDs=[]
        self.yellow_cone_IDs=[]
        self.carPose = Pose()
        self.error=0
        self.prev_error = 0
        self.vel_error = 0
        self.conearray = MarkerArray()

        self.speed = TwistWithCovarianceStamped()
        self.data_from_mrptstate=MarkerArray()
        self.paired_indexes=np.array([[0,0]])
        self.const_velocity= 2
        self.I = 0
        self.map_to_FSCar = TransformStamped()
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
    
    def vizualize_cones(self, data):
        '''
        '''
        self.track = data.track
        
           
        for cone in data.track:
            # print("cone")
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()

            marker.ns = "basic_shapes"
            marker.id =  self.id+1
            marker.type = 0
            

            marker.pose.position.x = cone.location.x
            marker.pose.position.y = cone.location.y
            marker.pose.position.z = cone.location.z

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 1.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            
            if cone.color == 0:  #blue
                marker.color.b = 1.0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.a = 1.0
            elif cone.color == 1: #yelow
                marker.color.b = 0.0
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.a = 1.0
            elif cone.color == 2:   # big orange
                marker.color.b = 0.0
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.a = 1.0

                marker.scale.x = 0.4
                marker.scale.y = 0.4
                marker.scale.z = 0.4
            elif cone.color == 3:  #small orange
                marker.color.b = 0.0
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.a = 1.0


            self.conearray.markers.append(marker)
            
            self.id = self.id + 1
        
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
            marker1.header.stamp = rospy.Time.now()

            marker1.ns = "basic_shapes"
            marker1.id =  self.id
            marker1.type = Marker.SPHERE

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
            marker2.header.stamp = rospy.Time.now()

            marker2.ns = "basic_shapes"
            marker2.id =  self.id
            marker2.type = Marker.SPHERE

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
            marker3.header.stamp = rospy.Time.now()

            marker3.ns = "basic_shapes"
            marker3.id =  self.id
            marker3.type = Marker.SPHERE

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
            marker4.header.stamp = rospy.Time.now()

            marker4.ns = "basic_shapes"
            marker4.id =  self.id
            marker4.type = Marker.CUBE

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
        marker4.header.stamp = rospy.Time.now()

        marker4.ns = "basic_shapes"
        marker4.id =  self.id
        marker4.type = Marker.CUBE

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

        marker4.color.b = 0.502
        marker4.color.r = 0
        marker4.color.g = 0.502
        marker4.color.a = 1.0
        self.id=self.id+1
        self.conearray.markers.append(marker4)

        marker4 = Marker()
        marker4.header.frame_id = "map"
        marker4.header.stamp = rospy.Time.now()

        marker4.ns = "basic_shapes"
        marker4.id =  self.id
        marker4.type = Marker.CUBE

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

        marker4.color.b = 0.502
        marker4.color.r = 0
        marker4.color.g = 0.502
        marker4.color.a = 1.0
        self.id=self.id+1
        self.conearray.markers.append(marker4)

        marker4 = Marker()
        marker4.header.frame_id = "map"
        marker4.header.stamp = rospy.Time.now()

        marker4.ns = "basic_shapes"
        marker4.id =  self.id
        marker4.type = Marker.CUBE

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

        marker4.color.b = 0.502
        marker4.color.r = 0
        marker4.color.g = 0.502
        marker4.color.a = 1.0
        self.id=self.id+1
        self.conearray.markers.append(marker4)

        marker4 = Marker()
        marker4.header.frame_id = "map"
        marker4.header.stamp = rospy.Time.now()

        marker4.ns = "basic_shapes"
        marker4.id =  self.id
        marker4.type = Marker.CUBE

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

        marker4.color.b = 0.502
        marker4.color.r = 0
        marker4.color.g = 0.502
        marker4.color.a = 1.0
        self.id=self.id+1
        self.conearray.markers.append(marker4)

        marker4 = Marker()
        marker4.header.frame_id = "map"
        marker4.header.stamp = rospy.Time.now()

        marker4.ns = "basic_shapes"
        marker4.id =  self.id
        marker4.type = Marker.CUBE

        marker4.pose.position.x = (self.orange_mid_x[0] - self.radius)
        marker4.pose.position.y = (self.orange_mid_y[0]- self.radius)
        marker4.pose.position.z = 0
        marker4.pose.orientation.x = 0.0
        marker4.pose.orientation.y = 0.0

        marker4.pose.orientation.z = 0.0
        marker4.pose.orientation.w = 1.0
        
        marker4.scale.x = 0.6
        marker4.scale.y = 0.6
        marker4.scale.z = 0.6

        marker4.color.b = 0.502
        marker4.color.r = 0
        marker4.color.g = 0.502
        marker4.color.a = 1.0
        self.id=self.id+1
        self.conearray.markers.append(marker4)

        rate = rospy.Rate(10)
        print('Lessgo')
        # print(markerArrayMsg)
        cones_viz =  rospy.Publisher('/track_visualization', MarkerArray, queue_size=10)

        while True:
            cones_viz.publish(self.conearray)
            rate.sleep()

    def get_fullmap(self, data):
        '''
        Takes in and stores the Track information provided by fsds through a latched (only publishes once) ros topic. 
        '''
        track = data.track
        blue_cones = []
        yellow_cones = []

        for cone in track:
            if cone.color == 0:
                blue_cones.append([cone.location.x, cone.location.y,cone.color])
            elif cone.color == 1:
                yellow_cones.append([cone.location.x, cone.location.y,cone.color])
    
        self.blue_cones = np.array(blue_cones)
        self.yellow_cones = np.array(yellow_cones)
        self.track_available = True

        
        
        
        
        return None
    

    def control(self, event):
            '''
            Description in short pls.
            '''
            if self.track_available == False:
                self.t_start = time.time()
                return None
           
                


            if time.time() < self.t_start + self.t_runtime:
                carControlsmsg = ControlCommand()

                pos_x = self.carPose.position.x
                pos_y = self.carPose.position.y

                q = self.carPose.orientation
                v_curr=np.sqrt(self.speed.twist.twist.linear.x**2 + self.speed.twist.twist.linear.y**2)
                car_yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
                
                # v_curr = np.sqrt(self.gss_data.twist.twist.linear.x**2+self.gss_data.twist.twist.linear.y**2)
                # car_yaw = self.carPose.orientation.x
                print(f"carpose_x: {pos_x}, carpose_y: {pos_y}, car_yaw : {car_yaw}")

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

                dt_vel = time.time() - self.t1   #Used to obtain the time difference for PID control.
                self.t1 = time.time()

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

                [steer_pp, x_p, y_p,closest_waypoint_index] = stanley_steering(final_x=self.final_x,final_y=self.final_y,v_curr=v_curr,pos_x=pos_x,pos_y=pos_y,car_yaw=car_yaw)
                
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

                

                vel_err = v_curr - self.const_velocity
                if vel_err<0:
                    throttle=0.2
                    brake=0
                    
                else:
                    throttle=0
                    brake=0.2
                
                carControlsmsg.throttle = throttle
                carControlsmsg.brake = brake
                carControlsmsg.steering = steer_pp

                if(self.final_stop == 1):
                    carControlsmsg.throttle = 0
                    carControlsmsg.brake = 0.6
                carControls.publish(carControlsmsg)
            
            else:
                '''
                do something. call reset service maybe?
                '''
                rospy.wait_for_service('/fsds/reset')
                reset = rospy.ServiceProxy('/fsds/reset', Reset)

                try:
                    reset(True)
                    print('Reset done...')
                except rospy.ServiceException as exc:
                    print("Service did not process request: " + self.str(exc))
                rospy.signal_shutdown("Time khatam")
                pass
            


if __name__ == '__main__':
    Obj = dv_car()

    rospy.init_node('fsds_ros')

    rospy.Subscriber('/fsds/testing_only/track', Track, Obj.get_fullmap)

    rospy.Subscriber('/fsds/testing_only/track', Track, Obj.vizualize_cones)

    rospy.Subscriber('/fsds/testing_only/odom', Odometry, Obj.get_carpose)
    # rospy.Subscriber('/corrected_pose_ppc', Odometry, Obj.get_carpose)
    # rospy.Subscriber('/fsds/gss', TwistWithCovarianceStamped, Obj.store_gss_data)

    rospy.Subscriber('/fsds/gss', TwistWithCovarianceStamped, Obj.get_speed)

    following_closest_waypoint_index = rospy.Publisher('closest_waypoint_index', Marker, queue_size=10) 

    carControls = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=10)

    # cones_viz =  rospy.Publisher('/track_visualization', MarkerArray, queue_size=10)

    rospy.Timer(rospy.Duration(Obj.dT), Obj.control)
    
    rospy.spin()