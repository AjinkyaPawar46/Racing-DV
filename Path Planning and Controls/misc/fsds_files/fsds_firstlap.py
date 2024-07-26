#/usr/bin/env python3

import numpy as np
import time
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros
import scipy.optimize
import matplotlib.pyplot as plt

from .submodules_ppc.trajectory_packages import *
from dv_msgs.msg import Track, ControlCommand
from dv_msgs.srv import Reset
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped, Pose,TransformStamped
from visualization_msgs.msg import MarkerArray,Marker

class dv_car(Node):
    '''
    Description in short pls.
    '''
    
    
    dT=0.05
    update_carPose_in=0.01
    cubeScale=0.1
    # Decides how long the simulation runs (Units = seconds)
    t_runtime = 60


    def __init__(self) -> None:
        super().__init__('firstlap')
        self.create_subscription(Track, '/fsds/testing_only/track', self.get_fullmap, 10)
        self.create_subscription(Track, '/fsds/testing_only/track', self.vizualize_cones, 10)
        self.create_subscription(Odometry, '/fsds/testing_only/odom', self.get_carpose, 10)
        self.create_subscription(TwistWithCovarianceStamped, '/fsds/gss', self.get_speed, 10)

        self.following_waypoint = self.create_publisher(Marker, 'waypoint', 10) 
        self.carControls = self.create_publisher(ControlCommand, '/fsds/control_command', 10)

        self.t1=time.time()-1  #variable to be used to calculate dt loop. This is the actual dt after inducing 
        self.t_start=time.time()
        self.integral = 0
        self.blue_cone_IDs=[]
        self.yellow_cone_IDs=[]
        self.carPose = Pose()
        self.error=0
        self.prev_error = 0
        self.vel_error = 0

        self.speed = TwistWithCovarianceStamped()
        self.data_from_mrptstate=MarkerArray()
        self.midpoints=np.array([[0,0]])
        self.paired_indexes=np.array([[0,0]])
        self.const_velocity=3
        self.I = 0
        self.map_to_FSCar = TransformStamped()
        self.pp_id=0
        self.id = 0
        self.track_available = False
        self.const_velocity = 8
        self.all_detected_blue_cones=np.array([[0,0,0]])
        self.all_detected_yellow_cones=np.array([[0,0,0]])
        self.paired_indexes=np.array([[0,0]])
        self.midpoints=np.array([[0,0]])
        self.create_timer(self.dT, self.control)

        


    def get_speed(self, data):
        '''
        Takes in speed from the GSS sensor.
        '''
        self.speed = data

    def get_carpose(self,data):

        self.carPose = data.pose.pose

    def vizualize_cones(self, data):
        '''
        '''
        self.track = data.track
        markerArrayMsg = MarkerArray()
        
        for cone in self.track:
            marker = Marker()
            marker.ns = "track"
            marker.type = 2
            marker.action = 0
            
            color = cone.color
            if color==0:
                    color_cone_r = 0
                    color_cone_g = 0
                    color_cone_b = 1
            if color==1:
                    color_cone_r = 1
                    color_cone_g = 1
                    color_cone_b = 0
            if color==2:
                    color_cone_r = 1
                    color_cone_g = 0.27
                    color_cone_b = 0 
            if color==3:
                    color_cone_r = 1
                    color_cone_g = 0.647
                    color_cone_b = 0.5
            if color==4:
                    color_cone_r = 1
                    color_cone_g = 1 
                    color_cone_b = 1
            
            marker.id = self.id
            marker.color.r = color_cone_r
            marker.color.g = color_cone_g
            marker.color.b = color_cone_b
            marker.color.a = 0.5
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.pose.position = cone.location
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.lifetime = Duration(0)
            marker.header.frame_id = 'map'
            markerArrayMsg.markers.append(marker)
            self.id += 1

        def timer_callback(self):
            self.cones_viz.publish(markerArrayMsg)

        print('hi')
        # print(markerArrayMsg)
        self.cones_viz = self.create_publisher(MarkerArray, '/track_visualization', 10)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, timer_callback)
        


    def vizulaize_pp_waypoint(self,x_pp,y_pp):
        '''
        Publishing parallel car pose BLUE-cubes
        '''
        
        data=Pose()
        data.position.x=x_pp
        data.position.y=y_pp
        Pure_pursuit_waypoint_msg = Marker()
        Pure_pursuit_waypoint_msg.header.frame_id = 'map'
        Pure_pursuit_waypoint_msg.ns = "Way_ppoint"
        Pure_pursuit_waypoint_msg.id = self.pp_id
        self.pp_id+=1
        Pure_pursuit_waypoint_msg.type = 1
        Pure_pursuit_waypoint_msg.action = 0
        Pure_pursuit_waypoint_msg.pose = data
        Pure_pursuit_waypoint_msg.scale.x = self.cubeScale
        Pure_pursuit_waypoint_msg.scale.y = self.cubeScale
        Pure_pursuit_waypoint_msg.scale.z = self.cubeScale
        Pure_pursuit_waypoint_msg.color.r = 0
        Pure_pursuit_waypoint_msg.color.g = 256
        Pure_pursuit_waypoint_msg.color.b = 256
        Pure_pursuit_waypoint_msg.color.a = 1
        Pure_pursuit_waypoint_msg.lifetime = Duration(1)        
        self.following_waypoint.publish(Pure_pursuit_waypoint_msg)
        #print('shreyashsax')

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
        # bounds_left=evaluate_bezier(blue_cones,3)
        # bounds_right=evaluate_bezier(yellow_cones,3)
        # bounds_right=bounds_right[:,0:2]
        # bounds_left=bounds_left[:,0:2]

        # def totalcurvature(params,bounds_left=bounds_left,bounds_right=bounds_right):
        #     [x,y]=genpath(bounds_left=bounds_left,bounds_right=bounds_right,params=params)
        #     # plt.plot(x,y)
        #     # plt.pause(0.05)
        #     dx_dt = np.gradient(x)
        #     dy_dt = np.gradient(y)
            
        #     d2x_dt2 = np.gradient(dx_dt)
        #     d2y_dt2 = np.gradient(dy_dt)

        #     curvature = ((d2x_dt2 * dy_dt - dx_dt * d2y_dt2) /(dx_dt * dx_dt + dy_dt * dy_dt)**1.5)
        #     objective=0
        #     for c in curvature:
        #         objective=objective+c**2
        #     return objective



        # params=np.zeros(len(bounds_left)-1)
        # for i in range(len(params)):
        #     params[i]=0.5
        # result=scipy.optimize.minimize(fun=totalcurvature,x0=params,bounds=scipy.optimize.Bounds(lb=0.25, ub=0.75, keep_feasible=False))
        # params=result.x
        # print(result)

        # # print(params)
        # [x,y]=genpath(bounds_left=bounds_left,bounds_right=bounds_right,params=params)



        # self.x=np.array(x)
        # self.y=np.array(y)
        
        # print(self.x,self.y)

        # [self.k, self.R] = curvature(x, y)
        # # plt.plot(bounds_left[:,0],bounds_left[:,1], c = 'blue')
        # # plt.plot(bounds_right[:,0],bounds_right[:,1], c = 'yellow')
        # # plt.plot(self.x, self.y, c = 'green')
        # # plt.show()

        # v2, expected_lap_time,v1,r,self.v = vel_find3(self.x, self.y, mu = 0.6, m = 230, g = 9.8)
        # self.track_available = True
        # # plt.plot(v2, c = 'green' )
        # # plt.plot(v1, c = 'black' )
        # # plt.plot(self.v, c = 'blue' )
        # # plt.show()

        return None

    def control(self):
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
                new_current_detected_cones,self.blue_cones,self.yellow_cones=sense_cones2(pos_x=pos_x,pos_y=pos_y,blue_cones=self.blue_cones,
                                                                                yellow_cones=self.yellow_cones,range=15,yaw=car_yaw,fov=100)
                #print(all_detected_blue_cones,all_detected_yellow_cones)
                for cone in new_current_detected_cones:

                    cone=np.array(cone)
                    
                    if cone[2]==0:            #blue=0(left),yellow=1(right)
                        self.all_detected_blue_cones=np.append(self.all_detected_blue_cones,[cone],axis=0)
                    else:
                        self.all_detected_yellow_cones=np.append(self.all_detected_yellow_cones,[cone],axis=0)
    
                new_pairs=pair_(detected_blue_cones=self.all_detected_blue_cones,detected_yellow_cones=self.all_detected_yellow_cones,paired=self.paired_indexes)
                self.paired_indexes=np.append(self.paired_indexes,new_pairs,axis=0)
                for pair in new_pairs:
                    blue_ofpair=self.all_detected_blue_cones[pair[0]]
                    yellow_ofpair=self.all_detected_yellow_cones[pair[1]]
                    x1=blue_ofpair[0]
                    x2=yellow_ofpair[0]
                    y1=blue_ofpair[1]
                    y2=yellow_ofpair[1]
                    self.midpoints=np.append(self.midpoints,[[(x1+x2)/2,(y1+y2)/2]],axis=0)
                # kp =  0.4
                # ki =  0.00
                # kd =  0.02
                dt_vel = time.time() - self.t1   #Used to obtain the time difference for PID control.
                self.t1 = time.time()
                closest_waypoint_index=np.argmin((pos_x-self.midpoints[:,0])**2+(pos_y-self.midpoints[:,1])**2)
                [steer_pp, x_p, y_p] = pure_pursuit(x=self.midpoints[:,0], y=self.midpoints[:,1], vf=v_curr, pos_x=pos_x, pos_y=pos_y, veh_head=car_yaw,pos=closest_waypoint_index)

                # closest_waypoint_index=np.argmin((pos_x-self.x)**2+(pos_y-self.y)**2)
                # [throttle,brake,self.integral,self.vel_error,diffn ] = vel_controller2(kp=kp, ki=ki, kd=kd,
                #                                             v_curr=v_curr, v_ref=self.v[closest_waypoint_index],
                #                                             dt=dt_vel, prev_integral=self.integral, prev_vel_error=self.vel_error)
                # print('close_index',closest_waypoint_index)
                # print('no. of midpoints',self.midpoints.shape)
                # print('paired',self.paired_indexes)

                # [steer_pp, x_p, y_p] = pure_pursuit(x=self.x, y=self.y, vf=v_curr, pos_x=pos_x, pos_y=pos_y, veh_head=car_yaw,pos=closest_waypoint_index)
                vel_err = v_curr - self.const_velocity
                if vel_err<0:
                    throttle=0.2
                    brake=0
                    
                else:
                    throttle=0
                    brake=0.2

                self.vizulaize_pp_waypoint(x_pp=x_p,y_pp=y_p)
                #print('following',x_p,y_p)
                #print('position',pos_x,pos_y)
                #print('steer',steer_pp,'yaw',car_yaw)

                carControlsmsg.throttle = throttle
                carControlsmsg.brake = brake
                carControlsmsg.steering = steer_pp

                self.carControls.publish(carControlsmsg)


            #     if len(self.x) > 0:
            #         pos = np.argmin((pos_x - np.array(self.x))**2 + (pos_y - np.array(self.y))**2)
            #         pos=pos+1

            #         self.x = shift(pos, self.x)
            #         self.y = shift(pos, self.y)
            #         self.v = shift(pos, self.v)

            #         v_curr=np.sqrt(self.speed.twist.twist.linear.x**2 + self.speed.twist.twist.linear.y**2)
                
                
            #         [throttle,brake ] = vel_controller(v_prev=v_prev, v_curr=v_curr, v_ref=self.v[0], dt_vel=self.dT)
            #         carControlsmsg.throttle = throttle
            #         carControlsmsg.brake = brake
                    
                    
            #         #velocity after giving accn and brake 
                    
            #         [steer, xc, yc] = pure_pursuit(np.array(self.x), np.array(self.y), v_curr, pos_x, pos_y, car_yaw, self.ROC)
            #         carControlsmsg.steering = steer
                    
            #         carControls.publish(carControlsmsg)
                    
            #         v_prev = np.sqrt(self.speed.twist.twist.linear.x**2 + self.speed.twist.twist.linear.y**2)
            else:
                '''
                do something. call reset service maybe?
                '''
                cli = self.create_client(Reset, '/fsds/reset')
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            req = Reset.Request()
            req.wait_on_last_task = True
            future = cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            print('Reset done...')
            rclpy.shutdown("Time khatam")
            
        

def main(args=None):
    rclpy.init(args=args)
    batgrip = dv_car()
    
    rclpy.spin(batgrip)
    batgrip.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
     main()

