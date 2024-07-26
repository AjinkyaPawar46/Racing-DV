#!/usr/bin/env python

import numpy as np
import time
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

# Standard Messages
from geometry_msgs.msg import Pose, TwistWithCovarianceStamped
from visualization_msgs.msg import Marker

# Custom messages/Actions
from dv_msgs.msg import Track, ControlCommand
from dv_msgs.srv import Reset
from dv_msgs.msg import Odometry
from dv_msgs.msg import TwistWithCovarianceStamped, Pose,TransformStamped
from dv_msgs.msg import MarkerArray

# Function definitions
from .submodules_ppc.trajectory_packages import *

class dv_car(Node):
    dT=0.05
    update_carPose_in=0.01
    cubeScale=0.1
    # Decides how long the simulation runs (Units = seconds)
    t_runtime = 15
    def __init__(self):
        super().__init__('fsds_raceline')
        self.create_subscription(Track, '/fsds/testing_only/track', self.get_map, 10) # queue size is 10
        self.create_subscription(Odometry, '/fsds/testing_only/odom', self.get_car_pose, 10)
        self.create_subscription(TwistWithCovarianceStamped, '/fsds/gss', self.get_speed, 10)
        self.following_waypoint = self.create_publisher(Marker, 'waypoint', 10)
        self.car_controls = self.create_publisher(ControlCommand, '/fsds/control_command', 10)
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
        self.track_available = False

    def get_speed(self, data):
        self.speed = data

    def get_car_pose(self, data):
        self.car_pose = data.pose.pose

    def visualize_pp_waypoint(self, x_pp, y_pp):
        data = Pose()
        data.position.x = x_pp
        data.position.y = y_pp
        pure_pursuit_waypoint_msg = Marker()
        pure_pursuit_waypoint_msg.header.frame_id = 'map'
        pure_pursuit_waypoint_msg.ns = "Way_ppoint"
        pure_pursuit_waypoint_msg.id = self.pp_id
        self.pp_id += 1
        pure_pursuit_waypoint_msg.type = 1
        pure_pursuit_waypoint_msg.action = 0
        pure_pursuit_waypoint_msg.pose = data
        pure_pursuit_waypoint_msg.scale.x = self.cube_scale
        pure_pursuit_waypoint_msg.scale.y = self.cube_scale
        pure_pursuit_waypoint_msg.scale.z = self.cube_scale
        pure_pursuit_waypoint_msg.color.r = 0
        pure_pursuit_waypoint_msg.color.g = 256
        pure_pursuit_waypoint_msg.color.b = 256
        pure_pursuit_waypoint_msg.color.a = 1
        pure_pursuit_waypoint_msg.lifetime = Duration(seconds = 1)
        self.following_waypoint.publish(pure_pursuit_waypoint_msg)

    def get_map(self, data):
        '''
        Takes in and stores the Track information provided by fsds through a latched (only publishes once) ros topic. 
        '''
        track = data.track
        blue_cones = []
        yellow_cones = []

        for cone in track:
            if cone.color == 0:
                blue_cones.append([cone.location.x, cone.location.y])
            elif cone.color == 1:
                yellow_cones.append([cone.location.x, cone.location.y])

        blue_cones = np.array(blue_cones)
        yellow_cones = np.array(yellow_cones)

        bounds_left = evaluate_bezier(blue_cones, 3)
        bounds_right = evaluate_bezier(yellow_cones, 3)
        bounds_right = bounds_right[:, 0:2]
        bounds_left = bounds_left[:, 0:2]

        def total_curvature(params, bounds_left=bounds_left, bounds_right=bounds_right):
            [x, y] = genpath(bounds_left=bounds_left, bounds_right=bounds_right, params=params)
            dx_dt = np.gradient(x)
            dy_dt = np.gradient(y)
            d2x_dt2 = np.gradient(dx_dt)
            d2y_dt2 = np.gradient(dy_dt)
            curvature = ((d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5)
            objective = 0
            for c in curvature:
                objective = objective + c ** 2
            return objective

        params = np.zeros(len(bounds_left) - 1)
        for i in range(len(params)):
            params[i] = 0.5
        result = scipy.optimize.minimize(fun=total_curvature, x0=params,
                                          bounds=scipy.optimize.Bounds(lb=0.25, ub=0.75, keep_feasible=False))
        params = result.x
        print(result)

        [x, y] = genpath(bounds_left=bounds_left, bounds_right=bounds_right, params=params)

        self.x = np.array(x)
        self.y = np.array(y)
        self.track_available = True
        print(self.x, self.y)

        [self.k, self.R] = curvature(x, y)
        v2, expected_lap_time,v1,r,self.v = vel_find3(self.x, self.y, mu = 0.6, m = 230, g = 9.8)
        return None

    def control(self, event):
        '''
        Description in short pls.
        '''
        if self.track_available == False:
            return

        if time.time() < self.t_start + self.t_runtime:
            car_controls_msg = ControlCommand()

            pos_x = self.car_pose.position.x
            pos_y = self.car_pose.position.y
            q = self.car_pose.orientation
            v_curr = np.sqrt(self.speed.twist.twist.linear.x ** 2 + self.speed.twist.twist.linear.y ** 2)

            car_yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z)

            kp = 0.4
            ki = 0.005
            kd = 0.02
            dt_vel = time.time() - self.t1
            self.t1 = time.time()

            closest_waypoint_index = np.argmin((pos_x - self.x) ** 2 + (pos_y - self.y) ** 2)
            [throttle, brake, self.integral, self.vel_error, diffn] = vel_controller2(kp=kp, ki=ki, kd=kd,
                                                                                      v_curr=v_curr,
                                                                                      v_ref=self.v[closest_waypoint_index],
                                                                                      dt=dt_vel,
                                                                                      prev_integral=self.integral,
                                                                                      prev_vel_error=self.vel_error)

            [steer_pp, x_p, y_p] = pure_pursuit(x=self.x, y=self.y, vf=v_curr, pos_x=pos_x, pos_y=pos_y, veh_head=car_yaw,
                                                pos=closest_waypoint_index)

            self.visualize_pp_waypoint(x_pp=x_p, y_pp=y_p)

            car_controls_msg.throttle = throttle
            car_controls_msg.brake = brake
            car_controls_msg.steering = steer_pp

            self.car_controls.publish(car_controls_msg)

        else:
            cli = self.create_client(Reset, '/fsds/reset')
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            req = Reset.Request()
            req.wait_on_last_task = True
            future = cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            print('Reset done...')
            rclpy.shutdown("Time khatam")
            pass


def main(args=None):
    rclpy.init(args=args)

    batgrip = dv_car()

    rclpy.spin(batgrip)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    batgrip.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()