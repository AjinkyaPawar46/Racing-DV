from dv_msgs.msg import VCU2AIStatus, VCU2AISteer, VCU2AISpeeds
from dv_msgs.msg import AI2VCUStatus
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from eufs_msgs.msg import CanState,WheelSpeedsStamped
import rclpy
from rclpy.node import Node
from dv_msgs.msg import AI2VCUBrake, AI2VCUDriveR, AI2VCUDriveF, AI2VCUSteer
from dv_msgs.msg import VCU2AIStatus, VCU2AISteer, VCU2AISpeeds, VCU2AIWheelcounts
from eufs_msgs.msg import ControlCommand
from std_msgs.msg import Float64

from std_srvs.srv import Trigger

import time
import numpy as np
PI = 3.1416
TICKS_PER_REV = 20
WHEEL_RADIUS = 0.505

class Demo(Node):

    def __init__(self):
        super().__init__('auto_demo')
        # self.steer_pub = self.create_publisher(AI2VCUSteer, '/AI2VCU_Steer', 1)
        # self.drive_r_pub = self.create_publisher(AI2VCUDriveR, '/AI2VCU_Drive_R', 1)
        # self.drive_f_pub = self.create_publisher(AI2VCUDriveF, '/AI2VCU_Drive_F', 1)
        # self.brake_pub = self.create_publisher(AI2VCUBrake, '/AI2VCU_Brake', 1)
        # self.status_pub = self.create_publisher(AI2VCUStatus, '/AI2VCU_Status', 1)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/cmd',1)
        self.mission_flag_pub= self.create_publisher(Bool, '/ros_can/mission_flag', 1)
        self.driving_flag_pub = self.create_publisher(Bool, '/state_machine/driving_flag',1)
        self.rpm_pub = self.create_publisher(Float64, '/rpm_data', 1)

        self.create_subscription(CanState, '/ros_can/state', self.status_callback, 10)
        self.create_subscription(WheelSpeedsStamped, '/ros_can/wheel_speeds', self.speeds_callback, 10)
        
        self.ebs_srv = self.create_client(Trigger, "/ros_can/ebs")
        # self.create_subscription(VCU2AIStatus, '/VCU2AI_Status', self.status_callback, 1)
        # self.create_subscription(VCU2AISpeeds, '/VCU2AI_Speeds', self.speeds_callback, 1)
        # self.create_subscription(VCU2AISteer, '/VCU2AI_Steer', self.steer_callback, 1)
        # self.create_subscription(VCU2AIWheelcounts, '/VCU2AIWheelcounts', self.counts_callback, 1)
        self.create_subscription(ControlCommand, '/cmd1', self.command_callback, 1)

        self.left_steer_phase = True
        self.right_steer_phase = False
        self.straight_steer_phase = False
        self.acceleration_phase_1 = False
        self.brake_phase = False
        self.acceleration_phase_2 = False
        self.ebs_phase = False
        self.t = time.time()-1
        self.fl_speed = 0
        self.fr_speed = 0
        self.rl_speed = 0
        self.rr_speed = 0
        self.curr_steer_angle = 0
        self.max_steer = 24.0
        self.distance_travelled = 0
        self.car_speed = 0
        self.prev_time = time.time()
        self.counts_in_tenmeter = TICKS_PER_REV * 10/(2 * 3.14 * WHEEL_RADIUS)
        self.counts = 0
        self.send_default_signal()

        self.timer = self.create_timer(0.01, self.timer_callback)

    def status_callback(self, msg: CanState):
        self.as_state = msg.as_state
        self.ami_state = msg.ami_state


    def command_callback(self, msg):
        self.ai_steer_req = msg.steering

    def torque_PID(self, rpm_requested, kp = 0.15, ki = 0.002):

        dt = time.time() - self.t
        self.t = time.time()

        error = rpm_requested - self.rl_speed

        self.integral = self.integral + error * dt

        pid_output = kp * error + ki * self.integral
        
        if rpm_requested == 0:
            pid_output = 0 
        
        pid_output = min(max(0, pid_output), 195)
        return pid_output
    
    def send_default_signal(self):
        
        cmd_msg = AckermannDriveStamped()
        cmd_msg.drive.acceleration = 0.0
        cmd_msg.drive.speed = 0.0
        cmd_msg.drive.jerk = 0.0
        cmd_msg.drive.steering_angle = 0.0
        cmd_msg.drive.steering_angle_velocity = 0.0
        self.cmd_pub.publish(cmd_msg)


    def speeds_callback(self, msg: WheelSpeedsStamped):
        self.fl_speed = msg.speeds.lf_speed
        self.rr_speed = msg.speeds.rb_speed
        self.fr_speed = msg.speeds.rf_speed
        self.rl_speed = msg.speeds.lb_speed
        self.curr_steer_angle = msg.speeds.steering
        car_rpm = (self.rl_speed + self.rr_speed)/2
        self.car_speed = car_rpm*WHEEL_RADIUS*PI/60  # in m/s


    # def steer_callback(self, msg: VCU2AISteer):
    #     self.curr_steer_angle = msg.angle
        # self.max_steer = msg.angle_max


    def timer_callback(self):
        # self.get_logger().info(f'Phase(State Machine):{[self.left_steer_phase, self.right_steer_phase, self.straight_steer_phase, self.acceleration_phase, self.brake_phase, self.finished]}')
        self.distance_travelled += self.car_speed*(time.time()-self.prev_time)
        self.prev_time = time.time()
        rpm_msg = Float64()
        rpm_msg.data = self.rr_speed
        self.rpm_pub.publish(rpm_msg)
        if self.as_state == CanState.AS_DRIVING:
            # The complete sequence of the inspection A
            self.get_logger().info('Autonmous Demo Starting', once = True)
            driving_flag_msg = Bool()
            driving_flag_msg.data = True
            self.driving_flag_pub.publish(driving_flag_msg)
            if self.left_steer_phase:
                self.get_logger().info(f'Demo - Left Steer Phase {self.curr_steer_angle}', throttle_duration_sec=1)
                cmd_msg = AckermannDriveStamped()
                cmd_msg.drive.acceleration = 0.0
                cmd_msg.drive.speed = 0.0
                cmd_msg.drive.steering_angle = np.radians(-24)
                cmd_msg.drive.steering_angle_velocity = 0.0
                self.cmd_pub.publish(cmd_msg)
                
                if self.curr_steer_angle > np.radians(23):
                    self.left_steer_phase = False
                    self.right_steer_phase = True
            elif self.right_steer_phase:
                self.get_logger().info(f'Demo - Right steer phase {self.curr_steer_angle}', throttle_duration_sec = 1)
                cmd_msg = AckermannDriveStamped()
                cmd_msg.drive.acceleration = 0.0
                cmd_msg.drive.speed = 0.0
                cmd_msg.drive.steering_angle = np.radians(24)
                cmd_msg.drive.steering_angle_velocity = 0.0
                self.cmd_pub.publish(cmd_msg)

                if self.curr_steer_angle < np.radians(-23.0):
                    self.right_steer_phase = False
                    self.straight_steer_phase = True
            elif self.straight_steer_phase:
                self.get_logger().info(f'Demo - Centering the steering {self.curr_steer_angle}', throttle_duration_sec = 1)
                cmd_msg = AckermannDriveStamped()
                cmd_msg.drive.acceleration = 0.0
                cmd_msg.drive.speed = 0.0
                cmd_msg.drive.steering_angle = 0.0
                cmd_msg.drive.steering_angle_velocity = 0.0
                self.cmd_pub.publish(cmd_msg)
                if abs(self.curr_steer_angle) < np.radians(1.0):
                    self.straight_steer_phase = False
                    self.acceleration_phase_1 = True
            elif self.acceleration_phase_1:
                self.get_logger().info(f'Demo - Acceleration Phase 1 {self.distance_travelled}', throttle_duration_sec = 1)
                torque = self.torque_PID(rpm_requested=250)
                cmd_msg = AckermannDriveStamped()
                cmd_msg.drive.acceleration = float(torque)
                cmd_msg.drive.speed = 4000.0 # Requesting Max
                cmd_msg.drive.jerk = 0.0
                cmd_msg.drive.steering_angle = self.ai_steer_req
                cmd_msg.drive.steering_angle_velocity = 0.0
                self.cmd_pub.publish(cmd_msg)

                if self.distance_travelled >= 10.0:
                    self.acceleration_phase_1 = False
                    self.brake_phase = True
                    
            elif self.brake_phase:
                self.get_logger().info(f'Demo - Braking Phase {self.rl_speed}', throttle_duration_sec = 1)
                cmd_msg = AckermannDriveStamped()
                cmd_msg.drive.acceleration = 0.0
                cmd_msg.drive.speed = 0.0
                cmd_msg.drive.jerk = 90.0
                cmd_msg.drive.steering_angle = 0.0
                cmd_msg.drive.steering_angle_velocity = 0.0
                self.cmd_pub.publish(cmd_msg)
                if self.rl_speed == 0.0 and self.rr_speed == 0.0:
                    self.brake_phase = False
                    self.acceleration_phase_2 = True
                    self.distance_travelled = 0
            elif self.acceleration_phase_2:
                self.get_logger().info(f'Demo - Acceleration Phase 2 {self.distance_travelled}', throttle_duration_sec = 1)
                torque = self.torque_PID(rpm_requested=250)
                cmd_msg = AckermannDriveStamped()
                cmd_msg.drive.acceleration = float(torque)
                cmd_msg.drive.speed = 4000.0 # Requesting Max
                cmd_msg.drive.jerk = 0.0
                cmd_msg.drive.steering_angle = 0.0
                cmd_msg.drive.steering_angle_velocity = 0.0
                self.cmd_pub.publish(cmd_msg)
                if self.distance_travelled >= 10.0:
                    self.acceleration_phase_2 = False
                    self.ebs_phase = True
            elif self.ebs_phase:
                self.get_logger().info('Demo - EM Brake', throttle_duration_sec = 1)
                if self.ebs_srv.wait_for_service(timeout_sec=1):
                    request = Trigger.Request()
                    result = self.ebs_srv.call_async(request)
                    self.get_logger().info("EBS successful")
                    self.get_logger().info(result)
                else:
                    self.get_logger().info(
                        "/ros_can/ebs service is not available")
        else: # When no go signal is received
            self.get_logger().info('Go not received', throttle_duration_sec = 1)
            driving_flag_msg = Bool()
            driving_flag_msg.data = False
            self.driving_flag_pub.publish(driving_flag_msg)        


def main(args=None):
    rclpy.init(args=args)

    auto_demo = Demo()

    rclpy.spin(auto_demo)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    auto_demo.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()