from dv_msgs.msg import VCU2AIStatus, VCU2AISteer, VCU2AISpeeds
from dv_msgs.msg import AI2VCUStatus
import rclpy
from rclpy.node import Node
from dv_msgs.msg import AI2VCUBrake, AI2VCUDriveR, AI2VCUDriveF, AI2VCUSteer
from dv_msgs.msg import VCU2AIStatus, VCU2AISteer, VCU2AISpeeds
from ackermann_msgs.msg import AckermannDriveStamped
from eufs_msgs.msg import WheelSpeedsStamped, CanState
import time
from std_msgs.msg import Bool
import numpy as np

class InspectionA(Node):

    def __init__(self):
        super().__init__('static_inspection_a')
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/cmd', 1)
        self.mission_flag_pub = self.create_publisher(Bool, '/ros_can/mission_completed', 1)
        self.driving_flag_pub = self.create_publisher(Bool, '/state_machine/driving_flag', 1)
        self.create_subscription(WheelSpeedsStamped, '/ros_can/wheel_speeds', self.speeds_callback, 1)
        self.create_subscription(CanState, '/ros_can/state', self.state_callback, 1)

        self.go = 0
        self.left_steer_phase = True
        self.right_steer_phase = False
        self.straight_steer_phase = False
        self.acceleration_phase = False
        self.brake_phase = False
        self.finished = False
        self.as_state = 1
        self.ami_state = 10

        self.t=time.time()-1
        self.integral = 0
        self.error=0
        self.prev_error = 0
        self.vel_error = 0

        self.send_default_signal()

        self.timer = self.create_timer(0.01, self.timer_callback)

    
    def send_default_signal(self):
        msg = AckermannDriveStamped()
        msg.drive.acceleration = 0.0
        msg.drive.jerk = 0.0
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        msg.drive.steering_angle_velocity = 0.0
        self.cmd_pub.publish(msg)


    def speeds_callback(self, msg: WheelSpeedsStamped):
        self.fl_speed = msg.speeds.lf_speed
        self.rr_speed = msg.speeds.rb_speed
        self.fr_speed = msg.speeds.rf_speed
        self.rl_speed = msg.speeds.lb_speed
        self.curr_steer_angle = msg.speeds.steering

    def state_callback(self, msg: CanState):
        self.as_state = msg.as_state
        # print(self.as_state)
        self.ami_state = msg.ami_state

    def torque_PID(self, rpm_requested, kp = 0.25, ki = 0.01):

        dt = time.time() - self.t
        self.t = time.time()

        error = rpm_requested - self.rl_speed

        self.integral = self.integral + error * dt

        pid_output = kp * error + ki * self.integral
        
        if rpm_requested == 0:
            pid_output = 0 
        
        pid_output = min(max(0, pid_output), 195)
        return pid_output


    def timer_callback(self):
        # self.get_logger().info(f'Phase(State Machine):{[self.left_steer_phase, self.right_steer_phase, self.straight_steer_phase, self.acceleration_phase, self.brake_phase, self.finished]}')
        if self.as_state == CanState.AS_DRIVING: # When go signal is received
            # The complete sequence of the inspection A
            self.get_logger().info('Static A Starting', once = True)
            driving_flag_msg = Bool()
            driving_flag_msg.data = True
            self.driving_flag_pub.publish(driving_flag_msg)
            if self.left_steer_phase:
                self.get_logger().info(f'A - Left Steer Phase (steer:{self.curr_steer_angle})', throttle_duration_sec=1)
                cmd_msg = AckermannDriveStamped()
                cmd_msg.drive.acceleration = 0.0
                cmd_msg.drive.speed = 0.0
                cmd_msg.drive.steering_angle = np.radians(-24.0)
                cmd_msg.drive.steering_angle_velocity = 0.0
                self.cmd_pub.publish(cmd_msg)
                
                if self.curr_steer_angle > np.radians(23):
                    self.left_steer_phase = False
                    self.right_steer_phase = True
            elif self.right_steer_phase:
                self.get_logger().info(f'A - Right steer phase (steer:{self.curr_steer_angle})', throttle_duration_sec = 1)
                cmd_msg = AckermannDriveStamped()
                cmd_msg.drive.acceleration = 0.0
                cmd_msg.drive.speed = 0.0
                cmd_msg.drive.steering_angle = np.radians(24.0)
                cmd_msg.drive.steering_angle_velocity = 0.0
                self.cmd_pub.publish(cmd_msg)
                if self.curr_steer_angle < np.radians(-23.0):
                    self.right_steer_phase = False
                    self.straight_steer_phase = True
            elif self.straight_steer_phase:
                self.get_logger().info(f'A - Centering the steering (steer:{self.curr_steer_angle})', throttle_duration_sec = 1)
                cmd_msg = AckermannDriveStamped()
                cmd_msg.drive.acceleration = 0.0
                cmd_msg.drive.speed = 0.0
                cmd_msg.drive.steering_angle = 0.0
                cmd_msg.drive.steering_angle_velocity = 0.0
                self.cmd_pub.publish(cmd_msg)
                if abs(self.curr_steer_angle) < np.radians(1.0):
                    self.straight_steer_phase = False
                    self.acceleration_phase = True
            elif self.acceleration_phase:
                self.get_logger().info(f'A - Acceleration Phase (rpm:{self.rl_speed})', throttle_duration_sec = 1)
                
                # kp =  0.2
                # ki =  0.18
                # kd =  0.00
                # rpm_ref = 200
                # dt_vel = time.time() - self.t   #Used to obtain the time difference for PID control.
                # self.t = time.time()
                # [throttle, brake,self.integral,self.vel_error,diffn] = rpm_controller(kp=kp, ki=ki, kd=kd,
                #                                         v_curr=self.rl_speed, v_ref=rpm_ref,
                #                                         dt=dt_vel, prev_integral=self.integral, prev_vel_error=self.vel_error)
                throttle = self.torque_PID(rpm_requested=300)
                cmd_msg = AckermannDriveStamped()
                cmd_msg.drive.acceleration = float(throttle)
                cmd_msg.drive.speed = 4000.0 # Requesting Max
                cmd_msg.drive.jerk = 0.0
                cmd_msg.drive.steering_angle = 0.0
                cmd_msg.drive.steering_angle_velocity = 0.0
                self.cmd_pub.publish(cmd_msg)
        
                if self.rl_speed>= 200.0 and self.rr_speed>=200.0:
                    self.acceleration_phase = False
                    self.brake_phase = True
            elif self.brake_phase:
                self.get_logger().info(f'A - Braking Phase (rpm:{self.rl_speed})', throttle_duration_sec = 1)

                cmd_msg = AckermannDriveStamped()
                cmd_msg.drive.acceleration = 0.0
                cmd_msg.drive.speed = 0.0
                cmd_msg.drive.jerk = 90.0
                cmd_msg.drive.steering_angle = 0.0
                cmd_msg.drive.steering_angle_velocity = 0.0
                self.cmd_pub.publish(cmd_msg)
                if self.rl_speed == 0.0 and self.rr_speed == 0.0:
                    self.brake_phase = False
                    self.finished = True
            elif self.finished:
                self.get_logger().info('A - Mission Finished', throttle_duration_sec = 1)
                mission_flag_msg = Bool()
                mission_flag_msg.data = True
                self.mission_flag_pub.publish(mission_flag_msg)
        else: # When no go signal is received
            # self.get_logger().info(f'no go {self.as_state}')
            driving_flag_msg = Bool()
            driving_flag_msg.data = False
            self.driving_flag_pub.publish(driving_flag_msg)


def main(args=None):
    rclpy.init(args=args)

    inspection_a = InspectionA()

    rclpy.spin(inspection_a)


    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    inspection_a.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()