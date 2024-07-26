from dv_msgs.msg import VCU2AIStatus, VCU2AISteer, VCU2AISpeeds
from dv_msgs.msg import AI2VCUStatus
from ackermann_msgs.msg import AckermannDriveStamped
import rclpy
import time
# from trajectory.submodules_ppc.trajectory_packages import *
from std_msgs.msg import Bool
from eufs_msgs.msg import CanState,WheelSpeedsStamped
from rclpy.node import Node
from dv_msgs.msg import AI2VCUBrake, AI2VCUDriveR, AI2VCUDriveF, AI2VCUSteer
from dv_msgs.msg import VCU2AIStatus, VCU2AISteer, VCU2AISpeeds

from std_srvs.srv import Trigger


class InspectionB(Node):

    def __init__(self):
        super().__init__('static_inspection_b')
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/cmd',1)
        self.mission_flag_pub= self.create_publisher(Bool, '/ros_can/mission_flag', 1)
        self.driving_flag_pub = self.create_publisher(Bool, '/state_machine/driving_flag',1)
            
        self.create_subscription(CanState, '/ros_can/state', self.status_callback, 10)
        self.create_subscription(WheelSpeedsStamped, '/ros_can/wheel_speeds', self.speeds_callback, 10)

        self.ebs_srv = self.create_client(Trigger, "/ros_can/ebs")


        self.go = 0
        self.initial_phase = True
        self.em_brake_phase = False
        self.rl_speed = 0
        self.rr_speed = 0
        self.as_state = 1
        self.t=time.time()-1
        self.integral = 0
        self.error=0
        self.prev_error = 0
        self.vel_error = 0
        self.timer = self.create_timer(0.01, self.timer_callback)

    def status_callback(self, msg: CanState):
        self.as_state = msg.as_state
        self.ami_state = msg.ami_state

    
    def speeds_callback(self, msg: WheelSpeedsStamped):
        self.fl_speed = msg.speeds.lf_speed
        self.fr_speed = msg.speeds.rf_speed
        self.rl_speed = msg.speeds.lb_speed
        self.rr_speed = msg.speeds.rb_speed

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
        if self.as_state == CanState.AS_DRIVING:
            if self.initial_phase:
                self.get_logger().info(f'Initial phase rpm: {self.rl_speed}', throttle_duration_sec=0.5)
                driving_flag_msg = Bool()
                driving_flag_msg.data = True
                self.driving_flag_pub.publish(driving_flag_msg)
                
                # kp =  0.2
                # ki =  0.18
                # kd =  0.00
                # rpm_ref = 50
                # dt_vel = time.time() - self.t   #Used to obtain the time difference for PID control.
                # self.t = time.time()
                # [throttle, brake,self.integral,self.vel_error,diffn] = rpm_controller(kp=kp, ki=ki, kd=kd,
                #                                         v_curr=self.rl_speed, v_ref=rpm_ref,
                #                                         dt=dt_vel, prev_integral=self.integral, prev_vel_error=self.vel_error)

                throttle = self.torque_PID(rpm_requested=100)
                cmd_msg = AckermannDriveStamped()
                cmd_msg.drive.acceleration = float(throttle)
                cmd_msg.drive.speed = 4000.0 # Requesting Max
                cmd_msg.drive.jerk = 0.0
                cmd_msg.drive.steering_angle = 0.0
                cmd_msg.drive.steering_angle_velocity = 0.0
                self.cmd_pub.publish(cmd_msg)
                
                if self.rl_speed>= 50.0 and self.rr_speed>=50.0:
                    # self.timer_1 = self.create_timer(2.0, self.embrake_phase)
                    self.get_logger().info("Requesting EBS")

                    if self.ebs_srv.wait_for_service(timeout_sec=1):
                        request = Trigger.Request()
                        result = self.ebs_srv.call_async(request)
                        self.get_logger().info("EBS successful")
                        self.get_logger().info(result)
                    else:
                        self.get_logger().info(
                            "/ros_can/ebs service is not available")
        else: # no go
            pass

            


def main(args=None):
    rclpy.init(args=args)

    inspection_b = InspectionB()

    rclpy.spin(inspection_b)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    inspection_b.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()