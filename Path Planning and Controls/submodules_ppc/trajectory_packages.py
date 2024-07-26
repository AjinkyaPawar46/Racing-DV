from cmath import nan
import math
import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy
import matplotlib.pyplot as plt
import time

def mapper(path_loc):
    map_file = open(path_loc,"r")

    line_num = 3
    lines_list = map_file.readlines()

    total_lines = len(lines_list)
    total_cones = int((total_lines-2)/6)

    all_cones = np.empty((total_cones, 3))

    # Initialise values
    row_index_x = 3
    row_index_y = 4
    row_index_color = 6

    for i in range(total_cones):
        # add x, y, color to main array
        all_cones[i, 0] = lines_list[row_index_x][9:-1]
        all_cones[i, 1] = lines_list[row_index_y][9:-1]
        all_cones[i, 2] = lines_list[row_index_color][11:-1]
        # update new index
        row_index_x += 6
        row_index_y += 6
        row_index_color += 6
    all_cones_list = all_cones.tolist()

    # Separate into blue and yellow cones
    blue_cones = np.array([cone for cone in all_cones_list if cone[2]==0])      # BLUE = 0
    yellow_cones = np.array([cone for cone in all_cones_list if cone[2]==1])    # YELLOW = 1
    return [blue_cones,yellow_cones]

def get_bezier_coef(points):
   # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B

# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

# return one cubic curve for each consecutive points
def get_bezier_cubic(points):
    A, B = get_bezier_coef(points)
    return [
        get_cubic(points[i], A[i], B[i], points[i + 1])
        for i in range(len(points) - 1)
    ]

# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, n):
    curves = get_bezier_cubic(points)
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n, endpoint = False)])

#Interpolates the path to obtain the path to be followed
def interpolate(blue_cones, yellow_cones):
    [x, y] = mid_point(yellow_cones[:, 0], yellow_cones[:, 1], blue_cones[:, 0], blue_cones[:, 1])
    x2 = [x[len(x) - 2], x[len(x) - 1], x[0], x[1]]
    y2 = [y[len(y) - 2], y[len(y) - 1], y[0], y[1]]

    points = np.column_stack([x, y]) #Covers the points from the start to the end point.
    points_extra = np.column_stack([x2,y2]) #Covers the points from the end to the start point. 

    path = evaluate_bezier(points, 5)
    path_extra = evaluate_bezier(points_extra, 5)

    x = path[:,0]
    y = path[:,1]
    x3 = path_extra[:,0]
    y3 = path_extra[:,1]

    x = np.concatenate((x, x3[5:10]))
    y = np.concatenate((y, y3[5:10]))

    return [x, y]

#Returns the central path. 
def mid_point(x_outer,y_outer,x_inner,y_inner):
    x = np.zeros(len(x_outer))
    y = np.zeros(len(y_outer))

    for i in range(len(x)):
        start = np.argmin((x_outer[i] - x_inner)**2 + (y_outer[i] - y_inner)**2)
        x[i] = (x_outer[i] + x_inner[start])/2
        y[i] = (y_outer[i] + y_inner[start])/2

    return [x, y]

def curvature(x, y):
    x = np.squeeze(np.array(x))
    y = np.squeeze(np.array(y))
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)

    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs((d2x_dt2 * dy_dt - dx_dt * d2y_dt2) /(dx_dt * dx_dt + dy_dt * dy_dt)**1.5)

    roc = np.zeros(curvature.size)

    for i in range(curvature.size):
        if curvature[i] < 0.001:
            roc[i] = 1000
        else:
            roc[i] = 1/curvature[i]
    return [curvature, roc]

def vel_find3(x, y, mu, m ,g):
    mu = 0.6   #coefficient of friction   
    m = 230     #mass of the car
    g = 9.8     #acceleration due to gravity
    
    [k, r]   = curvature(x, y)
 
    ds = np.zeros(len(x))  #distances between consecutive waypoints

    for i in range(len(x) - 1):
            ds[i] = math.sqrt((x[i + 1] - x[i])**2 + (y[i + 1] - y[i])**2)

    friction_available = mu*m*g
    v1 = np.sqrt(friction_available*r/m) #max allowed velocity on that waypoint- First profile not accounting accn and deceleration forces
    

    v2 = np.zeros(len(x))
    v2[0]=10   #Starting value
    print('Max Friction(in N):', friction_available)
    for i in range(0, len(x) - 1):

        centripetal_force_required = m*(v2[i]**2)/r[i]

        if friction_available>centripetal_force_required:
            available_tangential_force = math.sqrt(friction_available**2 - centripetal_force_required**2)

        else:
            v2[i+1] = min(v2[i],v1[i+1])
            continue

        
        upper_cap = math.sqrt(v2[i]**2 + 2*(available_tangential_force/m)*ds[i]) #using max acceleration
        lower_cap = math.sqrt(max(v2[i]**2 - 2*(available_tangential_force/m)*ds[i],0)) #using max deceleration
        
        if v1[i+1] > upper_cap:
            v2[i+1] = upper_cap
        elif v1[i+1] < lower_cap:
            v2[i+1] = lower_cap
        else :
            v2[i+1] = v1[i+1]
    

    
    v_final = np.copy(v2)
    
    for i in range(len(v_final)-1, 0,-1):

        centripetal_force_required = m*(v_final[i]**2)/r[i]

        if friction_available>centripetal_force_required:
            available_tangential_force = math.sqrt(friction_available**2 - centripetal_force_required**2)

        else:
            v_final[i-1] = min(v_final[i],v2[i-1])
            continue

        
        upper_cap = math.sqrt(v_final[i]**2 + 2*(available_tangential_force/m)*ds[i-1]) #using max acceleration
        lower_cap = math.sqrt(max(v_final[i]**2 - 2*(available_tangential_force/m)*ds[i-1],0)) #using max deceleration
        
        if v2[i-1] > upper_cap:
            v_final[i-1] = upper_cap
        elif v2[i-1] < lower_cap:
            v_final[i-1] = lower_cap
        else :
            v_final[i-1] = v2[i-1]



    expected_time = 0
    for i in range(len(ds)):
        expected_time +=ds[i]/v_final[i]


    return v2, expected_time, v1, r,v_final

def vel_controller2(prev_vel_error, v_curr, v_ref, dt, prev_integral, kp, ki, kd):
    error = v_ref - v_curr
    integral = prev_integral+error*dt
    diffn = (error - prev_vel_error)/dt
    pedal = kp * error + ki * integral + kd * diffn

    if pedal > 0:
        throttle = min(pedal,1)
        brake = 0
    else:
        throttle = 0
        brake = min(-pedal,1)

    return [throttle, brake,integral,error,diffn]

def pure_pursuit(x, y, vf, pos_x, pos_y, veh_head ,K = 0.4, L = 1.8, MAX_STEER = 22):
    '''
    L - Length of the car (in bicycle model?)
    look-ahead distance => tune minimum_look_ahead, K
    '''
    minimum_look_ahead = 1.5
    look_ahead_dist = minimum_look_ahead + K*vf

    # Remove points which are not of interest i.e the points which have been passed 
    # In first lap this part is redundant because we will only have points which lie ahead of us

    #   Necessary to initialise like this to be able to append using numpy, this point will always get discarded because its distance will be very high
    points_ahead = np.array([[0,0,10000]])  
    for i in range(len(x)):
        heading_vector = [math.cos(veh_head), math.sin(veh_head)]
        look_ahead_vector = [x[i] - pos_x ,y[i] - pos_y ]
        dot_product = np.dot(heading_vector, look_ahead_vector)
        # print(f'Dot:{dot_product}')
        if dot_product < 0:
            continue
        else:
            # Add how close is the distnace of the waypoint to the look ahead distance needed
            dist_waypoint = math.sqrt((x[i] - pos_x)**2 + (y[i] - pos_y)**2)
            points_ahead = np.append(points_ahead, [[x[i],y[i],abs(dist_waypoint - look_ahead_dist)]], axis = 0)
    
    # Remove the extra point added while creating the varible
    points_ahead = points_ahead[1:]
    
    #Select the waypoint which is closest after remo
    final_waypoint_index = np.argmin(points_ahead[:,2])
    waypoint_x = points_ahead[final_waypoint_index,0]
    waypoint_y = points_ahead[final_waypoint_index,1]
    
    # Angle of the vector connecting car to the waypoint
    theta = math.atan((waypoint_y - pos_y)/(waypoint_x - pos_x)) 

    if (waypoint_y - pos_y)*(waypoint_x - pos_x) < 0:
        if (waypoint_y - pos_y) > 0:
            theta = theta + math.pi
    else:
        if (waypoint_y - pos_y) < 0:
            theta = theta - math.pi

    #alpha_pp - the change in angle that should be made in car heading
    alpha_pp = theta - veh_head 
    waypoint_distance = ((pos_x - waypoint_x)**2 + (pos_y - waypoint_y)**2)**0.5

    steer = math.atan(2*L*math.sin(alpha_pp)/waypoint_distance) # steer in radians

    max_steer_radians = MAX_STEER * math.pi / 180

    #Clip the steering to max values
    final_steer = max( - max_steer_radians, min(steer , max_steer_radians))
    
    return [final_steer, waypoint_x, waypoint_y]

def stanley_steering(final_x, final_y, v_curr, pos_x, pos_y, car_yaw):
    """
    Calculate the steering angle based on the Stanley method for path tracking.

    Parameters:
    - final_x, final_y: Arrays of x and y coordinates of the path waypoints.
    - v_curr: Current velocity of the vehicle.
    - pos_x, pos_y: Current x and y position of the vehicle.
    - car_yaw: Current yaw angle of the vehicle in radians.

    Returns:
    - steering_angle: The recommended steering angle in radians.
    - nearest_x: The x coordinate of the nearest path point.
    - nearest_y: The y coordinate of the nearest path point.
    """
    if len(final_x) != len(final_y) or len(final_x) == 0:
        raise ValueError("Path coordinate arrays must be non-empty and of equal length.")
    
    k = 2.5  # Controller gain
    max_steering_angle = np.deg2rad(24)  # Maximum steering angle in radians
    front_axle_offset = 1.7  # Distance from vehicle position to front axle

    # Compute the position of the front tire
    f_tire_x = pos_x + front_axle_offset * math.cos(car_yaw)
    f_tire_y = pos_y + front_axle_offset * math.sin(car_yaw)

    # Compute path slopes and store them
    theta_p = [math.atan2(final_y[i+1] - final_y[i], final_x[i+1] - final_x[i]) for i in range(len(final_x)-1)]
    theta_p.append(math.atan2(final_y[0] - final_y[-1], final_x[0] - final_x[-1]))

    # Compute the nearest path point to the front tire
    distances = [np.hypot(final_x[i] - f_tire_x, final_y[i] - f_tire_y) for i in range(len(final_x))]
    min_cte_index = np.argmin(distances)
    min_cte = distances[min_cte_index]

    # Determine if the front axle should correct to the left or right
    front_axle_vector = np.array([math.cos(car_yaw), math.sin(car_yaw)])
    nearest_path_vector = np.array([final_x[min_cte_index] - f_tire_x, final_y[min_cte_index] - f_tire_y])
    cross_prod = np.cross(front_axle_vector, nearest_path_vector)
    cte = -min_cte if cross_prod < 0 else min_cte

    # if (abs(cte) > 1):
    #     cte=0
    # Calculate steering angle
    yaw_error = car_yaw - theta_p[min_cte_index]
    steering_angle = (yaw_error/10 + math.atan((k * cte) / (v_curr + 1)))
    steering_angle = max(-max_steering_angle, min(max_steering_angle, steering_angle))

    return steering_angle, final_x[min_cte_index], final_y[min_cte_index],min_cte_index

def genpath(bounds_left,bounds_right,params):
    #the first point should be in the center, assuming car starts there
    x=[]
    y=[]
    x.append(bounds_left[0,0]+0.5*(bounds_right[0,0]-bounds_left[0,0]))  
    y.append(bounds_left[0,1]+0.5*(bounds_right[0,1]-bounds_left[0,1]))
    for i in range(1,len(params)+1):
        x.append(bounds_left[i,0]+params[i-1]*(bounds_right[i,0]-bounds_left[i,0]))
        y.append(bounds_left[i,1]+params[i-1]*(bounds_right[i,1]-bounds_left[i,1]))
    return x,y

def genpath2(bounds_left,bounds_right,params):
    #doesnt assume the start point to be the center of the track
    x=[]
    y=[]
    for i in range(len(params)):
        x.append(bounds_left[i,0]+params[i-1]*(bounds_right[i,0]-bounds_left[i,0]))
        y.append(bounds_left[i,1]+params[i-1]*(bounds_right[i,1]-bounds_left[i,1]))
    return x,y

def sense_cones(pos_x, pos_y, blue_cones, yellow_cones ):
    cones_in_region=[]
    range=15*15  #squared
    for cone in blue_cones:
        if (pos_x-cone[0])**2+(pos_y-cone[1])**2 <= range:
            cones_in_region.append(cone)
    for cone in yellow_cones:
        if (pos_x-cone[0])**2+(pos_y-cone[1])**2 <= range:
            cones_in_region.append(cone)
    return cones_in_region

