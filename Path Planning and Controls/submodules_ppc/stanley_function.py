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

    min_cte = 10000
    i=1
    for i in range(len(final_x)):
        # dist = np.hypot((final_x[i] - f_tire_x),(final_y[i] - f_tire_y))
        dist = np.sqrt((final_x[i] - f_tire_x)**2 + (final_y[i] - f_tire_y)**2)
        
        if dist < min_cte:
            min_cte = dist
            min_cte_index=i
            # print(f"min_cte :{i} distance: {dist}")
    # print("DONEEEEEEEEEEEEEEEEE")
    cte = min_cte
    front_axle_vector=np.array([math.cos(car_yaw), math.sin(car_yaw)])
    nearest_path_vector=np.array([(final_x[min_cte_index] - f_tire_x),(final_y[min_cte_index] - f_tire_y)])
    cross_prod = np.cross(front_axle_vector, nearest_path_vector)
    
    if(cross_prod > 0):
        cte = -1*cte
    
    yaw_error = car_yaw - theta_p[min_cte_index] 
    steering_angle = (yaw_error/10 + math.atan((k * cte) / (v_curr+1)))
    print(f"yaw term {yaw_error}, {cte}, cte term{math.atan((k * cte) / (v_curr+1))}, steer :{steering_angle} , index : {min_cte_index}")
    print(f"final_x : {final_x[min_cte_index]}, final_y : {final_y[min_cte_index]}")
    
    steering_angle = max(-max_steering_angle, min(max_steering_angle, steering_angle))

    return steering_angle, final_x[min_cte_index], final_y[min_cte_index]