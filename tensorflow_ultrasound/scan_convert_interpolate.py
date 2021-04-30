import tensorflow as tf

@tf.function
def scan_convert_interpolate_precompute(image, y_seg, x_seg, irad, frad, iang, fang):
    '''
    Step 0: Initialize variables for use
    '''
    ### Initialize dimension and shift variables ###
    image_dim = tf.shape(image) # Image dimensions [height, width]
    image_height, image_width = tf.gather(image_dim, 0), tf.gather(image_dim, 1)

    horizontal_shift = tf.round(tf.math.multiply(-1.0, tf.cast(tf.math.divide(image_width, 2), tf.float32)))
    vertical_shift = tf.round(irad)

    ### Initialize dimensions for result image and create zerod tensor ### 
    a1 = tf.math.multiply(frad, tf.sin(fang))
    a2 = tf.cast(tf.math.divide(image_width, 2), tf.float32)
    horizontal_pad = tf.cast(tf.round(tf.math.subtract(a1, a2)), tf.int32)
    vertical_pad = tf.cast(tf.round(irad), tf.int32)
    res_height = image_height + vertical_pad
    res_width = image_width + 2 * horizontal_pad

    empty_res_image = tf.zeros([res_height+1, res_width+1], tf.float32)


    '''
    Step 1: Obtain grid points (points of interpolation) in r-theta and x-y space
            Additionally, obtain the max/min values of the interpolations in x-y space
    '''
    ### Initialize grid points (x, y, and val) in r-theta space ###
    grid_y_rtheta = tf.cast(tf.linspace(0, image_height, y_seg+1), tf.float32) + vertical_shift
    grid_x_rtheta = tf.cast(tf.linspace(0, image_width, x_seg+1), tf.float32) + horizontal_shift
    grid_x_rtheta, grid_y_rtheta = tf.meshgrid(grid_x_rtheta, grid_y_rtheta)
    grid_y_rtheta, grid_x_rtheta = tf.reshape(grid_y_rtheta, [-1]), tf.reshape(grid_x_rtheta, [-1])
    grid_vals_rtheta = tf.gather(image, tf.cast(tf.stack([grid_y_rtheta,grid_x_rtheta], axis=-1), tf.int32))

    ### Initialize grid points (x, y) in x-y space ###
    angles = tf.math.divide(
        tf.math.subtract(grid_x_rtheta, tf.reduce_min(grid_x_rtheta)), 
        tf.math.subtract(tf.reduce_max(grid_x_rtheta), tf.reduce_min(grid_x_rtheta))
        )
    angles = tf.math.subtract(tf.math.multiply(2.0, angles), 1)
    angles = tf.math.multiply(angles, (fang))
    radius = grid_y_rtheta

    grid_y_xy = tf.multiply(radius, tf.cos(angles))
    grid_x_xy = tf.multiply(radius, tf.sin(angles))

    ### Generate max and min pairs in grid_xy plane ###
    points_min_x, points_min_y = tf.math.floor(tf.reduce_min(grid_x_xy)), tf.math.floor(tf.reduce_min(grid_y_xy))
    points_max_x, points_max_y = tf.math.ceil(tf.reduce_max(grid_x_xy)), tf.math.ceil(tf.reduce_max(grid_y_xy))



    '''
    Step 2: Obtain points of interest in x-y plane and create mask that indicates 
            which ones are within bounds of given r-theta space.
    '''
    ### Initialize all points within x-y space ###
    points_y_xy = tf.cast(tf.range(points_min_y, points_max_y, 1), tf.float32) 
    points_x_xy = tf.cast(tf.range(points_min_x, points_max_x, 1), tf.float32)
    points_x_xy, points_y_xy = tf.meshgrid(points_x_xy, points_y_xy)
    points_y_xy, points_x_xy = tf.reshape(points_y_xy, [-1]), tf.reshape(points_x_xy, [-1])

    ### Mask for points of interest in x-y space ###
    dist_from_center = tf.sqrt(tf.square(points_y_xy) + tf.square(points_x_xy))
    points_radial_mask = tf.math.logical_and(dist_from_center > irad, dist_from_center < frad)

    angle_limit1 = tf.cast(tf.math.multiply(points_x_xy, tf.math.divide(1.0, iang)), tf.float32)
    angle_limit2 = tf.cast(tf.math.multiply(points_x_xy, tf.math.divide(1.0, fang)), tf.float32)
    points_angle_mask = tf.math.logical_and(angle_limit1 < points_y_xy, angle_limit2 < points_y_xy)

    points_mask = tf.math.logical_and(points_radial_mask, points_angle_mask)

    ### Masking points of interest in x-y space ###
    points_y_xy_masked = tf.boolean_mask(points_y_xy, points_mask)
    points_x_xy_masked = tf.boolean_mask(points_x_xy, points_mask)
    points_xy = tf.cast(tf.stack([points_y_xy_masked, points_x_xy_masked + tf.cast(res_width/2, tf.float32)], axis=-1), tf.int32)
    # return points_xy in y, x form for indexing



    '''
    Step 3: Interpolate points of interest in x-y to r-theta
    '''
    ### Obtain transformation from x-y to r-theta for points of interest ###
    angles = tf.math.atan(tf.math.divide(points_x_xy_masked, points_y_xy_masked))
    radius = tf.sqrt(tf.square(points_x_xy_masked) + tf.square(points_y_xy_masked))

    # Points of interest x coordinate conversion
    x_scale = tf.math.divide(
                    tf.math.subtract(angles, tf.reduce_min(angles)), 
                    tf.math.subtract(tf.reduce_max(angles), tf.reduce_min(angles))
                )
    x_scale = tf.math.subtract(tf.math.multiply(2.0, x_scale), 1)
    scale = tf.cast(tf.math.divide(tf.gather(tf.shape(image), [1]), 2), tf.float32)
    points_x_rtheta_masked = (scale - (-1 * scale)) * tf.math.divide(
                                tf.math.subtract(x_scale, tf.reduce_min(x_scale)), 
                                tf.math.subtract(tf.reduce_max(x_scale), tf.reduce_min(x_scale))
                            ) + (-1 * scale)
    
    # Points of interest y coordinate conversion
    points_y_rtheta_masked = radius



    '''
    Step 4: Find weights (w1, w2, w3, w4) for each point of interest in r-theta plane 
            (for bilinear interpolation)
    '''
    ### Normalize image to range [0, x_seg-1] and [0, y_seg-1]
    points_x_memspace = (x_seg-1 - 0) * tf.math.divide(
                            tf.math.subtract(points_x_rtheta_masked, tf.reduce_min(points_x_rtheta_masked)), 
                            tf.math.subtract(tf.reduce_max(points_x_rtheta_masked), tf.reduce_min(points_x_rtheta_masked))
                        ) + 0
    points_y_memspace = (y_seg-1 - 0) * tf.math.divide(
                            tf.math.subtract(points_y_rtheta_masked, tf.reduce_min(points_y_rtheta_masked)), 
                            tf.math.subtract(tf.reduce_max(points_y_rtheta_masked), tf.reduce_min(points_y_rtheta_masked))
                        ) + 0

    ### Floor and ceil points of interest ###
    points_x_floor = tf.cast(tf.math.floor(points_x_memspace), tf.int32)
    points_x_ceil = tf.cast(tf.math.ceil(points_x_memspace), tf.int32)
    points_y_floor = tf.cast(tf.math.floor(points_y_memspace), tf.int32)
    points_y_ceil = tf.cast(tf.math.ceil(points_y_memspace), tf.int32)

    ### Gather coordinates of each value relative to the points of interest; Generate [x,y] pairs ###
    points_rtheta = tf.stack([points_y_rtheta_masked, points_x_rtheta_masked], axis=-1)
    grid_y_rtheta_shifted = tf.cast(tf.linspace(0, image_height, y_seg+1), tf.float32) + vertical_shift
    grid_x_rtheta_shifted = tf.cast(tf.linspace(0, image_width, x_seg+1), tf.float32) + horizontal_shift

    val1_rtheta = tf.stack([tf.gather(grid_y_rtheta_shifted, points_y_floor), tf.gather(grid_x_rtheta_shifted, points_x_floor)], axis=-1)
    val2_rtheta = tf.stack([tf.gather(grid_y_rtheta_shifted, points_y_floor), tf.gather(grid_x_rtheta_shifted, points_x_ceil)], axis=-1)
    val3_rtheta = tf.stack([tf.gather(grid_y_rtheta_shifted, points_y_ceil), tf.gather(grid_x_rtheta_shifted, points_x_ceil)], axis=-1)
    val4_rtheta = tf.stack([tf.gather(grid_y_rtheta_shifted, points_y_ceil), tf.gather(grid_x_rtheta_shifted, points_x_floor)], axis=-1)

    ### Calculate weighting for each value relative to the points of interest ###
    val1_dist = tf.norm(val1_rtheta - points_rtheta, axis=-1)
    val2_dist = tf.norm(val2_rtheta - points_rtheta, axis=-1)
    val3_dist = tf.norm(val3_rtheta - points_rtheta, axis=-1)
    val4_dist = tf.norm(val4_rtheta - points_rtheta, axis=-1)
    val_total_dist = val1_dist + val2_dist + val3_dist + val4_dist

    val1_weights = tf.math.divide(val1_dist, val_total_dist)
    val2_weights = tf.math.divide(val2_dist, val_total_dist)
    val3_weights = tf.math.divide(val3_dist, val_total_dist)
    val4_weights = tf.math.divide(val4_dist, val_total_dist)
    val_weights = tf.stack([val1_weights, val2_weights, val3_weights, val4_weights], axis=-1)
    # val_weights = tf.math.divide(val_weights, val_total_dist)

    ### Shifted version of bilinear coordinates that fit into the image dimensions ###
    grid_y_rtheta_shifted -= vertical_shift
    grid_x_rtheta_shifted -= horizontal_shift
    val1_rtheta = tf.stack([tf.gather(grid_y_rtheta_shifted, points_y_floor), tf.gather(grid_x_rtheta_shifted, points_x_floor)], axis=-1)
    val2_rtheta = tf.stack([tf.gather(grid_y_rtheta_shifted, points_y_floor), tf.gather(grid_x_rtheta_shifted, points_x_ceil)], axis=-1)
    val3_rtheta = tf.stack([tf.gather(grid_y_rtheta_shifted, points_y_ceil), tf.gather(grid_x_rtheta_shifted, points_x_ceil)], axis=-1)
    val4_rtheta = tf.stack([tf.gather(grid_y_rtheta_shifted, points_y_ceil), tf.gather(grid_x_rtheta_shifted, points_x_floor)], axis=-1)
    val_rtheta = tf.stack([val1_rtheta, val2_rtheta, val3_rtheta, val4_rtheta], axis=-1)

    return empty_res_image, points_xy, val_rtheta, val_weights
  
  



@tf.function
def scan_convert_interpolate_dynamic(image, empty_res_image, irad, points_xy, val_rtheta, val_weights):
    '''
    Step 5: Dynamically find pixel values for all four bilinear point for each of interest in r-theta plane
    '''
    ### Unstack and find all pixel values for bilinear interpolation ###
    val_rtheta = tf.unstack(val_rtheta, axis=2)
    val1 = tf.squeeze(tf.gather(val_rtheta, [0], axis=0))
    val2 = tf.squeeze(tf.gather(val_rtheta, [1], axis=0))
    val3 = tf.squeeze(tf.gather(val_rtheta, [2], axis=0))
    val4 = tf.squeeze(tf.gather(val_rtheta, [3], axis=0))

    val1 = tf.gather_nd(image, tf.cast(val1, tf.int32))
    val2 = tf.gather_nd(image, tf.cast(val2, tf.int32))
    val3 = tf.gather_nd(image, tf.cast(val3, tf.int32))
    val4 = tf.gather_nd(image, tf.cast(val4, tf.int32))

    ### Unstack and find all weights for bilinear interpolation ###
    val_weights = tf.unstack(val_weights, axis=1)
    val1_weights = tf.squeeze(tf.gather(val_weights, [0], axis=0))
    val2_weights = tf.squeeze(tf.gather(val_weights, [1], axis=0))
    val3_weights = tf.squeeze(tf.gather(val_weights, [2], axis=0))
    val4_weights = tf.squeeze(tf.gather(val_weights, [3], axis=0))

    ### Multiply weights with pixel values ###
    val1 = tf.math.multiply(val1_weights, val1)
    val2 = tf.math.multiply(val2_weights, val2)
    val3 = tf.math.multiply(val3_weights, val3)
    val4 = tf.math.multiply(val4_weights, val4)

    ### Obtain point values at each point of interest (currently in r-theta space) ###
    points_val = val1 + val2 + val3 + val4



    '''
    Step 6: Remap all points value back to x-y space at the correct coordinates
    '''
    ### Populate zerod tensor with values ###
    res = tf.tensor_scatter_nd_add(empty_res_image, points_xy, points_val)

    return res[tf.cast(irad, tf.int32):, :]

