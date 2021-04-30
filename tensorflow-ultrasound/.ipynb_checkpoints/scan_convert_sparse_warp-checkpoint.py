
# New Scan Convert function
def scan_convert_with_sparse_warp(image, ele, y_seg=17, x_seg=5):
    image_padded, start_points, end_points = scan_convert_tf(image, ele, y_seg, x_seg)
    res = fan_out_tf(image_padded, ele, start_points, end_points)
    return res


@tf.function
def scan_convert_tf(image, ele, y_seg, x_seg):
    '''All coordinates are set as [y,x]'''

    # Initialize variables
    height, width = image.shape[0], image.shape[1]
    irad, frad = ele['initial_radius'], ele['final_radius']
    iang, fang, = ele['initial_angle'], ele['final_angle']

    # Pad image
    horizontal_pad = tf.math.floor(frad*tf.sin(fang) - width/2)
    vertical_pad = tf.math.floor(irad)
    image_padded = tf.pad(image, [[vertical_pad, 0], 
                                    [horizontal_pad, horizontal_pad]], 
                                    "CONSTANT")
    
    # Initialize start points
    y = tf.cast(tf.linspace(0, height, y_seg), tf.float32)

    y = tf.cast(tf.linspace(0, height, y_seg), tf.float32) + vertical_pad
    x = tf.cast(tf.linspace(0, width, x_seg), tf.float32) + horizontal_pad
    x, y = tf.meshgrid(x, y)
    y, x = tf.reshape(y, [-1]), tf.reshape(x, [-1])
    start_points = tf.stack([y,x], axis=-1)

    # Generate end points
    angles = tf.linspace(-1.0, 1.0, x_seg) * fang
    angles = tf.tile(angles , tf.constant([y_seg], tf.int32))
    # x_coords = start_points[:,1]
    x_coords = start_points[:,1]
    y_coords = start_points[:,0]
    r = tf.sqrt(tf.square(tf.abs(x_coords - (horizontal_pad + width/2))) + tf.square(y_coords - irad))
    y_converted = y_coords - y_coords * (1-tf.cos(angles)) # y coordinates conversion
    x_converted = x_coords + y_coords * tf.sin(angles)   # x coordinates conversion

    end_points = tf.stack([y_converted, x_converted], axis=-1)

    # Return padded image, start points, and end points
    start_points = tf.expand_dims(start_points, axis=0)
    end_points = tf.expand_dims(end_points, axis=0)

    return image_padded, tf.round(start_points), tf.round(end_points)


@tf.function
def fan_out_tf(image_padded, ele, start_points, end_points):
    # Returns a fanout using padded image, start points, and end points
    res, _ = tfa.image.sparse_image_warp(image_padded, start_points, end_points)

    # Creating mask to remove artifacts from sparse warp
    height, width = image_padded.shape[0], image_padded.shape[1]
    X, Y = tf.meshgrid(tf.linspace(0, width, width), tf.linspace(0, height, height))
    dist_from_center = tf.cast(tf.sqrt((X - width/2)**2 + (Y - 0)**2), tf.float32)
    mask = dist_from_center <= ele['final_radius']
    zeros = tf.zeros(shape=(height,width))

    # Masking result image
    res = tf.where(mask, res, zeros)
    return res[tf.cast(ele['initial_radius'], tf.int32):,:]

