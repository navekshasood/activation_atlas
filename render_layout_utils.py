import numpy as np
import tensorflow as tf
from image_utils import initialize_image_ref
from optimization_utils import OptimizationParameters, visualize_layer

def render_icons(activations, model, layer, iterations, n_attempts, icon_size, step_size):
    batch = activations.shape[0]
    image = initialize_image_ref(batch, icon_size)
    print (image.shape)
    optimizer = tf.keras.optimizers.Adam(epsilon=1e-08, learning_rate=0.05)

    def my_trans(img):
        return img
    learning_rate = 0.7
    opt_param = OptimizationParameters(iterations, learning_rate, optimizer=optimizer)
    activation, image = visualize_layer(activations, image, model, layer, opt_param, transformation=my_trans)
    print ("Shape:", image.shape)
    return image

def grid(xpts=None, ypts=None, grid_size=(8,8), x_extent=(0., 1.), y_extent=(0., 1.)):
    xpx_length = grid_size[0]
    ypx_length = grid_size[1]

    xpt_extent = x_extent
    ypt_extent = y_extent

    xpt_length = xpt_extent[1] - xpt_extent[0]
    ypt_length = ypt_extent[1] - ypt_extent[0]

    xpxs = ((xpts - xpt_extent[0]) / xpt_length) * xpx_length
    ypxs = ((ypts - ypt_extent[0]) / ypt_length) * ypx_length

    ix_s = range(grid_size[0])
    iy_s = range(grid_size[1])
    xs = []
    for xi in ix_s:
        ys = []
        for yi in iy_s:
            xpx_extent = (xi, (xi + 1))
            ypx_extent = (yi, (yi + 1))

            in_bounds_x = np.logical_and(xpx_extent[0] <= xpxs, xpxs <= xpx_extent[1])
            in_bounds_y = np.logical_and(ypx_extent[0] <= ypxs, ypxs <= ypx_extent[1])
            in_bounds = np.logical_and(in_bounds_x, in_bounds_y)

            in_bounds_indices = np.where(in_bounds)[0]
            ys.append(in_bounds_indices)
        xs.append(ys)
    return np.asarray(xs)
    
def render_layout(model, layer, activations, xs, ys, S, raw_activations, n_steps, n_attempts=2, min_density=0, grid_size=(10, 10), icon_size=256, x_extent=(0., 1.0), y_extent=(0., 1.0)):
    grid_layout = grid(xpts=xs, ypts=ys, grid_size=grid_size, x_extent=x_extent, y_extent=y_extent)
    icons = []
    X = []
    Y = []
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            indices = grid_layout[x, y]
            if len(indices) > min_density:
                average_activation = np.average(activations[indices], axis=0)
                icons.append(average_activation)
                X.append(x)
                Y.append(y)

    icons = np.asarray(icons)
    print ("Icons:", icons.shape)
    
    icon_batch = render_icons(icons, model, layer, n_steps, n_attempts, icon_size, step_size = 0.01)
    print ("Icon batch:", icon_batch.shape)

    canvas = np.ones((icon_size * grid_size[0], icon_size * grid_size[1], 3))
    for i in range(icon_batch.shape[0]):
        icon = icon_batch[i]
        y = int(X[i])
        x = int(Y[i])
        canvas[(grid_size[0] - x - 1) * icon_size:(grid_size[0] - x) * icon_size, (y) * icon_size:(y + 1) * icon_size] = icon

    return canvas