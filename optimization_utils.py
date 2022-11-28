from typing import Optional
import tensorflow as tf
from dataclasses import dataclass
from objective_utils import LayerActivationObjective

@dataclass  
class OptimizationParameters( ):
    """object for generalizing optimization parameters."""
    iterations: int
    learning_rate: Optional[int]
    optimizer: Optional[object]

def optimize(img, objective, transform_image, optimization_parameters, minimize):
    def compute_loss():
        transformed_image = transform_image(img)
        return objective.loss(transformed_image)

    if not minimize:
        with tf.GradientTape() as tape:
            tape.watch(img)
            # Record operations to compute gradients with respect to img
            loss = compute_loss()
            print ("Loss:", loss)

        # Compute gradients with respect to img using backpropagation.
        grads = tape.gradient(loss, img)

        # Normalize gradients.
        if optimization_parameters.optimizer is None:
            grads = tf.math.l2_normalize(grads)
            # fallback to standard learning for apply gradient ascent
            learning_rate = optimization_parameters.learning_rate or 0.7
            img = img + learning_rate * grads
        else:
            optimization_parameters.optimizer.apply_gradients(
                zip([grads], [img]))
        activation = -loss
    else:
        activation = optimization_parameters.optimizer.minimize(compute_loss, [img])
    return activation, img

def visualize(image, objective, optimization_parameters, transformation=None, threshold=None, minimize=False):
    def transform_image(img):
        if transformation:
            if not callable(transformation):
                raise ValueError("transformation needs to be a function.")
            transformed = transformation(img)

            if transformed.shape[1] != img.shape[1] or transformed.shape[2] != img.shape[2]:
                transformed = tf.image.resize(
                    transformed, [img.shape[1], img.shape[2]])
            return transformed
        return img

    image = tf.Variable(image)
    print("Starting Feature Vis Process")
    for iteration in range(optimization_parameters.iterations):
        activation, image = optimize(image, objective, transform_image, optimization_parameters, minimize)

        print('>>', int(iteration / optimization_parameters.iterations * 100), '%',
              end="\r", flush=True)

        # if isinstance(threshold, list) and (iteration in threshold):
        #     threshold_image = _threshold_figures.add_subplot(
        #         1, len(threshold), threshold.index(iteration) + 1
        #     )
        #     threshold_image.title.set_text(f"Step {iteration}")
        #     threshold_view(image)

    print('>> 100 %')
    return activation, image.numpy()

def visualize_layer(activations, image, model, layer, optimization_parameters, transformation=None, regularization=None, threshold=None, minimize=False):
    objective = LayerActivationObjective(model, layer, activations, regularization)
    return visualize(image, objective, optimization_parameters, transformation, threshold, minimize)

