import tensorflow as tf

def direction_neuron_cossim_S(layer_name, vec, batch=None, x=None, y=None, cossim_pow=1, S=None):
    def inner(T):
        layer = T(layer_name)
        shape = tf.shape(layer)
        x_ = shape[1] // 2 if x is None else x
        y_ = shape[2] // 2 if y is None else y
        if batch is None:
          raise RuntimeError("requires batch")

        acts = layer[batch, x_, y_]
        vec_ = vec
        if S is not None: vec_ = tf.matmul(vec_[None], S)[0]
        mag = tf.sqrt(tf.reduce_sum(acts**2))
        dot = tf.reduce_mean(acts * vec_)
        cossim = dot/(1e-4 + mag)
        cossim = tf.maximum(0.1, cossim)
        return dot * cossim ** cossim_pow
    return inner
    
class direction_neuron_cossim_S:
    def __init__(self, model, layer, target_activation, regularization=None):
        self.model = get_feature_extractor(model, layer)
        self.regularization = regularization
        self.target_activation = target_activation

    def loss(self, input_image):
        activation = self.model(input_image)
        activation_score = -tf.reduce_mean((self.target_activation - activation)**2)
        if self.regularization:
            if not callable(self.regularization):
                raise ValueError("The regularizations need to be a function.")
            activation_score = self.regularization(
                activation, activation_score)
        return -activation_score

class LayerActivationObjective:
    def __init__(self, model, layer, target_activation, regularization=None):
        self.model = get_feature_extractor(model, layer)
        self.regularization = regularization
        self.target_activation = target_activation

    def loss(self, input_image):
        activation = self.model(input_image)
        activation_score = -tf.reduce_mean((self.target_activation - activation)**2)
        if self.regularization:
            if not callable(self.regularization):
                raise ValueError("The regularizations need to be a function.")
            activation_score = self.regularization(
                activation, activation_score)
        return -activation_score

    def __repr__(self) -> str:
        return f"ActivationObjective({self.model}, {self.regularization}, {self.target_activation})"

def get_feature_extractor(model, layer_name):
    layer = model.get_layer(name=layer_name)
    return tf.keras.Model(inputs=model.inputs, outputs=layer.output)