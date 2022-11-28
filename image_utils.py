import numpy as np
import tensorflow as tf

COLOR_CORRELATION_SVD_SQRT = np.asarray([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]).astype("float32")
MAX_NORM_SVD_SQRT = np.max(np.linalg.norm(COLOR_CORRELATION_SVD_SQRT, axis=0))
COLOR_MEAN = [0.48, 0.46, 0.41]

def deprocess_image(img):
    # compute the normal scores (z scores) and add little noise for uncertainty
    img = ((img - img.mean()) / img.std()) + 1e-5
    # ensure that the variance is 0.15
    img *= 0.15
    # croping the center adn clip the values between 0 and 1
    img = img[25:-25, 25:-25, :]
    img += 0.5
    img = np.clip(img, 0, 1)

    # convert to RGB
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")

    return img

def rfft2d_freqs(height, width):
    freq_y = np.fft.fftfreq(height)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if width % 2 == 1:
        freq_x = np.fft.fftfreq(width)[: width // 2 + 2]
    else:
        freq_x = np.fft.fftfreq(width)[: width // 2 + 1]
    return np.sqrt(freq_x * freq_x + freq_y * freq_y)
    
def fft_image(batch, width, height, channels=3, std=None, decay_power=1):
    # real valued fft
    freqs = rfft2d_freqs(height, width)
    init_val_size = (2, batch, channels) + freqs.shape
    spectrum_real_imag_t = tf.Variable(np.random.normal(
        size=init_val_size, scale=(std or 0.01)).astype(np.float32))

    # Normalize energy
    scale = 1.0 / np.maximum(freqs, 1.0 / max(width, height)) ** decay_power
    # Scale it by the square root
    scale *= np.sqrt(width * height)
    spectrum_t = tf.complex(spectrum_real_imag_t[0], spectrum_real_imag_t[1])
    scaled_spectrum_t = scale * spectrum_t

    # convert the spectrum to spatial domain
    image_t = tf.transpose(tf.signal.irfft2d(scaled_spectrum_t), (0, 2, 3, 1))
    image_t = image_t[:batch, :height, :width, :channels]
    image_t = image_t / 4.0
    return image_t

def _linear_decorrelate_color(image):
    t_flat = tf.reshape(image, [-1, 3])
    color_correlation_normalized = COLOR_CORRELATION_SVD_SQRT / MAX_NORM_SVD_SQRT
    t_flat = tf.matmul(t_flat, color_correlation_normalized.T)
    image = tf.reshape(t_flat, tf.shape(image))
    return image

def to_valid_rgb(image, decorrelate=False, sigmoid=True):
    if decorrelate:
        image = _linear_decorrelate_color(image)
    if decorrelate and not sigmoid:
        image += COLOR_MEAN
    if sigmoid:
        image = tf.nn.sigmoid(image)
    else:
        val = (2 * image - 1)
        image = (val / tf.maximum(1.0, tf.abs(val))) / 2 + 0.5
    return image

def initialize_image_ref(batch, size, channels=3, std=None, fft=True, decorrelate=True,  seed=None):
    height = size
    width = size
    if fft:
        image_f = fft_image(batch, width, height, channels, std=std)

    if channels == 3:
        output = to_valid_rgb(
            image_f[..., :3], decorrelate=decorrelate, sigmoid=True)
    else:
        output = tf.nn.sigmoid(image_f)
    return output


