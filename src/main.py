import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from lib.AIHelper import deprocess, get_gram_matrix, get_total_loss
from lib.images import image_to_numpy_array
# todo: plotting
from keras.applications import vgg19


style_image_path = "./data/wave.jpg"
style_image = image_to_numpy_array(style_image_path, target_size=(512,512))

# show that thing
#plt.figure(figsize=(15,15))
#plt.imshow(style_image)
#plt.show()

content_image_path = "./data/golden-gate-bridge.jpg"
content_image = image_to_numpy_array(content_image_path, target_size=(512,512))


# AI Model
CONTENT_LAYERS = ["block5_conv2"]
OUTPUT_LAYERS = ["block4_conv1", "block4_conv2", "block3_conv3", "block4_conv4"]

def make_model(include_full=False, input_shape=None):
    if include_full:
        base_model = vgg19.VGG19(include_top=False, weights="imagenet")
        return  base_model
    
    if input_shape:
        base_model = vgg19.VGG19(include_top=False, input_shape=input_shape ,weights="imagenet")
    else:
        base_model = vgg19.VGG19(include_top=False, weights="imagenet")
        
    base_model.trainable = False
    content_layers = CONTENT_LAYERS
    style_layers = OUTPUT_LAYERS
    output_layers = [base_model.get_layer(layer).output for layer in (content_layers+style_layers)]
    return tf.keras.models.Model(base_model.input, output_layers)


content_img_noised = content_image + np.random.randn(*content_image.shape)*10
content_img_noised = content_img_noised.astype("float32")

processed_style = vgg19.preprocess_input(np.expand_dims(style_image, axis=0))
processed_content = vgg19.preprocess_input(np.expand_dims(content_image, axis=0))


VGG_BIASES = vgg19.preprocess_input((np.zeros((3))).astype("float32"))
    
#plt.figure(figsize=(15,15))
#plt.imshow(np.round(deprocess(processed_content, VGG_BIASES)[0])/255)
#plt.show()     

base_model = make_model()
        
content_image_outputs = base_model(processed_content)
style_image_outputs = base_model(processed_style)

content_image_content = content_image_outputs[0]
style_image_content = style_image_outputs[0]

gram_matrix, N = get_gram_matrix(style_image_outputs[2])
#plt.figure(figsize=(15,15))
#plt.imshow(gram_matrix.numpy())
#plt.show()

#tl = get_total_loss(style_image_outputs, content_image_outputs, style_image_outputs, CONTENT_LAYERS)        


# Now training..
base_style_outputs = base_model(processed_style)
base_content_outputs = base_model(processed_content)

processed_content_var = tf.Variable(processed_content+tf.random.normal(processed_content.shape))
optimizer = tf.optimizers.Adam(5, beta_1=.99, epsilon=1e-3)

images = []
losses = []

best_loss = 200_000
min_vals = VGG_BIASES
max_vals = 255+VGG_BIASES

for i in range(200):
    with tf.GradientTape() as tape:
        tape.watch(processed_content_var)
        content_var_outputs = base_model(processed_content_var)
        loss = get_total_loss(content_var_outputs, base_content_outputs, base_style_outputs, CONTENT_LAYERS)
        grad = tape.gradient(loss, processed_content_var)
        
        losses.append(loss)
        optimizer.apply_gradients(zip([grad], [processed_content_var]))
        clipped = tf.clip_by_value(processed_content_var, min_vals, max_vals)
        processed_content_var.assign(clipped)
        
        if i%5 == 0:
            images.append(deprocess(processed_content_var, VGG_BIASES))
            
        if loss < best_loss:
            best_image = processed_content_var
            best_loss = loss 
        
        
deprocesse_best_image = deprocess(best_image, VGG_BIASES)
plt.figure(figsize=(15,15))
plt.imshow(deprocesse_best_image[0]/255)
plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        