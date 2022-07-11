import tensorflow as tf
import numpy as np

def deprocess(proc_image, vgg_biases=0):
    deprocessed_img = proc_image-vgg_biases
    deprocessed_img = tf.unstack(deprocessed_img, axis=-1)
    deprocessed_img = tf.stack([deprocessed_img[2], deprocessed_img[1], deprocessed_img[0]], axis=-1)
    return deprocessed_img

def get_content_loss(new_image_content, base_image_content):
    return np.mean(np.square(new_image_content-base_image_content))

def get_gram_matrix(output):
    first_style_layer = output
    A = tf.reshape(first_style_layer, (-1, first_style_layer.shape[-1]))
    n = A.shape[0]
    gram_matrix = tf.matmul(A,A,transpose_a=True)
    n = gram_matrix.shape[0]
    return gram_matrix/tf.cast(n, "float32"), n

def get_style_loss(new_image_style, base_image_style):
    new_style_gram, new_gram_height = get_gram_matrix(new_image_style)
    base_style_gram, base_gram_height = get_gram_matrix(base_image_style)
    assert new_gram_height == base_gram_height
    gram_num_features = new_style_gram.shape[0]
    loss = tf.reduce_sum(tf.square(base_style_gram-new_style_gram)/(4*(new_gram_height**2)*(gram_num_features**2)))
    return loss

def get_total_loss(new_image_output, base_content_image_output, base_style_image_output, content_layers, alpha=.999):
    new_image_styles = new_image_output[len(content_layers)]
    base_image_styles = base_style_image_output[len(content_layers)]
    style_loss = 0
    N = len(new_image_styles)
    for i in range(N):
        style_loss += get_style_loss(new_image_styles[i], base_image_styles[i])
        
    new_image_contents = new_image_output[:len(content_layers)]
    base_image_contents = base_content_image_output[:len(content_layers)]
    content_loss = 0
    N = len(new_image_contents)
    for i in range(N):
        content_loss += get_style_loss(new_image_contents[i], base_image_contents[i])
        
    return (1-alpha)*style_loss+alpha*content_loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    