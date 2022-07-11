import numpy as np
from PIL import Image

def image_to_numpy_array(image_location, target_size:tuple=None):
    """
    @param image_location: location of image to load
    @param target_size (tuple): desized size of np array
    @return: numpy array
    """    
    img = Image.open(image_location)
    if target_size:
        return np.array(img.resize(target_size))
    else:
        return np.array(img)
    
    
    
    
if __name__ == "__main__":
    print(type((0,1)))