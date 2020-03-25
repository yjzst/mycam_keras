from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import numpy as np
import keras.backend as K
from keras.applications.mobilenet import preprocess_input,decode_predictions
import cv2
model = VGG16(weights='imagenet')
img_path = r"testimage/5.png"
img = image.load_img(img_path,target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)
preds = model.predict(x)
print(np.argmax(preds[0]))
elephant = model.output[:,669]
last_conv = model.get_layer('block5_conv3')
# [14,14,512]
grads = K.gradients(elephant,last_conv.output)[0]
pool_grads = K.mean(grads,axis=(0,1,2))
iterate = K.function([model.input],[pool_grads,last_conv.output[0]])
pool_grads_value,last_conv_value = iterate([x])
for i in range(512):
    last_conv_value[:,:,i] *= pool_grads_value[i]
print(last_conv_value.shape)
heatmap = np.mean(last_conv_value,axis=-1)
heatmap = np.maximum(heatmap,0)
print(np.max(heatmap))
heatmap /= np.max(heatmap)
print(heatmap.shape)
img2 = cv2.imread(img_path)
heatmap = cv2.resize(heatmap,(img2.shape[1],img2.shape[0]))
heatmap = np.uint8(heatmap * 255)
heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
final_img = heatmap * 0.4 + img2
cv2.imwrite(r"D:\DeepLearning_relate(2)\mycam_karas\testimage\4.jpg",final_img)
