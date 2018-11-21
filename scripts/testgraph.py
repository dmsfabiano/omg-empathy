import networks as net
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import metrics, losses

print(tf.__version__)
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0
ntrain=train_images[10000:30000]
ntlabels=train_labels[10000:30000]

# Can modify anything BELOW this comment
nvalidation = ntrain[0:4000]
nvalidation_labels = ntlabels[0:4000]
ntrain = ntrain[4000:]
ntlabels = ntlabels[4000:]
ntrain = ntrain.reshape(16000,28,28,1)
nvalidation = nvalidation.reshape(4000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['acc', 'mse'])

history = model.fit(x=ntrain, y=ntlabels, validation_data=(nvalidation, nvalidation_labels), batch_size=256, epochs=5)

net.graphTrainingData(history, imagePath='train_graph.png', metrics=['acc', 'mean_squared_error'], show = True)
