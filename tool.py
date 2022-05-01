import tensorflow as tf 

def generate_model():
    model = tf.keras.Sequential([

        tf.keras.layers.Conv2D(32, filter_size = 3, activation = 'relu'),
        tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2),

        tf.keras.layers.Conv2D(64, filter_size = 3, activation = 'relu'),
        tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2),

        # tf.keras.layers.Conv2D(128, filter_size = 3, activation = 'relu'),
        # tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    return model

model = generate_model()
model.summary()
