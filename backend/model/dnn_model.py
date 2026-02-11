import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_dim):
    """
    Creates a Deep Neural Network model for illness risk prediction.
    """
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=(input_dim,)),

        # Hidden Layer 1
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden Layer 2
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Hidden Layer 3
        layers.Dense(16, activation='relu'),

        # Output Layer (Binary Classification: Low Risk vs High Risk)
        # Using sigmoid activation for probability output (0 to 1)
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
