import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from model_evaluation import evaluate_model

#Dummy CNN Model
class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
num_classes = len(class_names)

#Fake Images
#Get from preprocess test data image
X_test = np.random.rand(100, 224, 224, 3).astype(np.float32)

# Create 0-6 class labels for each images ID, then one-hot encoded
y_test = np.random.randint(0, num_classes, size=(100,))
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Split for fake training to get "history"
X_train = X_test[:80]
y_train = y_test_cat[:80]
X_val = X_test[80:]
y_val = y_test_cat[80:]

# Simple EfficientNet Model
def build_efficientnetb2_model(input_shape=(224, 224, 3), num_classes=7):
    base_model = EfficientNetB2(weights=None, include_top=False, input_tensor=Input(shape=input_shape))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=outputs)

# Argument: model
model = build_efficientnetb2_model(input_shape=(224, 224, 3), num_classes=num_classes)

# Compile model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Dummy training
#Argument: histroy
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=3,
                    batch_size=16)

# Run evaluation
evaluate_model(model, history, X_test, y_test_cat, class_names)
