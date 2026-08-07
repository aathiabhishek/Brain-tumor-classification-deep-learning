

# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization,
#                                      GlobalAveragePooling2D, Dense, Flatten, SpatialDropout2D)
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# try:
#     import keras_tuner as kt
# except:
#     os.system("pip install keras-tuner")
#     import keras_tuner as kt

# # Paths and constants
# train_path = "data/Training"
# test_path = "data/Testing"
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 128
# NUM_CLASSES = 4

# # Data Preparation
# train_gen = ImageDataGenerator(rescale=1./255,
#                                rotation_range=15,
#                                zoom_range=0.1,
#                                width_shift_range=0.1,
#                                height_shift_range=0.1,
#                                horizontal_flip=True,
#                                vertical_flip=False,
#                                fill_mode='nearest').flow_from_directory(
#     train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
#     test_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# labels = list(test_gen.class_indices.keys())

# # Evaluation & Plotting
# def evaluate_model(model, name, history):
#     preds = model.predict(test_gen)
#     y_pred = np.argmax(preds, axis=1)
#     y_true = test_gen.classes

#     # Confusion Matrix
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
#     plt.title(f"Confusion Matrix ({name})")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.savefig(f"confusion_matrix_{name}.png")

#     # Accuracy Plot
#     plt.figure(figsize=(10, 5))
#     plt.plot(history.history['accuracy'], label='Train Acc')
#     plt.plot(history.history['val_accuracy'], label='Val Acc')
#     plt.title(f"Accuracy Plot ({name})")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.savefig(f"accuracy_plot_{name}.png")

#     # Report
#     print(f"\nClassification Report ({name}):")
#     print(classification_report(y_true, y_pred, target_names=labels))

#     model.save(f"brain_tumor_model_{name}.h5")
#     print(f"✅ Model saved as brain_tumor_model_{name}.h5")

# # EarlyStopping
# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# # CNN Model
# def build_cnn_model():
#     model = Sequential([
#         Input(shape=(224, 224, 3)),
#         Conv2D(32, (3,3), activation='relu', padding='same'),
#         BatchNormalization(),
#         Conv2D(32, (3,3), activation='relu', padding='same'),
#         BatchNormalization(),
#         MaxPooling2D(2,2),
#         SpatialDropout2D(0.25),

#         Conv2D(64, (3,3), activation='relu', padding='same'),
#         BatchNormalization(),
#         Conv2D(64, (3,3), activation='relu', padding='same'),
#         BatchNormalization(),
#         MaxPooling2D(2,2),
#         SpatialDropout2D(0.3),

#         Conv2D(128, (3,3), activation='relu', padding='same'),
#         BatchNormalization(),
#         Conv2D(128, (3,3), activation='relu', padding='same'),
#         BatchNormalization(),
#         MaxPooling2D(2,2),
#         SpatialDropout2D(0.35),

#         Conv2D(256, (3,3), activation='relu', padding='same'),
#         BatchNormalization(),
#         Conv2D(256, (3,3), activation='relu', padding='same'),
#         BatchNormalization(),
#         MaxPooling2D(2,2),
#         SpatialDropout2D(0.4),

#         Conv2D(512, (3,3), activation='relu', padding='same'),
#         BatchNormalization(),
#         Conv2D(512, (3,3), activation='relu', padding='same'),
#         BatchNormalization(),
#         MaxPooling2D(2,2),
#         SpatialDropout2D(0.45),

#         GlobalAveragePooling2D(),
#         Dense(512, activation='relu'),
#         Dropout(0.5),
#         Dense(256, activation='relu'),
#         Dropout(0.4),
#         Dense(NUM_CLASSES, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # Hyperparameter Tuning Model
# def build_tuned_model(hp):
#     model = Sequential()
#     model.add(Input(shape=(224, 224, 3)))
#     for i in range(hp.Int("conv_blocks", 2, 4)):
#         filters = hp.Int(f"filters_{i}", 32, 128, step=32)
#         model.add(Conv2D(filters, 3, activation='relu', padding='same'))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D())
#         model.add(Dropout(hp.Float(f"dropout_{i}", 0.2, 0.5, step=0.1)))
#     model.add(GlobalAveragePooling2D())
#     model.add(Dense(hp.Int("dense_units", 64, 256, step=64), activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(NUM_CLASSES, activation='softmax'))
#     model.compile(optimizer=tf.keras.optimizers.Adam(
#         hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
#         loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # Autoencoder Classifier
# def build_autoencoder_classifier():
#     input_img = Input(shape=(224, 224, 3))
#     x = Conv2D(32, 3, activation='relu', padding='same')(input_img)
#     x = MaxPooling2D(2, padding='same')(x)
#     x = Conv2D(64, 3, activation='relu', padding='same')(x)
#     encoded = MaxPooling2D(2, padding='same')(x)
#     flat = Flatten()(encoded)
#     dense = Dense(128, activation='relu')(flat)
#     output = Dense(NUM_CLASSES, activation='softmax')(dense)
#     model = Model(input_img, output)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # Pretrained Models
# def build_vgg16_model():
#     base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#     base.trainable = False
#     x = GlobalAveragePooling2D()(base.output)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     output = Dense(NUM_CLASSES, activation='softmax')(x)
#     model = Model(inputs=base.input, outputs=output)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def build_resnet50_model():
#     base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#     base.trainable = False
#     x = GlobalAveragePooling2D()(base.output)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     output = Dense(NUM_CLASSES, activation='softmax')(x)
#     model = Model(inputs=base.input, outputs=output)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def build_mobilenet_model():
#     base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#     base.trainable = False
#     x = GlobalAveragePooling2D()(base.output)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     output = Dense(NUM_CLASSES, activation='softmax')(x)
#     model = Model(inputs=base.input, outputs=output)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# Training Loop for All Models
# print("🔧 Training Base CNN...")
# cnn_model = build_cnn_model()
# cnn_model_history = cnn_model.fit(train_gen, validation_data=test_gen, epochs=20, callbacks=[early_stop])
# evaluate_model(cnn_model, "cnn", cnn_model_history)

# print("\n Tuning CNN Hyperparameters...")
# tuner = kt.RandomSearch(build_tuned_model, objective='val_accuracy', max_trials=2, directory='kt_logs', project_name='cnn_tuning')
# tuner.search(train_gen, validation_data=test_gen, epochs=10)
# tuned_model = tuner.get_best_models(1)[0]
# tuned_model_history = tuned_model.fit(train_gen, validation_data=test_gen, epochs=20, callbacks=[early_stop])
# evaluate_model(tuned_model, "tuned", tuned_model_history)

# print("\n Training Autoencoder Classifier...")
# auto_model = build_autoencoder_classifier()
# auto_model_history = auto_model.fit(train_gen, validation_data=test_gen, epochs=20, callbacks=[early_stop])
# evaluate_model(auto_model, "autoencoder", auto_model_history)

# print("\n Training VGG16...")
# vgg_model = build_vgg16_model()
# vgg_model_history = vgg_model.fit(train_gen, validation_data=test_gen, epochs=10, callbacks=[early_stop])
# evaluate_model(vgg_model, "vgg16", vgg_model_history)

# print("\n Training ResNet50...")
# resnet_model = build_resnet50_model()
# resnet_model_history = resnet_model.fit(train_gen, validation_data=test_gen, epochs=10, callbacks=[early_stop])
# evaluate_model(resnet_model, "resnet50", resnet_model_history)

# print("\n Training MobileNetV2...")
# mobilenet_model = build_mobilenet_model()
# mobilenet_model_history = mobilenet_model.fit(train_gen, validation_data=test_gen, epochs=20, callbacks=[early_stop])
# evaluate_model(mobilenet_model, "mobilenetv2", mobilenet_model_history)



# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Flatten,
    SpatialDropout2D
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import (
    VGG16,
    ResNet50,
    MobileNetV2
)

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Dataset directories
TRAIN_PATH = "data/Training"
TEST_PATH = "data/Testing"

# Input image size
IMG_SIZE = (224, 224)

# Number of images processed at once
BATCH_SIZE = 32

# Number of tumor classes
NUM_CLASSES = 4

# Maximum number of training epochs
EPOCHS = 15

# Random seed
SEED = 42


# ============================================================
# 3. OUTPUT DIRECTORIES
# ============================================================

# All trained models
MODEL_DIR = "models"

# All generated results
RESULTS_DIR = "results"

# Confusion matrices
CONFUSION_DIR = os.path.join(
    RESULTS_DIR,
    "confusion_matrices"
)

# Accuracy plots
ACCURACY_DIR = os.path.join(
    RESULTS_DIR,
    "accuracy_plots"
)

# Loss plots
LOSS_DIR = os.path.join(
    RESULTS_DIR,
    "loss_plots"
)

# Training history files
HISTORY_DIR = os.path.join(
    RESULTS_DIR,
    "training_history"
)


# Create directories if they do not exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)
os.makedirs(ACCURACY_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


# ============================================================
# 4. CHECK DATASET DIRECTORIES
# ============================================================

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_PATH}"
    )

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"Testing directory not found: {TEST_PATH}"
    )


print("\nDataset directories found successfully.")


# ============================================================
# 5. DATA PREPARATION
# ============================================================

# ------------------------------------------------------------
# Training augmentation
# ------------------------------------------------------------
#
# Augmentation is applied ONLY to training data.
#
# This helps the model learn variations of MRI images.
# ------------------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,

    rotation_range=15,
    zoom_range=0.10,

    width_shift_range=0.10,
    height_shift_range=0.10,

    horizontal_flip=True,

    vertical_flip=False,

    fill_mode="nearest",

    # 20% of training data is used for validation
    validation_split=0.20
)


# ------------------------------------------------------------
# Training generator
# ------------------------------------------------------------

train_gen = train_datagen.flow_from_directory(
    TRAIN_PATH,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True,

    seed=SEED
)


# ------------------------------------------------------------
# Validation generator
# ------------------------------------------------------------
#
# IMPORTANT:
#
# Validation images should NOT receive augmentation.
# They are only rescaled.
# ------------------------------------------------------------

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.20
)


val_gen = validation_datagen.flow_from_directory(
    TRAIN_PATH,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False,

    seed=SEED
)


# ------------------------------------------------------------
# Testing generator
# ------------------------------------------------------------
#
# Testing data is never augmented.
# ------------------------------------------------------------

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


test_gen = test_datagen.flow_from_directory(
    TEST_PATH,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False
)


# ============================================================
# 6. CLASS MAPPING
# ============================================================

print("\n" + "=" * 60)
print("CLASS MAPPING")
print("=" * 60)

print("\nTraining classes:")
print(train_gen.class_indices)

print("\nTesting classes:")
print(test_gen.class_indices)


# Make sure training and testing have the same classes
if train_gen.class_indices != test_gen.class_indices:
    raise ValueError(
        "Training and testing class mappings are different."
    )


# Save the class mapping
class_indices_path = os.path.join(
    MODEL_DIR,
    "class_indices.json"
)

with open(class_indices_path, "w") as f:
    json.dump(
        train_gen.class_indices,
        f,
        indent=4
    )


# Create labels in the correct numerical order
labels = [
    name
    for name, index
    in sorted(
        train_gen.class_indices.items(),
        key=lambda x: x[1]
    )
]


print("\nClass order used by the model:")
for index, label in enumerate(labels):
    print(f"{index} -> {label}")


print(
    f"\nClass mapping saved to: {class_indices_path}"
)


# ============================================================
# 7. EARLY STOPPING
# ============================================================

# Stop training when validation loss stops improving.
#
# restore_best_weights=True means the best model weights
# are restored after training finishes.

early_stop = EarlyStopping(
    monitor="val_loss",

    patience=3,

    restore_best_weights=True,

    verbose=1
)


# ============================================================
# 8. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, name, history):

    print("\n" + "=" * 60)
    print(f"EVALUATING: {name.upper()}")
    print("=" * 60)


    # --------------------------------------------------------
    # Reset test generator
    # --------------------------------------------------------

    test_gen.reset()


    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    predictions = model.predict(
        test_gen,
        verbose=1
    )


    # Convert probabilities into class indexes
    y_pred = np.argmax(
        predictions,
        axis=1
    )


    # Actual labels
    y_true = test_gen.classes


    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print(
        f"\nClassification Report - {name}"
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        digits=4
    )

    print(report)


    # Save classification report
    report_path = os.path.join(
        RESULTS_DIR,
        f"classification_report_{name}.txt"
    )

    with open(report_path, "w") as f:
        f.write(
            f"Classification Report - {name}\n\n"
        )

        f.write(report)


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )


    plt.figure(
        figsize=(8, 6)
    )


    sns.heatmap(
        cm,

        annot=True,

        fmt="d",

        xticklabels=labels,

        yticklabels=labels,

        cmap="Blues"
    )


    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )


    plt.tight_layout()


    confusion_path = os.path.join(
        CONFUSION_DIR,
        f"confusion_matrix_{name}.png"
    )


    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Confusion matrix saved: {confusion_path}"
    )


    # --------------------------------------------------------
    # Accuracy Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )


    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )


    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )


    plt.title(
        f"Accuracy Curve - {name}"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()


    accuracy_path = os.path.join(
        ACCURACY_DIR,
        f"accuracy_plot_{name}.png"
    )


    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Accuracy plot saved: {accuracy_path}"
    )


    # --------------------------------------------------------
    # Loss Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )


    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )


    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )


    plt.title(
        f"Loss Curve - {name}"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()


    loss_path = os.path.join(
        LOSS_DIR,
        f"loss_plot_{name}.png"
    )


    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Loss plot saved: {loss_path}"
    )


    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"brain_tumor_model_{name}.h5"
    )


    model.save(
        model_path
    )


    print(
        f"\nModel saved successfully:"
    )

    print(
        model_path
    )


    # --------------------------------------------------------
    # Save Training History
    # --------------------------------------------------------

    history_path = os.path.join(
        HISTORY_DIR,
        f"history_{name}.json"
    )


    # Convert NumPy values to standard Python floats
    history_data = {
        key: [
            float(value)
            for value in values
        ]

        for key, values
        in history.history.items()
    }


    with open(
        history_path,
        "w"
    ) as f:

        json.dump(
            history_data,
            f,
            indent=4
        )


    print(
        f"Training history saved: {history_path}"
    )


# ============================================================
# 9. CUSTOM CNN
# ============================================================

def build_cnn_model():

    model = Sequential([

        Input(
            shape=(224, 224, 3)
        ),


        # ---------------- BLOCK 1 ----------------
        # Learns basic edges and textures.

        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(
            2,
            2
        ),

        SpatialDropout2D(
            0.25
        ),


        # ---------------- BLOCK 2 ----------------

        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(
            2,
            2
        ),

        SpatialDropout2D(
            0.30
        ),


        # ---------------- BLOCK 3 ----------------

        Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(
            2,
            2
        ),

        SpatialDropout2D(
            0.35
        ),


        # ---------------- BLOCK 4 ----------------

        Conv2D(
            256,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        Conv2D(
            256,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(
            2,
            2
        ),

        SpatialDropout2D(
            0.40
        ),


        # ---------------- BLOCK 5 ----------------

        Conv2D(
            512,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        Conv2D(
            512,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(
            2,
            2
        ),

        SpatialDropout2D(
            0.45
        ),


        # Convert feature maps into a vector
        GlobalAveragePooling2D(),


        # Classification head
        Dense(
            512,
            activation="relu"
        ),

        Dropout(
            0.5
        ),

        Dense(
            256,
            activation="relu"
        ),

        Dropout(
            0.4
        ),


        # Four tumor classes
        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])


    model.compile(
        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=["accuracy"]
    )


    return model


# ============================================================
# 10. TUNED CNN
# ============================================================

def build_tuned_model():

    model = Sequential([

        Input(
            shape=(224, 224, 3)
        ),


        # ---------------- BLOCK 1 ----------------

        Conv2D(
            64,
            3,
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(),

        Dropout(
            0.25
        ),


        # ---------------- BLOCK 2 ----------------

        Conv2D(
            128,
            3,
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(),

        Dropout(
            0.30
        ),


        # ---------------- BLOCK 3 ----------------

        Conv2D(
            128,
            3,
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(),

        Dropout(
            0.35
        ),


        GlobalAveragePooling2D(),


        Dense(
            256,
            activation="relu"
        ),

        Dropout(
            0.5
        ),


        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])


    # Lower learning rate for stable training
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001
    )


    model.compile(
        optimizer=optimizer,

        loss="categorical_crossentropy",

        metrics=["accuracy"]
    )


    return model


# ============================================================
# 11. AUTOENCODER-BASED CLASSIFIER
# ============================================================

def build_autoencoder_classifier():

    input_img = Input(
        shape=(224, 224, 3)
    )


    # Encoder
    # Compresses image information into features.

    x = Conv2D(
        32,
        3,
        activation="relu",
        padding="same"
    )(input_img)


    x = MaxPooling2D(
        2,
        padding="same"
    )(x)


    x = Conv2D(
        64,
        3,
        activation="relu",
        padding="same"
    )(x)


    encoded = MaxPooling2D(
        2,
        padding="same"
    )(x)


    # Convert feature maps to a vector
    flat = Flatten()(
        encoded
    )


    dense = Dense(
        128,
        activation="relu"
    )(flat)


    # Classification layer
    output = Dense(
        NUM_CLASSES,
        activation="softmax"
    )(dense)


    model = Model(
        input_img,
        output
    )


    model.compile(
        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=["accuracy"]
    )


    return model


# ============================================================
# 12. VGG16 TRANSFER LEARNING
# ============================================================

def build_vgg16_model():

    # VGG16 pretrained on ImageNet
    base = VGG16(
        weights="imagenet",

        include_top=False,

        input_shape=(224, 224, 3)
    )


    # Freeze pretrained layers
    base.trainable = False


    x = GlobalAveragePooling2D()(
        base.output
    )


    x = Dense(
        256,
        activation="relu"
    )(x)


    x = Dropout(
        0.5
    )(x)


    output = Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)


    model = Model(
        inputs=base.input,

        outputs=output
    )


    model.compile(
        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=["accuracy"]
    )


    return model


# ============================================================
# 13. RESNET50 TRANSFER LEARNING
# ============================================================

def build_resnet50_model():

    # ResNet50 pretrained on ImageNet
    base = ResNet50(
        weights="imagenet",

        include_top=False,

        input_shape=(224, 224, 3)
    )


    # Freeze pretrained layers
    base.trainable = False


    x = GlobalAveragePooling2D()(
        base.output
    )


    x = Dense(
        256,
        activation="relu"
    )(x)


    x = Dropout(
        0.5
    )(x)


    output = Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)


    model = Model(
        inputs=base.input,

        outputs=output
    )


    model.compile(
        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=["accuracy"]
    )


    return model


# ============================================================
# 14. MOBILENETV2 TRANSFER LEARNING
# ============================================================

def build_mobilenet_model():

    # MobileNetV2 is lightweight and efficient.
    base = MobileNetV2(
        weights="imagenet",

        include_top=False,

        input_shape=(224, 224, 3)
    )


    # Freeze pretrained layers
    base.trainable = False


    x = GlobalAveragePooling2D()(
        base.output
    )


    x = Dense(
        128,
        activation="relu"
    )(x)


    x = Dropout(
        0.3
    )(x)


    output = Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)


    model = Model(
        inputs=base.input,

        outputs=output
    )


    model.compile(
        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=["accuracy"]
    )


    return model


# ============================================================
# 15. TRAIN CUSTOM CNN
# ============================================================

print("\n" + "=" * 60)
print("TRAINING CUSTOM CNN")
print("=" * 60)


cnn_model = build_cnn_model()


cnn_history = cnn_model.fit(
    train_gen,

    validation_data=val_gen,

    epochs=EPOCHS,

    callbacks=[early_stop]
)


evaluate_model(
    cnn_model,

    "cnn",

    cnn_history
)


# ============================================================
# 16. TRAIN TUNED CNN
# ============================================================

print("\n" + "=" * 60)
print("TRAINING TUNED CNN")
print("=" * 60)


tuned_model = build_tuned_model()


tuned_history = tuned_model.fit(
    train_gen,

    validation_data=val_gen,

    epochs=EPOCHS,

    callbacks=[early_stop]
)


evaluate_model(
    tuned_model,

    "tuned",

    tuned_history
)


# ============================================================
# 17. TRAIN AUTOENCODER CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("TRAINING AUTOENCODER CLASSIFIER")
print("=" * 60)


auto_model = build_autoencoder_classifier()


auto_history = auto_model.fit(
    train_gen,

    validation_data=val_gen,

    epochs=EPOCHS,

    callbacks=[early_stop]
)


evaluate_model(
    auto_model,

    "autoencoder",

    auto_history
)


# ============================================================
# 18. TRAIN VGG16
# ============================================================

print("\n" + "=" * 60)
print("TRAINING VGG16")
print("=" * 60)


vgg_model = build_vgg16_model()


vgg_history = vgg_model.fit(
    train_gen,

    validation_data=val_gen,

    epochs=EPOCHS,

    callbacks=[early_stop]
)


evaluate_model(
    vgg_model,

    "vgg16",

    vgg_history
)


# ============================================================
# 19. TRAIN RESNET50
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RESNET50")
print("=" * 60)


resnet_model = build_resnet50_model()


resnet_history = resnet_model.fit(
    train_gen,

    validation_data=val_gen,

    epochs=EPOCHS,

    callbacks=[early_stop]
)


evaluate_model(
    resnet_model,

    "resnet50",

    resnet_history
)


# ============================================================
# 20. TRAIN MOBILENETV2
# ============================================================

print("\n" + "=" * 60)
print("TRAINING MOBILENETV2")
print("=" * 60)


mobilenet_model = build_mobilenet_model()


mobilenet_history = mobilenet_model.fit(
    train_gen,

    validation_data=val_gen,

    epochs=EPOCHS,

    callbacks=[early_stop]
)


evaluate_model(
    mobilenet_model,

    "mobilenetv2",

    mobilenet_history
)


# ============================================================
# 21. TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("ALL SIX MODELS TRAINED SUCCESSFULLY")
print("=" * 60)


print("\nModels:")
print(
    "  models/brain_tumor_model_cnn.h5"
)

print(
    "  models/brain_tumor_model_tuned.h5"
)

print(
    "  models/brain_tumor_model_autoencoder.h5"
)

print(
    "  models/brain_tumor_model_vgg16.h5"
)

print(
    "  models/brain_tumor_model_resnet50.h5"
)

print(
    "  models/brain_tumor_model_mobilenetv2.h5"
)


print("\nResults:")
print(
    "  results/confusion_matrices/"
)

print(
    "  results/accuracy_plots/"
)

print(
    "  results/loss_plots/"
)

print(
    "  results/training_history/"
)


print("\nClass mapping:")
print(
    "  models/class_indices.json"
)


print("\nTraining pipeline completed successfully.")
