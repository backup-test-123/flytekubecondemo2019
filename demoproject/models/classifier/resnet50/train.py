import os

import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam

from app.constants.resnet50_constants import CHECKPOINT_FILE_NAME
from app.constants.resnet50_constants import FINAL_FILE_NAME


def print_dir(directory, logger):
    for r, d, files in os.walk(directory):
        logger.info(r, d, files)


def train_resnet50_model(
    train_directory,
    validation_directory,
    output_model_folder,
    logger,
    patience,
    epochs,
    batch_size,
    size,
    weights,
):
    logger.info(
        f"Train Resnet 50 called with Train: {train_directory}, Validation: {validation_directory}"
    )
    print_dir(train_directory, logger)
    print_dir(validation_directory, logger)

    gen = keras.preprocessing.image.ImageDataGenerator()
    val_gen = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True, vertical_flip=True
    )

    batches = gen.flow_from_directory(
        train_directory,
        target_size=size,
        class_mode="categorical",
        shuffle=True,
        batch_size=batch_size,
    )
    num_train_steps = len(batches)
    if not num_train_steps:
        raise Exception("No training batches")
    logger.info("num_train_steps = %s" % num_train_steps)

    val_batches = val_gen.flow_from_directory(
        validation_directory,
        target_size=size,
        class_mode="categorical",
        shuffle=True,
        batch_size=batch_size,
    )
    num_valid_steps = len(val_batches)
    if not num_valid_steps:
        raise Exception("No validation batches.")
    logger.info("num_valid_steps = %s" % num_valid_steps)

    model = keras.applications.resnet50.ResNet50(weights=weights)

    classes = list(iter(batches.class_indices))
    model.layers.pop()

    for layer in model.layers:
        layer.trainable = False

    last = model.layers[-1].output
    x = Dense(len(classes), activation="softmax")(last)

    finetuned_model = Model(model.input, x)
    finetuned_model.compile(
        optimizer=Adam(lr=0.00001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=patience)
    checkpointer = ModelCheckpoint(
        output_model_folder + "/" + CHECKPOINT_FILE_NAME, verbose=1, save_best_only=True
    )

    finetuned_model.fit_generator(
        batches,
        steps_per_epoch=num_train_steps,
        epochs=epochs,
        callbacks=[early_stopping, checkpointer],
        validation_data=val_batches,
        validation_steps=num_valid_steps,
    )
    finetuned_model.save(output_model_folder + "/" + FINAL_FILE_NAME)
