import os

import keras
import ujson
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam

from flytekit.sdk.tasks import python_task, inputs, outputs
from flytekit.common import utils as flytekit_utils
from flytekit.sdk.tasks import inputs, outputs
from flytekit.sdk.types import Types

from models.classifier.resnet50.constants import CHECKPOINT_FILE_NAME
from models.classifier.resnet50.constants import FINAL_FILE_NAME
from models.classifier.resnet50.constants import DEFAULT_IMG_SIZE
from models.classifier.resnet50.constants import DEFAULT_BATCH_SIZE
from models.classifier.resnet50.constants import DEFAULT_CLASS_LABELS
from models.classifier.resnet50.constants import DEFAULT_POSITIVE_LABEL
from models.classifier.resnet50.constants import DEFAULT_PATIENCE
from models.classifier.resnet50.constants import DEFAULT_EPOCHS
from models.classifier.resnet50.constants import DEFAULT_WEIGHTS

from os.path import join, isfile, basename
from os import listdir


def print_dir(directory, logger):
    for r, d, files in os.walk(directory):
        logger.info(r, d, files)


def collect_blobs(folder_path):
    onlyfiles = [
        join(folder_path, f)
        for f in sorted(listdir(folder_path))
        if isfile(join(folder_path, f))
    ]
    my_blobs = []
    file_names = []
    for local_filepath in onlyfiles:

        my_blob = Types.Blob()
        with my_blob as fileobj:
            with open(local_filepath, mode="rb") as file:  # b is important -> binary
                fileobj.write(file.read())
        my_blobs.append(my_blob)
        file_names.append(basename(local_filepath))
    return my_blobs, file_names


# Training function
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

    # Creating a data generator for training data
    gen = keras.preprocessing.image.ImageDataGenerator()

    # Creating a data generator and configuring online data augmentation for validation data
    val_gen = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True, vertical_flip=True
    )

    # Organizing the training images into batches
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

    # Organizing the validation images into batches
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

    # Picking the predefined ResNet50 as our model, and initialize it with a weight file
    model = keras.applications.resnet50.ResNet50(weights=weights)

    # Change resnet from a binary classifier to a multi-class classifier by removing the last later
    classes = list(iter(batches.class_indices))
    model.layers.pop()

    # Since we don't have much training data, we want to leverage the feature learned from a larger dataset, in this,
    # case, imagenet. So we fine-tune based on a pre-trained weight by freezing the weights except for the last layer
    for layer in model.layers:
        layer.trainable = False

    # Attaching a fully-connected layer with softmax activation as the last layer to support multi-class classification
    last = model.layers[-1].output
    x = Dense(len(classes), activation="softmax")(last)

    finetuned_model = Model(inputs=model.input, outputs=x)

    # Compile the model with an optimizer, a loss function, and a list of metrics of choice
    finetuned_model.compile(
        optimizer=Adam(lr=0.00001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    # Setting early stopping thresholds to reduce training time
    early_stopping = EarlyStopping(patience=patience)

    # Checkpoint the current best model
    checkpointer = ModelCheckpoint(
        output_model_folder + "/" + CHECKPOINT_FILE_NAME, verbose=1, save_best_only=True
    )

    # Train it
    finetuned_model.fit_generator(
        batches,
        steps_per_epoch=num_train_steps,
        epochs=epochs,
        callbacks=[early_stopping, checkpointer],
        validation_data=val_batches,
        validation_steps=num_valid_steps,
    )
    finetuned_model.save(output_model_folder + "/" + FINAL_FILE_NAME)


def download_data(base_dir, mpblobs):
    for label, mpblob in mpblobs:
        dir = os.path.join(base_dir, label)
        mpblob.download(local_path=dir)


@inputs(
    training_clean_mpblobs=[Types.MultiPartBlob],
    training_dirty_mpblobs=[Types.MultiPartBlob],
    validation_clean_mpblobs=[Types.MultiPartBlob],
    validation_dirty_mpblobs=[Types.MultiPartBlob],
)
@outputs(
    model_blobs=[Types.Blob],
    model_files_names=[Types.String],
)
@python_task(cache=True, cache_version="1", gpu_request="1", memory_request="64Gi")
def train_on_datasets(
        wf_params,
        training_clean_mpblob,
        validation_clean_mpblob,
        training_dirty_mpblob,
        validation_dirty_mpblob,
        model_blobs,
        model_files_names,
):

    with flytekit_utils.AutoDeletingTempDir("output_models") as output_models_dir:
        with flytekit_utils.AutoDeletingTempDir("training") as training_dir:
            with flytekit_utils.AutoDeletingTempDir("validation") as validation_dir:
                download_data(training_dir.name, {"clean": training_clean_mpblob, "dirty": training_dirty_mpblob})
                download_data(validation_dir.name, {"clean": validation_clean_mpblob, "dirty": validation_dirty_mpblob})

                train_resnet50_model(
                    train_directory=training_dir.name,
                    validation_directory=validation_dir.name,
                    output_model_folder=output_models_dir.name,
                    logger=wf_params.logging,
                    patience=DEFAULT_PATIENCE,
                    size=DEFAULT_IMG_SIZE,
                    batch_size=DEFAULT_BATCH_SIZE,
                    epochs=DEFAULT_EPOCHS,
                    weights=DEFAULT_WEIGHTS,
                )
                # save results to Workflow output
                blobs, files_names_list = collect_blobs(output_models_dir.name)
                model_blobs.set(blobs)
                model_files_names.set(files_names_list)

    """
    # write results to storage path also
    for file in files_names_list:
        location = model_output_path + file
        out_blob = Types.Blob.create_at_known_location(location)

        with out_blob as out_writer:
            with open(output_folder + "/" + file, mode="rb") as in_reader:
                out_writer.write(in_reader.read())

    # keep the model_config with the trained model
    location = model_output_path + MODEL_CONFIG_FILE_NAME
    out_blob = Types.Blob.create_at_known_location(location)
    with out_blob as out_writer:
        out_writer.write((model_config_string).encode("utf-8"))

    # write metadata to track what execution this was done by
    location = model_output_path + MODEL_GENERATED_BY_FILE_NAME
    out_blob = Types.Blob.create_at_known_location(location)
    with out_blob as out_writer:
        out_writer.write((f"workflow_id: {wf_params.execution_id}").encode("utf-8"))
    """