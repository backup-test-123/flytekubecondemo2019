import numpy as np
from keras.models import load_model
from keras.preprocessing import image


def predict(img_path, model, img_size):
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)

    return preds


def predict_with_resnet50_model(model_path, evaluation_dataset, batch_size, img_size):

    print("Loading model: " + model_path)
    model = load_model(model_path)
    print("Loaded")

    evaluation_datagen = image.ImageDataGenerator()

    print(f"Evaluation dataset: {evaluation_dataset}")
    evaluation_batches = evaluation_datagen.flow_from_directory(
        evaluation_dataset,
        target_size=img_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
    )
    ground_truths = evaluation_batches.classes

    num_test_steps = len(evaluation_batches)
    predictions = model.predict_generator(evaluation_batches, steps=num_test_steps)

    return ground_truths, predictions.tolist()
