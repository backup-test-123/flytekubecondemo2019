
# Default ResNet50-specific hyper-parameters
DEFAULT_IMG_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 16
DEFAULT_CLASS_LABELS = sorted(
    ["clean", "dirty"]
)  # Keras sorts class labels alphabetically
DEFAULT_POSITIVE_LABEL = "dirty"
DEFAULT_PATIENCE = 3  # 100
DEFAULT_EPOCHS = 5 # 1000
DEFAULT_WEIGHTS = "imagenet"

CHECKPOINT_FILE_NAME = "resnet50_best.h5"
FINAL_FILE_NAME = "resnet50_final.h5"
MODEL_CONFIG_FILE_NAME = "model_config.json"