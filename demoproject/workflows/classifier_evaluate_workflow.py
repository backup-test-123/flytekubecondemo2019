from pathlib import Path

import ujson
from flytekit.common import utils as flytekit_utils
from flytekit.sdk.tasks import python_task, inputs, outputs
from flytekit.sdk.types import Types
from flytekit.sdk.workflow import workflow_class, Output, Input
from models.classifier.resnet50.constants import DEFAULT_BATCH_SIZE, DEFAULT_IMG_SIZE
from models.classifier.resnet50.constants import DEFAULT_CLASS_LABELS, DEFAULT_POSITIVE_LABEL
from models.classifier.resnet50.evaluate_resnet50 import predict_with_resnet50_model
from models.classifier.resnet50.train_resnet50 import download_data
from utils.flyte_utils.collect_blobs import collect_blobs
from utils.flyte_utils.fetch_executions import fetch_workflow_execution
from utils.metric_utils.metric_utils import calculate_roc_curve, calculate_precision_recall_curve, \
    calculate_cutoff_youdens_j, export_results
from workflows.classifier_train_workflow import rearrange_data, DEFAULT_VALIDATION_DATA_RATIO

DEFAULT_PROJECT_NAME = "flytekubecondemo2019"
DATAPREP_WORKFLOW_NAME = "workflows.data_preparation_workflow.DataPreparationWorkflow"
CLASSIFIER_TRAIN_WORKFLOW_NAME = "workflows.classifier_train_workflow.ClassifierTrainWorkflow"
DEFAULT_DOMAIN = "development"
DEFAULT_DATAPREP_WF_EXECUTION_ID = "ff25dd48a39934dc5b96"  # staging
DEFAULT_CLASSIFIER_TRAIN_WF_EXECUTION_ID = "f65c48801eba846cf95f"  # staging
DEFAULT_EVALUATION_CONFIG_FILE = "models/classifier/resnet50/configs/model_evaluation_config_demo.json"


@inputs(model_config_path=Types.Blob)
@outputs(model_config_string=Types.String)
@python_task(cache=False, memory_request="500Mi")
def validate_model_config(wf_params, model_config_path, model_config_string):
    with model_config_path as model_config_fileobj:
        config = ujson.loads(model_config_fileobj.read())
    model_config_string.set(ujson.dumps(config))
    # VALIDATE HERE


@inputs(
    model=Types.Blob,
    evaluation_clean_mpblob=Types.MultiPartBlob,
    evaluation_dirty_mpblob=Types.MultiPartBlob,
)
@outputs(
    ground_truths_out=[Types.Integer],
    predictions_out=[[Types.Float]],
)
@python_task(cache=True, cache_version="1", gpu_request="1", gpu_limit="1", memory_request="64Gi")
def evaluate_on_datasets(
    wf_params,
    model,
    evaluation_clean_mpblob,
    evaluation_dirty_mpblob,
    ground_truths_out,
    predictions_out,
):
    """ Map prediction task on a set of zip files of images to sub tasks"""

    with flytekit_utils.AutoDeletingTempDir("results") as output_models_dir:
        with flytekit_utils.AutoDeletingTempDir("evaluation") as evaluation_dir:
            download_data(evaluation_dir.name, {"clean": evaluation_clean_mpblob, "dirty": evaluation_dirty_mpblob})
            model.download()
            ground_truths, predictions = predict_with_resnet50_model(
                model_path=model.local_path,
                evaluation_dataset=evaluation_dir.name,
                batch_size=DEFAULT_BATCH_SIZE,
                img_size=DEFAULT_IMG_SIZE,
            )

            ground_truths_out.set(ground_truths)
            predictions_out.set(predictions)


@inputs(
    ground_truths=[Types.Integer],
    predictions=[[Types.Float]]
)
@outputs(
    result_blobs=[Types.Blob],
    result_files_names=[Types.String]
)
@python_task(cache=True, cache_version="1")
def analyze_prediction_results(
    wf_params,
    ground_truths,
    predictions,
    result_blobs,
    result_files_names,
):

    temp_result_dir = wf_params.working_directory.get_named_tempfile("results")
    Path(temp_result_dir).mkdir(0o777, parents=True, exist_ok=False)

    tpr, fpr, roc_thresholds = calculate_roc_curve(
        ground_truths,
        predictions,
        pos_label_idx=DEFAULT_CLASS_LABELS.index(DEFAULT_POSITIVE_LABEL),
    )

    export_results(
        path=temp_result_dir + "/roc.csv",
        columns={"tpr": tpr, "fpr": fpr, "thresholds": roc_thresholds},
        index_col="thresholds",
    )

    precisions, recalls, prc_thresholds = calculate_precision_recall_curve(
        ground_truths,
        predictions,
        pos_label_idx=DEFAULT_CLASS_LABELS.index(DEFAULT_POSITIVE_LABEL),
    )

    export_results(
        path=temp_result_dir + "/precision_recall_curve.csv",
        columns={
            "precision": precisions,
            "recall": recalls,
            "thresholds": prc_thresholds,
        },
        index_col="thresholds",
    )

    # write results to output
    blobs, files_names_list = collect_blobs(temp_result_dir)
    result_blobs.set(blobs)
    result_files_names.set(files_names_list)


@inputs(model=Types.Blob)
@outputs(model_blob=Types.Blob)
@python_task(cache=True, cache_version="1")
def fetch_model(wf_params, model, model_blob):
    if not model:
        print("Fetching model from a pinned previous execution")
        classifier_train_wf_exec = fetch_workflow_execution(
            project=DEFAULT_PROJECT_NAME, domain=DEFAULT_DOMAIN, exec_id=DEFAULT_CLASSIFIER_TRAIN_WF_EXECUTION_ID)
        model_blob.set(classifier_train_wf_exec.outputs["trained_models"][1])  # resnet50_final.h5
    else:
        model_blob.set(model.uri)


@inputs(
    ground_truths=[Types.Integer],
    probabilities=[[Types.Float]])
@outputs(predictions=[Types.Integer], threshold=Types.Float, thresholds=[Types.Float])
@python_task(cache=False, cache_version="1")
def generate_predictions(wf_params, ground_truths, probabilities, predictions, threshold, thresholds):
    pos_label_idx = DEFAULT_CLASS_LABELS.index(DEFAULT_POSITIVE_LABEL)
    tpr, fpr, roc_thresholds = calculate_precision_recall_curve(
        ground_truths,
        probabilities,
        pos_label_idx=pos_label_idx,
    )

    threshold_val = float(calculate_cutoff_youdens_j(tpr, fpr, roc_thresholds))
    predictions.set([1 if p[pos_label_idx] > threshold_val else 0 for p in probabilities])
    threshold.set(threshold_val)
    thresholds.set([float(v) for v in roc_thresholds])


@inputs(
    ground_truths=[Types.Integer],
    probabilities=[[Types.Float]])
@outputs(predictions=[Types.Integer])
@python_task(cache=False, cache_version="1.0")
def predict(wf_params, ground_truths, probabilities, predictions):
    predictions.set([0 if p[0] > p[1] else 1 for p in probabilities])


@workflow_class
class ClassifierEvaluateWorkflow:
    available_streams_mpblobs = Input([Types.MultiPartBlob], required=True)
    available_streams_names = Input([Types.String], required=True)
    validation_data_ratio = Input(Types.Float, default=DEFAULT_VALIDATION_DATA_RATIO)
    streams_metadata_path = Input(Types.String, required=True)
    model = Input(Types.Blob, default=None)
    evaluation_config_json = Input(Types.Generic, default=ujson.loads(open(DEFAULT_EVALUATION_CONFIG_FILE).read()))

    fetch_model_task = fetch_model(
        model=model
    )

    rearrange_data_task = rearrange_data(
        available_streams_mpblobs=available_streams_mpblobs,
        available_streams_names=available_streams_names,
        training_validation_config_json=evaluation_config_json,
        streams_metadata_path=streams_metadata_path,
        validation_data_ratio=validation_data_ratio,
    )

    evaluate_on_datasets_task = evaluate_on_datasets(
        model=fetch_model_task.outputs.model_blob,
        evaluation_clean_mpblob=rearrange_data_task.outputs.validation_clean_mpblob,
        evaluation_dirty_mpblob=rearrange_data_task.outputs.validation_dirty_mpblob,
    )

    analyze_task = analyze_prediction_results(
        ground_truths=evaluate_on_datasets_task.outputs.ground_truths_out,
        predictions=evaluate_on_datasets_task.outputs.predictions_out,
    )

    predict = predict(
        ground_truths=evaluate_on_datasets_task.outputs.ground_truths_out,
        probabilities=evaluate_on_datasets_task.outputs.predictions_out
    )

    analyze_results_blobs = Output(analyze_task.outputs.result_blobs, sdk_type=[Types.Blob])
    analyze_results_files_names = Output(analyze_task.outputs.result_files_names, sdk_type=[Types.String])
    ground_truths = Output(evaluate_on_datasets_task.outputs.ground_truths_out, sdk_type=[Types.Integer])
    predictions = Output(predict.outputs.predictions, sdk_type=[Types.Integer])


evaluate_lp = ClassifierEvaluateWorkflow.create_launch_plan()
