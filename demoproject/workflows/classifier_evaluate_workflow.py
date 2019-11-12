from pathlib import Path

import os
import shutil
import ujson
from flytekit.sdk.workflow import workflow_class, Output, Input
from flytekit.sdk.types import Types
from flytekit.sdk.tasks import python_task, inputs, outputs
from flytekit.common import utils as flytekit_utils
from models.classifier.resnet50.train_resnet50 import train_resnet50_model, download_data
from utils.flyte_utils.fetch_executions import fetch_workflow_execution
from models.classifier.resnet50.evaluate_resnet50 import predict_with_resnet50_model
from models.classifier.resnet50.constants import DEFAULT_BATCH_SIZE, DEFAULT_IMG_SIZE
from models.classifier.resnet50.constants import DEFAULT_CLASS_LABELS, DEFAULT_POSITIVE_LABEL
from utils.metric_utils.metric_utils import calculate_roc_curve, calculate_precision_recall_curve, calculate_cutoff_youdens_j, export_results
from utils.flyte_utils.collect_blobs import collect_blobs

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


@inputs(
    evaluation_config_path=Types.String,  # The path to a json file listing the streams needed for training, and other parameters
    evaluation_config_json=Types.Generic,
    streams_metadata_path=Types.String,  # The path to a json file listing the metadata (e.g., class) of each stream
)
@outputs(
    evaluation_clean_mpblob=Types.MultiPartBlob,
    evaluation_dirty_mpblob=Types.MultiPartBlob,
)
@python_task(cache=True, cache_version="1", memory_request="16Gi")
def rearrange_data(
        wf_params,
        evaluation_config_path,
        evaluation_config_json,
        streams_metadata_path,
        evaluation_clean_mpblob,
        evaluation_dirty_mpblob,
):
    # Get the latest execution of the data_prep_workflow
    dataprep_wf_execution = fetch_workflow_execution(
        project=DEFAULT_PROJECT_NAME, domain=DEFAULT_DOMAIN, exec_id=DEFAULT_DATAPREP_WF_EXECUTION_ID)

    print("Data Prep Workflow:")
    print(dataprep_wf_execution)

    available_streams_mpblobs = dataprep_wf_execution.outputs["selected_frames_mpblobs"]
    available_streams_names = dataprep_wf_execution.outputs["streams_names_out"]

    # Download the config file and metadata
    training_validation_config_blob = Types.Blob.fetch(remote_path=evaluation_config_path)
    config_fp = open(training_validation_config_blob.local_path)
    config = ujson.load(config_fp)

    streams_metadata_blob = Types.Blob.fetch(remote_path=streams_metadata_path)
    metadata_fp = open(streams_metadata_blob.local_path)
    streams_metadata = ujson.load(metadata_fp)

    all_streams = streams_metadata.get("streams", {})
    wf_params.logging.info("all streams from metadata: ")
    wf_params.logging.info(all_streams)
    selections = config.get("evaluation_datasets", {})
    wf_params.logging.info("selections: ")
    wf_params.logging.info(selections)
    evaluation_streams = [{"stream": name, "class": metadata["class"]} for name, metadata in all_streams.items()
                          if name in selections.keys()]

    # Splitting the set of streams into validation and training
    labeled_streams = {
        "clean": [s["stream"] for s in evaluation_streams if s["class"] == "clean"],
        "dirty": [s["stream"] for s in evaluation_streams if s["class"] == "dirty"],
    }

    # Download multipartblobs to the target folders and then upload it
    # final_mpblobs = {k: {} for k in PURPOSES}
    final_mpblobs = {
        'dirty': evaluation_dirty_mpblob,
        'clean': evaluation_clean_mpblob,
    }
    for label in labeled_streams.keys():
        with flytekit_utils.AutoDeletingTempDir() as output_dir:
            for stream in labeled_streams[label]:
                idx = available_streams_names.index(stream)
                mpblob = available_streams_mpblobs[idx]
                mpblob.download()
                files = os.listdir(mpblob.local_path)
                for f in files:
                    shutil.move(os.path.join(mpblob.local_path, f), output_dir.name)
                files = os.listdir(output_dir.name)
                print("There are {} files in output dir {} ({})".format(len(files), output_dir.name, label))
            final_mpblobs[label].set(output_dir.name)


@inputs(model_path=Types.String)
@outputs(model_blob=Types.Blob)
@python_task(cache=True, cache_version="1")
def fetch_model(wf_params, model_path, model_blob):
    if not model_path:
        print("Fetching model from a pinned previous execution")
        classifier_train_wf_exec = fetch_workflow_execution(
            project=DEFAULT_PROJECT_NAME, domain=DEFAULT_DOMAIN, exec_id=DEFAULT_CLASSIFIER_TRAIN_WF_EXECUTION_ID)
        model_blob.set(classifier_train_wf_exec.outputs["trained_models"][1])  # resnet50_final.h5
    else:
        b = Types.Blob.fetch(model_path)
        model_blob.set(b)


@inputs(
    ground_truths=[Types.Integer],
    probabilities=[[Types.Float]])
@outputs(predictions=[Types.Integer], threshold=Types.Float, thresholds=[Types.Float])
@python_task(cache=False, cache_version="1")
def generate_predictions(wf_params, ground_truths, probabilities, predictions, threshold, thresholds):
    pos_label_idx = DEFAULT_CLASS_LABELS.index(DEFAULT_POSITIVE_LABEL)
    tpr, fpr, roc_thresholds = calculate_roc_curve(
        ground_truths,
        probabilities,
        pos_label_idx=pos_label_idx,
    )

    threshold_val = float(calculate_cutoff_youdens_j(tpr, fpr, roc_thresholds))
    predictions.set([1 if t > threshold_val else 0 for t in ground_truths])
    threshold.set(threshold_val)
    thresholds.set([float(v) for v in roc_thresholds])


@workflow_class
class ClassifierEvaluateWorkflow:
    streams_metadata_path = Input(Types.String, required=True)
    model_path = Input(Types.String, default="")
    evaluation_config_path = Input(Types.String, default=DEFAULT_EVALUATION_CONFIG_FILE)
    evaluation_config_json = Input(Types.Generic, default=ujson.loads(open(DEFAULT_EVALUATION_CONFIG_FILE).read()))

    # validate_model_config_task = validate_model_config(
    #     model_config_path=model_config_path
    # )

    fetch_model_task = fetch_model(
        model_path=model_path
    )

    rearrange_data_task = rearrange_data(
        evaluation_config_path=evaluation_config_path,
        evaluation_config_json=evaluation_config_json,
        streams_metadata_path=streams_metadata_path,
    )

    evaluate_on_datasets_task = evaluate_on_datasets(
        model=fetch_model_task.outputs.model_blob,
        evaluation_clean_mpblob=rearrange_data_task.outputs.evaluation_clean_mpblob,
        evaluation_dirty_mpblob=rearrange_data_task.outputs.evaluation_dirty_mpblob,
    )

    analyze_task = analyze_prediction_results(
        ground_truths=evaluate_on_datasets_task.outputs.ground_truths_out,
        predictions=evaluate_on_datasets_task.outputs.predictions_out,
    )

    predict = generate_predictions(
        ground_truths=evaluate_on_datasets_task.outputs.ground_truths_out,
        probabilities=evaluate_on_datasets_task.outputs.predictions_out
    )

    analyze_results_blobs = Output(analyze_task.outputs.result_blobs, sdk_type=[Types.Blob])
    analyze_results_files_names = Output(analyze_task.outputs.result_files_names, sdk_type=[Types.String])
    ground_truths = Output(evaluate_on_datasets_task.outputs.ground_truths_out, sdk_type=[Types.Integer])
    predictions = Output(predict.outputs.predictions, sdk_type=[Types.Integer])
