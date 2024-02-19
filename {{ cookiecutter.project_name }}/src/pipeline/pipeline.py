import argparse
from datetime import datetime
from typing import List

import yaml
from constants import (
    COMPILED_PIPELINE_PATH,
    INGESTOR_DOCKER_IMAGE,
    MODEL_DOCKER_IMAGE,
    PIPELINE_NAME,
    PREPROCESSING_DOCKER_IMAGE,
    PROJECT_ID,
)
from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.vertex_notification_email import (
    VertexNotificationEmailOp,
)
from kfp import compiler

# SCHEDULE_DISPLAY_NAME,
from kfp.dsl import (
    Artifact,
    Dataset,
    ExitHandler,
    Input,
    Metrics,
    Model,
    Output,
    component,
    pipeline,
)


@component(base_image=INGESTOR_DOCKER_IMAGE)
def ingestor(
    project_id: str,
    gcs_bucket_path: str,
    dummy_data: Output[Dataset],
) -> None:
    from ingestor import run_ingestor

    (output_df) = run_ingestor(gcs_bucket_path, project_id)

    dataframes = [output_df]
    outputs = [dummy_data]

    for df, output in zip(dataframes, outputs, strict=True):
        df.to_parquet(output.path)
        output.metadata["shape"] = df.shape


@component(base_image=PREPROCESSING_DOCKER_IMAGE)
def preprocess(
    gcs_bucket_path: str,
    dummy_data: Input[Dataset],
    preprocessed_dataset: Output[Dataset],
) -> None:
    from preprocess import run_preprocess

    output_preprocessed_dataset = run_preprocess(
        gcs_bucket_path,
        dummy_data.uri,
    )

    dataframes = [output_preprocessed_dataset]
    outputs = [dummy_data]

    for df, output in zip(dataframes, outputs, strict=True):
        df.to_parquet(output.path)
        output.metadata["shape"] = df.shape


@component(base_image=MODEL_DOCKER_IMAGE)
def model(
    project_id: str,
    gcs_bucket_path: str,
    enable_gpu: bool,
    preprocessed_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    predictions: Output[Dataset],
    optuna_experiments_url: Output[Artifact],
) -> None:
    import logging

    import joblib
    from google.cloud import storage
    from model import run_model

    (output_model, output_metrics) = run_model(
        project_id,
        gcs_bucket_path,
        preprocessed_data.uri,
        enable_gpu,
    )

    local_path = "model.joblib"
    joblib.dump(output_model, local_path)
    blob = storage.blob.Blob.from_string(model.uri, client=storage.Client())
    blob.upload_from_filename(local_path)
    logging.info(f"model saved at: {model.uri}")

    for key, value in output_metrics.items():
        metrics.log_metric(key, value)

    logging.info(f"metrics saved at: {metrics.uri}")

    # Initialize AI Platform client
    # aiplatform.init(project=project_id, location=location)

    # Create a model resource
    # model = aiplatform.Model(display_name=model_display_name)
    # model.upload(model.uri)

    # Deploy the model to an endpoint
    # endpoint = model.deploy(
    #    machine_type="n1-standard-4",
    #    endpoint_display_name="your-endpoint-name"
    # )


@pipeline
def example_pipeline(
    project_id: str,
    gcs_bucket_path: str,
    email_notification_list: List[str],
) -> None:
    notify_email_task = VertexNotificationEmailOp(recipients=email_notification_list)

    with ExitHandler(notify_email_task):
        ingestor(
            project_id=project_id,
            gcs_bucket_path=gcs_bucket_path,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["submit", "schedule"], required=True)
    parser.add_argument("--env", choices=["dev", "stg", "prod"], required=True)
    parser.add_argument(
        "--enable_hyperparameter_tuning", default=False, action="store_true"
    )
    args = parser.parse_args()

    with open("env_config.yaml", "r") as config_file:
        env_config = yaml.safe_load(config_file)[args.env]

    compiler.Compiler().compile(example_pipeline, COMPILED_PIPELINE_PATH)

    email_notification_list = list(
        filter(None, env_config["email_notification_list"](";"))
    )

    params_dict = {
        "project_id": PROJECT_ID,
        "gcs_bucket_path": env_config["gcs_bucket_path"],
        "email_notification_list": email_notification_list,
        "enable_hyperparameter_tuning": args.enable_hyperparameter_tuning,
        "timeout_in_hours": env_config["timeout_in_hours"],
    }

    current_date = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    pipeline_job = aiplatform.PipelineJob(
        project=PROJECT_ID,
        display_name=PIPELINE_NAME,
        template_path=COMPILED_PIPELINE_PATH,
        pipeline_root=args.gcs_bucket_path,
        job_id=f"{PIPELINE_NAME}-{current_date}",
        enable_caching=False,
        parameter_values=params_dict,
    )

    if args.mode == "submit":
        pipeline_job.submit(service_account=args.service_account)
    elif args.mode == "schedule":
        cron_schedule = (
            env_config["tuning_schedule"]
            if args.enable_hyperparameter_tuning
            else env_config["cron_schedule"]
        )
        pipeline_job.create_schedule(
            display_name=f"{PIPELINE_NAME}-{current_date}",
            cron=cron_schedule,
        )


if __name__ == "__main__":
    main()
