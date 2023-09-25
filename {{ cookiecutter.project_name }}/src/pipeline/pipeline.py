import argparse
import datetime as dt
from typing import List

from constants import (
    COMPILED_PIPELINE_PATH,
    INGESTOR_DOCKER_IMAGE,
    MODEL_DOCKER_IMAGE,
    PIPELINE_NAME,
    PREPROCESSING_DOCKER_IMAGE,
    PROJECT_ID,
    SCHEDULE_DISPLAY_NAME,
)
from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.vertex_notification_email import(
    VertexNotificationEmailOp
)
from kfp import compiler
from kfp.dsl import (
    Artifact,
    Dataset,
    ExitHandler,
    Metrics,
    Model,
    Output,
    component,
    pipeline,
)


@component(base_image=INGESTOR_DOCKER_IMAGE)
def ingestor(
    gcs_bucket_path: str,
    project_id: str,
    dataset: Output[Dataset]
) -> None:
    from ingestor import run_ingestor

    (output_df) = run_ingestor(gcs_bucket_path, project_id)

    dataframes = [output_df]
    outputs = [dataset]  # put the outputs from the function

    for df, output in zip(dataframes, outputs, strict=True):
        df.to_parquet(output.path)
        output.metadata["shape"] = df.shape


@component(base_image=PREPROCESSING_DOCKER_IMAGE)
def preprocess(
    gcs_bucket: str,
    preprocessed_dataset: Output[Dataset]
) -> None:
    from preprocess import run_preprocess

    output_preprocessed_dataset = run_preprocess(gcs_bucket_path)

    dataframes = [output_preprocessed_dataset]
    outputs = [preprocessed_dataset]

    for df, output in zip(dataframes, outputs, strict=True):
        df.to_parquet(output.path)
        output.metadata["shape"] = df.shape

@component(base_image=MODEL_DOCKER_IMAGE)
def model(
    gcs_bucket_path: str,
    model: Output[Model],
    metrics: Output[Model],
    gcs_params_url: Output[Artifact]
) -> None:
    import logging 

    import joblib
    from google.cloud import storage
    from model import run_model

    (output_model, output_metrics, output_gcs_params_url) = run_model(
        gcs_bucket_path
    )

    local_path = "model.joblib"
    joblib.dump(output_model, local_path)
    blob = storage.blob.Blob.from_string(model.uri, client=storage.Client())
    blob.upload_from_filename(local_path)
    logging.info("model saved at:" + model.uri)

    for key, value in output_metrics.items():
        metrics.log_metric(key, value)

    logging.info("metrics saved at:" + metrics.uri)
    gcs_params_url.uri = output_gcs_params_url


@pipeline
def example_pipeline(
    gcs_bucket_path: str,
    project_id: str,
    email_notification_list: List[str],
) -> None:
    notify_email_task = VertexNotificationEmailOp(recipients=email_notification_list)

    with ExitHandler(notify_email_task):
        ingestor_step = ingestor(
            gcs_bucket_path,
            project_id,
        )
        preprocess_step = preprocess(
            gcs_bucket_path
        )

        model(
            gcs_bucket_path
        )


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--mode", choices=["submit", "schedule"], required=True)
    args.add_argument("--gcs_bucket_path", required=True)
    args.add_argument("--service_account", required=True)
    args.add_argument("--email_notification_list", required=True)
    args.add_argument("--cron_schedule")

    if args.mode == "schedule" and not args.cron_scheduler:
        raise ValueError("--cron_schedule is required when using schedule mode")

    compiler.Compiler().compile(example_pipeline, COMPILED_PIPELINE_PATH)

    email_notification_list = list(
        filter(None, args.email_notification_list.split(";"))
    )
    current_date = dt.utcnow().strftime("%Y%m%d-%H%M%S")

    params_dict = {
        "gcs_bucket_path": args.gcs_bucket_path,
        "project_id": PROJECT_ID,
        "email_notification_list": email_notification_list,
    }

    pipeline_job = aiplatform.PipelineJob(
        project=PROJECT_ID,
        display_name=PIPELINE_NAME,
        template_path=COMPILED_PIPELINE_PATH,
        pipeline_root=args.gcs_bucket_path,
        job_id=f"{PIPELINE_NAME}-{current_date}",
        enable_caching=False,
        parameter_values=params_dict
    )

    if args.mode == "submit":
        pipeline_job.submit(service_account=args.service_account)
    # elif args.mode == "schedule":
    # 	create_or_update_pipeline_schedule()


if __name__ == "__main__":
    main()
