import datetime
import json
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pytz
from google.cloud import storage
from jinja2 import Template


def read_metrics(bucket_name: str, prefix: str) -> dict[str, float]:
    # metrics are stored in uri formatted as:
    # gs://{bucket_name}/{project_number}/{pipeline_job_id}/model_*/metrics
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        metrics_blob = None
        for blob in blobs:
            if re.match(rf"{prefix}/model_\d+/metrics", blob.name):
                metrics_blob = blob
                break

        json_string = metrics_blob.open("r").read().replace("'", '"')
        metrics_dict = json.loads(json_string)

    except Exception:
        metrics_dict = None

    return metrics_dict


def get_elapsed_time_str(
    current_time: datetime.datetime, start_time: datetime.datetime
) -> str:
    elapsed_time = current_time - start_time
    elapsed_hours = elapsed_time.seconds // 3600 + elapsed_time.days * 24
    elapsed_minutes = elapsed_time.seconds // 60 % 60
    elapsed_time_str = f"{elapsed_hours}Hrs {elapsed_minutes}Min"
    return elapsed_time_str


def get_start_time(pipeline_job_id, santiago_timezone):
    # The formatting of datetime can be either
    # YYYYMMDDHHMMSS(ms)(ms)(ms)
    # or YYYYMMDD-HHMMSS
    # depending if the ppeline is scheduled or submitted
    datetime_pattern = r"\d+(-)*\d+"
    numeric_part = re.search(datetime_pattern, pipeline_job_id).group()
    scheduling_pattern = r"\d{17}"
    submitted_pattern = r"\d{8}-\d{6}"
    is_scheduling_datetime_format = re.match(scheduling_pattern, numeric_part)
    is_submitted_datetime_format = re.match(submitted_pattern, numeric_part)

    if is_scheduling_datetime_format:
        start_time = datetime.datetime.strptime(numeric_part[:-3], "%Y%m%d%H%M%S")
    elif is_submitted_datetime_format:
        start_time = datetime.datetime.strptime(numeric_part, "%Y%m%d-%H%M%S")
    else:
        raise Exception

    return start_time.astimezone(santiago_timezone)


def email_notification(
    sender: str,
    recipients: list[str],
    subject: str,
    metrics_to_notify: list[str],
    project_id: str,
    gcs_bucket: str,
    status: str,
    pipeline_job_resource_name: str,
) -> None:
    mailing_path = Path(__file__).parent / "template"
    metrics_table = open(mailing_path / "metrics_table.txt", "r").read()
    content = open(mailing_path / "content.txt", "r").read()
    footnote = open(mailing_path / "footnote.txt", "r").read()

    bucket_name = gcs_bucket.split("gs://")[-1]
    project_number = pipeline_job_resource_name.split("/")[1]
    pipeline_job_id = pipeline_job_resource_name.split("/")[-1]

    metrics = read_metrics(bucket_name, prefix=f"{project_number}/{pipeline_job_id}")

    template_table = Template(metrics_table)

    table_content = ""
    if metrics:
        metrics_as_kwargs = {
            metric_name: round(metrics[metric_name], 3)
            for metric_name in metrics_to_notify
        }
        table_content = template_table.render(**metrics_as_kwargs)

    template_content = Template(content)
    santiago_timezone = pytz.timezone("Chile/Continental")
    current_time = datetime.datetime.now(pytz.utc).astimezone(santiago_timezone)

    start_time = get_start_time(pipeline_job_id, santiago_timezone)
    elapsed_time = get_elapsed_time_str(current_time, start_time)

    msg_content = template_content.render(
        project_id=project_id,
        job_id=pipeline_job_id,
        status=status,
        current_time=current_time.strftime("%d-%m-%Y %H:%M:%S"),
        start_time=start_time.strftime("%d-%m-%Y %H:%M:%S"),
        elapsed_time=elapsed_time,
    )

    msg = MIMEMultipart("alternative")
    msg.attach(MIMEText(table_content + msg_content + footnote, "html"))

    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ",".join(recipients)

    s = smtplib.SMTP("")

    s.send_message(msg)
    s.quit()
