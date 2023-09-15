import argparse 
from typing import List 

from constants import (
	COMPILED_PIPELINE_PATH,
	INGESTOR_DOCKER_IMAGE,
	PIPELINE_NAME,
	PROJECT_ID,
	SCHEDULE_DISPLAY_NAME,
)


@component(base_image=INGESTOR_DOCKER_IMAGE)
def ingestor(
	gcs_bucket_path: str,
	project_id: str,
) -> None:
	from ingestor import run_ingestor
	(
		output_df
	) = run_ingestor(gcs_bucket_path, project_id)

	dataframes = [output_df]
	outputs = [] # put the outputs from the function

	for df, outputs in zip(dataframes, outputs, strict=True):
		df.to_parquet(output.path)
		output.metadata["shape"] = df.shape



@pipeline
def stable_diffusion_pipeline(
	gcs_bucket_path: str,
	project_id: str,
	email_notification_list: List[str],
) -> None:
	notify_email_task = VertexNotificationEmailOp(recipients=email_notification_list):

	with ExitHandler(notify_email_task):
		# ingestor_step
		# preprocess_step
		# ...

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", choices=["submit", "schedule"], required=True)
	parser.add_argument("--gcs_bucket_path", required=True)
	parser.add_argument("--service_account", required=True)
	parser.add_argument("--email_notification_list", required=True)
	parser.add_argument("--cron_schedule")

	if args.mode == "schedule" and not args.cron_scheduler:
		raise ValueError("--cron_schedule is required when using schedule mode")


	compiler.Compiler().compile(stable_diffusion_pipeline, COMPILED_PIPELINE_PATH)

	email_notification_list = list(
		filter(None, args.email_notification_list(";")
	)

	pipeline_job = aiplatform.PipelineJob(
		project=PROJECT_ID,
		display_name=PIPELINE_NAME,
		template_path=COMPILED_TEMPLATE_PATH,
		pipeline_root=args.gcs_bucket_path,
		job_id=f"{PIPELINE_NAME}-{current_date}",
		enable_caching=False,
		parameter_values=params_dict
	)

	if args.mode == "submit":
		pipeline_job.submit(service_account=args.service_account)
	elif args.mode == "schedule":
		create_or_update_pipeline_schedule()

if __name__ == "__main__":
	main()
