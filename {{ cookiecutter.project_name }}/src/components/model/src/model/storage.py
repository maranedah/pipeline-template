import datetime as dt 
import json 
import pathlib

from google.cloud import storage 

class OptunaStorage:
    def __init__(self, gcs_bucket: str):
        self.client = storage.Client()
        self.bucket_name = pathlib.Path(gcs_bucket).parts[1]
        self.bucket = self.client.bucket(self.bucket_name)

    def store_best_trial_params(self, best_trial_params: dict[str, float | int | str]):
        json_data = json.dumps(best_trial_params)
        today = str(dt.date.today())
        blob = self.bucket.blob(f"best_params/{today}.json")
        blob.upload_from_string(json_data)
        blob_url = self.get_blob_url(blob)
        return blob_url

    def get_stored_params(self):
        blobs = list(self.client.list_blobs(self.bucket_name, prefix="best_params/"))
        if len(blobs) == 0:
            return None, None 
        latest_blob = max(blobs, key=lambda x: x.time_created)
        file_contents = latest_blob.download_as_string()
        params_dict = json.loads(file_contents)
        latest_blob_url = self.get_blob_url(latest_blob)
        return params_dict, latest_blob_url

    def get_blob_url(self, blob):
        return "gs://" + "/".join(pathlib.Path(blob.public_url).parts[2:])