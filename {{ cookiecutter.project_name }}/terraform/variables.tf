variable "project_id" {
description = "Google Project ID."
type        = string
}

variable "bucket_name" {
description = "GCS Bucket name. Value should be unique ."
type        = string
}

variable "region" {
description = "Google Cloud region"
type        = string
default     = "europe-west2"
}

variable "bucket_suffixes" {
  default     = ["dev", "stg", "prod"]
  description = "List of suffixes for the buckets"
}