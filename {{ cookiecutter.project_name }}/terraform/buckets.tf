# Specify the GCP Provider
provider "google" {
project = var.project_id
region  = var.region
}

# Create GCS Buckets
resource "google_storage_bucket" "dev_bucket" {
name     = "${var.bucket_name}-dev"
location = var.region
}

resource "google_storage_bucket" "stg_bucket" {
name     = "${var.bucket_name}-stg"
location = var.region
}

resource "google_storage_bucket" "prod_bucket" {
name     = "${var.bucket_name}-prod"
location = var.region
}