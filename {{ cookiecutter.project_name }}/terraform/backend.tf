terraform {
backend "gcs" {
  bucket = "terraform-state-bucket"   # GCS bucket name to store terraform tfstate
  prefix = "{{ cookiecutter.project_name }}"           # Update to desired prefix name. Prefix name should be unique for each Terraform project having same remote state bucket.
  }
}