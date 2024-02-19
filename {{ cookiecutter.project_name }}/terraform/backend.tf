terraform {
backend "gcs" {
  bucket = "{{ cookiecutter.project_id }}-tfstate-bucket"   # GCS bucket name to store terraform tfstate
  prefix = {{ cookiecutter.project_name }}
  }
}