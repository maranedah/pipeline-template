provider "google" {
  credentials = file("C:\\Users\\mauri\\Desktop\\Mauro\\projects\\ml-projects-dev-c7c3a47c0879.json")
  project     = "{{ cookiecutter.project_id }}"
  region      = "us-central1"
}

resource "google_storage_bucket" "dev_bucket" {
  name     = "{{ cookiecutter.project_name }}-bucket-dev"
  location = "US"
}
