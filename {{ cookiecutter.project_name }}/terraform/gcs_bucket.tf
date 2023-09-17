provider "google" {
  credentials = file("~/.secrets/gcp_credentials.json")
  project = "ml-projects-399119"
}

resource "google_storage_bucket" "ml-projects-dev" {
  name     = "ml-projects-dev-bucket"
  location = "US"
  storage_class = "STANDARD"
  force_destroy = true
}
