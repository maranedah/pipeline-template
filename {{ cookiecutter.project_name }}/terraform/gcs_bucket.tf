provider "google" {
  project = "ml-projects-399119"
  region = "us-central1"
}

resource "google_service_account" "sa" {
  project    = "ml-projects-399119"
  account_id = "test-storage-sa"
}

resource "google_project_iam_member" "project" {
  project = "ml-projects-399119"
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.sa.email}"
}


resource "google_storage_bucket" "ml-projects-dev" {
  name     = "ml-projects-dev-bucket"
  location = "US"
  storage_class = "STANDARD"
  force_destroy = true
}

module "oidc" {
  source      = "terraform-google-modules/github-actions-runners/google//modules/gh-oidc"
  project_id  = "ml-projects-399119"
  pool_id     = "example-pool"
  provider_id = "example-gh-provider"
  sa_mapping = {
    (google_service_account.sa.account_id) = {
      sa_name   = google_service_account.sa.name
      attribute = "attribute.repository/user/repo"
    }
  }
}
