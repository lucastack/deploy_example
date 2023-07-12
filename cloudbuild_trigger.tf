resource "google_cloudbuild_trigger" "filename-trigger" {
  location = var.location

  trigger_template {
    branch_name = "main"
    repo_name   = "deploy_example"
  }

  substitutions = {
    _REGION = var.location
  }

  filename = "cloudbuild.yaml"
}
