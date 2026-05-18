#!/usr/bin/env Rscript
Sys.setenv(RENV_CONFIG_SANDBOX_ENABLED = "FALSE")
Sys.setenv(RENV_CONFIG_PROMPT = "FALSE")
options(BiocManager.check_repositories = FALSE)

script_path <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    normalizePath(sub("^--file=", "", file_arg[1]))
  } else {
    normalizePath(sys.frame(1)$ofile)
  }
}, error = function(e) file.path(getwd(), "install_r_packages.R"))
r_project <- dirname(script_path)
setwd(r_project)
message("r_project: ", r_project)

bioc_version <- "3.22"
bioc_packages <- c("anndataR", "rhdf5")

Sys.setenv(RENV_CONFIG_EXTERNAL_LIBRARIES = paste(.libPaths(), collapse = ","))
if (file.exists(file.path(r_project, "renv.lock"))) {
  renv::activate(project = r_project)
} else {
  renv::init(project = r_project, bare = TRUE, restart = FALSE)
}

renv::install("BiocManager")
BiocManager::install(version = bioc_version, ask = FALSE, update = TRUE, force = TRUE)
BiocManager::install("BiocVersion", version = bioc_version, ask = FALSE, force = TRUE)

for (pkg in bioc_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    BiocManager::install(pkg, ask = FALSE, update = FALSE, version = bioc_version)
    message("Installed Bioconductor package: ", pkg)
  } else {
    message("Bioconductor package already installed: ", pkg)
  }
}

# Install local forks.
renv::install(paste0("local::", file.path(r_project, "lle_1.1.tar.gz")))
renv::install(paste0("local::", file.path(r_project, "SLICER")))
renv::install(paste0("local::", file.path(r_project, "splatter")))

renv::snapshot(prompt = FALSE)
message("renv snapshot complete")
