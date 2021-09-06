# Dockerfile to setup and run the Cascade toolbox for Calibrated spike inference from calcium imaging data in a Docker container
# Repository: https://github.com/HelmchenLabSoftware/Cascade
# See README_Docker.md for instructions


ARG BASE_CONTAINER=jupyter/scipy-notebook:9b87b1625445
FROM $BASE_CONTAINER

LABEL maintainer="Henry Luetcke <hluetcke@ethz.ch>"

# install the Python dependencies
COPY requirements_mac.txt environment_mac.yml /tmp/
RUN conda env update -q -f /tmp/environment_mac.yml && \
	pip install --quiet --no-cache-dir -r /tmp/requirements_mac.txt && \
	conda clean -y --all && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
