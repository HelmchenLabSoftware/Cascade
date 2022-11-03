# Dockerfile to setup and run the Cascade toolbox for Calibrated spike inference from calcium imaging data in a Docker container
# Repository: https://github.com/HelmchenLabSoftware/Cascade
# See README_Docker.md for instructions


ARG BASE_CONTAINER=jupyter/scipy-notebook:latest
FROM $BASE_CONTAINER

LABEL maintainer="Henry Luetcke <hluetcke@ethz.ch>"

# install the Python dependencies
COPY environment_mac.yml /tmp/
RUN conda env create -f /tmp/environment_mac.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "Cascade", "/bin/bash", "-c"]

RUN conda clean -y --all && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "Cascade", "jupyter", "notebook"]
