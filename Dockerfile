FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest
LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER root

RUN apt-get update \
    && apt-get install -yq --no-install-recommends vim emacs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER $NB_USER

ENV TORCH=1.12.0
ENV CUDA=cu116

RUN mamba install -c pytorch -c conda-forge pytorch=1.12.0 torchvision torchaudio cudatoolkit=11.6 \
    && mamba clean --all -f -y

RUN set -x \
    pip3 install coffea tables mplhep jetnet weaver-core pre-commit ogb \
    && pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html \
    && pip3 install pyg-nightly

RUN set -x \
    fix-permissions /home/$NB_USER
