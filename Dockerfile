FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest
LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER $NB_USER

ENV USER=${NB_USER}

RUN sudo apt-get update \
    && sudo apt-get install -y vim emacs \
    && sudo rm -rf /var/lib/apt/lists/*

ENV TORCH=1.11.0
ENV CUDA=cu115

RUN set -x \
    pip3 install coffea tables mplhep jetnet weaver-core pre-commit \
    pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

RUN set -x \
    && fix-permissions /home/$NB_USER
