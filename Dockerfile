FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest
LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER $NB_USER
ENV USER=${NB_USER}

RUN sudo apt-get update \
    && sudo apt-get install -yq --no-install-recommends vim emacs \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

ENV TORCH=1.12.0
ENV CUDA=cu116

RUN mamba install -c pytorch -c conda-forge pytorch=1.12.0 torchvision torchaudio cudatoolkit=11.6 \
    && mamba clean --all -f -y

RUN set -x \
    pip3 install coffea \
    	 	 tables \
		 mplhep \
		 jetnet \
		 weaver-core \
		 pre-commit \
		 torch-scatter \
		 torch-sparse \
		 torch-cluster \
		 torch-spline-conv \
		 pyg-nightly \
		 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

RUN set -x \
    fix-permissions /home/$NB_USER
