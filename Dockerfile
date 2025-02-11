FROM docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

RUN echo "export PS1='\[\e]0;\u@\h: \w\a\]\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /root/.bashrc

# Install apt-get dependencies
RUN apt-get update \
 && apt-get install -y git curl wget python3 python3-dev python3-venv vim

# Install most recent version of Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /root/script.sh \
 && bash /root/script.sh -y \
 && rm -f /root/script.sh

# Set up local pip environment
RUN python3 -m venv /root/.venv \
 && source /root/.venv/bin/activate \
 && pip install maturin

RUN mkdir /src

RUN echo "source /root/.venv/bin/activate" >> /root/.bashrc

# Install NPBench
RUN source /root/.venv/bin/activate \
 && pip install dace cupy-cuda12x "jax[cuda12]" \
 && cd /src \
 && git clone https://github.com/larshum/npbench.git \
 && cd /src/npbench \
 && pip install -r requirements.txt \
 && pip install .

# Copy over this repo
COPY . /src/pyparir
WORKDIR /src/pyparir
