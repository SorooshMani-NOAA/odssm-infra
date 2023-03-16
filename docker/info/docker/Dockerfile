FROM continuumio/miniconda3:4.10.3-alpine

# Create a non-root user
ARG username=hurricaner
ARG uid=1000
ARG gid=100

ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER

# Get necessary packages
RUN apk update && apk upgrade && apk add \
    git

# New user
RUN adduser --disabled-password --gecos "Non-root user" --uid $UID --home $HOME $USER

# Create a project directory inside user home
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
RUN chown $UID:$GID $PROJECT_DIR
WORKDIR $PROJECT_DIR


# Build the conda environment
ENV ENV_PREFIX $HOME/icogsc

COPY environment.yml /tmp/
RUN chown $UID:$GID /tmp/environment.yml

RUN conda install mamba -n base -c conda-forge && \
    mamba update --name base --channel defaults conda && \
    mamba env create --prefix $ENV_PREFIX --file /tmp/environment.yml --force && \
    mamba clean --all --yes

RUN git clone https://github.com/oceanmodeling/stormevents && \
    git -C stormevents checkout 2ebcc72 && \
    conda run -p $ENV_PREFIX --no-capture-output \
        pip install ./stormevents && \
    rm -rf stormevents

ENV CONDA_DIR /opt/conda

RUN conda clean --all
RUN apk del git

RUN mkdir -p $PROJECT_DIR/scripts
COPY docker/hurricane_data.py ${PROJECT_DIR}/scripts/
ENV PYTHONPATH ${PROJECT_DIR}/scripts/


RUN mkdir -p $PROJECT_DIR/io

USER $USER


# Ref: https://pythonspeed.com/articles/activate-conda-dockerfile/
ENTRYPOINT [ "conda", "run", "-p", "$ENV_PREFIX", "--no-capture-output", "python", "-m", "hurricane_data" ]
