# Start from a core stack version
FROM jupyter/datascience-notebook

# install pygraphviz 
USER root
RUN apt-get update && apt-get install -y \
  graphviz \
  libgraphviz-dev \
  graphviz-dev \
  pkg-config
USER jovyan

# Install in the default python3 environment
RUN pip install pomegranate jupyterthemes pygraphviz

# enable dark themee
RUN jt -t chesterish
