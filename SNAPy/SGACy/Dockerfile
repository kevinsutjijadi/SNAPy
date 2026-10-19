# Build SGACy Linux shared objects (graph/geom .so) for the requested CPython.
#   docker build -t sgacy-linux --build-arg PY=3.11 .
#   docker run --rm -v "${PWD}:/src" sgacy-linux
ARG PY=3.11
FROM python:${PY}-slim-bookworm
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "setuptools>=68" "cython>=3.0" "numpy>=1.23.5"
WORKDIR /src
CMD ["python", "setup.py", "build_ext", "--inplace"]
