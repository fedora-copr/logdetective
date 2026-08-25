FROM fedora:42
# Fedora's llama-cpp-python is segfaulting on the model we use :/
# python3-alembic is needed for the alembic-3
RUN dnf install -y \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    git-core \
    csdiff \
    krb5-devel \
    && dnf clean all

RUN mkdir /src

COPY ./logdetective/ /src/logdetective/
COPY ./alembic.ini /src/alembic.ini
COPY ./alembic /src/alembic
COPY ./files /src/files
COPY ./server /src/server
COPY ./pyproject.toml /src/pyproject.toml
COPY ./README.md /src/README.md
COPY ./LICENSE /src/LICENSE

COPY ./files/Current-IT-Root-CAs.pem /etc/pki/ca-trust/source/anchors/
RUN update-ca-trust

WORKDIR /src

RUN pip3 install .[server]
