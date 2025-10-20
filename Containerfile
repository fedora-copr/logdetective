FROM fedora:42
# Fedora's llama-cpp-python is segfaulting on the mistral model we use :/
RUN dnf install -y \
    fastapi-cli \
    python3-fastapi \
    python3-drain3 \
    python3-pip \
    python3-pydantic-settings \
    python3-starlette+full \
    gcc \
    gcc-c++ \
    python3-scikit-build \
    git-core python3-gunicorn \
    python3-gitlab \
    python3-diskcache \
    python3-sqlalchemy+asyncio \
    python3-alembic \
    python3-matplotlib \
    python3-aiohttp \
    python3-backoff \
    python3-sentry-sdk+fastapi \
    python3-koji \
    csdiff \
    python3-asyncpg \
    && dnf clean all
# the newest 0.2.86 fails to build, it seems vendored llama-cpp is missing in the archive
RUN pip3 install aiolimiter llama_cpp_python==0.2.85 sse-starlette starlette-context openai==1.82.1 \
    && mkdir /src

# uncomment below if you need to download the model, otherwise just bindmount your local
# models inside the container
# RUN pip3 install huggingface_hub[cli] \
#     && mkdir /models \
#     && huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir /models --local-dir-use-symlinks False

COPY ./logdetective/ /src/logdetective/
COPY ./alembic.ini /src/alembic.ini
COPY ./alembic /src/alembic
COPY ./files /src/files
COPY ./server /src/server
COPY ./pyproject.toml /src/pyproject.toml
COPY ./README.md /src/README.md

COPY ./files/Current-IT-Root-CAs.pem /etc/pki/ca-trust/source/anchors/
RUN update-ca-trust

WORKDIR /src

RUN pip3 install .
