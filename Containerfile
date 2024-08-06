FROM fedora:40
RUN dnf install -y fastapi-cli python3-fastapi python3-requests python3-drain3 python3-pip python3-pydantic-settings python3-starlette+full \
    && pip3 install sse-starlette starlette-context huggingface_hub[cli] \
    && mkdir /src

# we need to bind mount models: this takes a lot of time to download and makes the image huge
RUN mkdir /models \
    && huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir /models --local-dir-use-symlinks False

# Fedora's llama-cpp-python is segfaulting on the mistral model above :/
RUN dnf install -y gcc gcc-c++ python3-scikit-build \
    && pip3 install -U llama_cpp_python

COPY ./logdetective/ /src/logdetective/

# --no-reload: doesn't work in a container - `PermissionError: Permission denied (os error 13) about ["/proc"]`
CMD ["fastapi", "dev", "/src/logdetective/server.py", "--host", "0.0.0.0", "--port", "8080", "--no-reload"]
