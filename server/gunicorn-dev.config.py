import os

bind = f"0.0.0.0:{os.environ['LOGDETECTIVE_SERVER_PORT']}"
worker_class = "uvicorn.workers.UvicornWorker"
workers = 1
# timeout set to 240 - 4 minutes should be enough for one LLM execution locally
# on a CPU
timeout = 600
# write to stdout
accesslog = "-"
# certfile = "/src/server/cert.pem"
# keyfile = "/src/server/privkey.pem"
# ca_certs = "/src/server/fullchain.pem"
