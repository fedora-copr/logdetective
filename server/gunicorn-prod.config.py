import os
bind = f"0.0.0.0:{os.environ['LOGDETECTIVE_SERVER_PORT']}"
worker_class = "uvicorn.workers.UvicornWorker"
workers = 2
# timeout set to 120 - 2 minutes should be enough for one LLM execution in production on a GPU
timeout = 120
# write to stdout
accesslog = '-'
certfile = "/src/server/cert.pem"
keyfile = "/src/server/privkey.pem"
ca_certs = "/src/server/fullchain.pem"
