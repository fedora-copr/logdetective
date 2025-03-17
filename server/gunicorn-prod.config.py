import os
bind = f"0.0.0.0:{os.environ['LOGDETECTIVE_SERVER_PORT']}"
worker_class = "uvicorn.workers.UvicornWorker"
workers = 2
# timeout set to 600 seconds; with 32 clusters and several runs in parallel, it
# can take even 10 minutes for a query to complete
timeout = 600
# write to stdout
accesslog = '-'
# certfile = "/src/server/cert.pem"
# keyfile = "/src/server/privkey.pem"
# ca_certs = "/src/server/fullchain.pem"
