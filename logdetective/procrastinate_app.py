"""Lightweight Procrastinate application shared by workers and schema setup."""

from procrastinate import App, PsycopgConnector

from logdetective.database.base import get_pg_conninfo

app = App(connector=PsycopgConnector(conninfo=get_pg_conninfo()))
