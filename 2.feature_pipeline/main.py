import logging
import bytewax.operators as op
from bytewax.dataflow import Dataflow
from data_flow.stream_output import MongoOutput, MemgraphOutput
from utils.dest_db import mongo_connection
from utils.source_db import csv_source
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

flow = Dataflow("Streaming ingestion pipeline")

stream = op.input("input", flow, lambda: iter(csv_source))
op.output(
    "data is inserted to destination database - memgraph",
    stream,
    MemgraphOutput(uri=settings.MEMGRAPH_URI, auth=("", "")),
)
op.output(
    "data is inserted to destination database - mongodb",
    stream,
    MongoOutput(connection=mongo_connection),
)

