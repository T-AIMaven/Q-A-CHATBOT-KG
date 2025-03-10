import logging
import bytewax.operators as op
from bytewax.dataflow import Dataflow
from data_flow.stream_output import MongoOutput, MemgraphOutput
from db.ddb import mongo_connection
from db.sdb import csv_source

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

flow = Dataflow("Streaming ingestion pipeline")

stream = op.input("input", flow, lambda: iter(csv_source))
op.output(
    "data is inserted to destination database - memgraph",
    stream,
    MemgraphOutput(uri="bolt://localhost:7687", auth=("", "")),
)
op.output(
    "data is inserted to destination database - mongodb",
    stream,
    MongoOutput(connection=mongo_connection),
)

