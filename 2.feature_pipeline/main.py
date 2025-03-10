import logging
import bytewax.operators as op
from bytewax.dataflow import Dataflow

from db.ddb import mongo_connection
from db.sdb import csv_source

from data_flow.stream_output import MongoOutput, QdrantOutput
from data_logic.dispatchers import (
    ChunkingDispatcher,
    # CleaningDispatcher, 
    EmbeddingDispatcher,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

flow = Dataflow("Streaming ingestion pipeline")

def log_stage(stage_name):
    def log(data):
        logger.info(f"{stage_name}: {data}")
        return data
    return log

stream = op.input("input", flow, lambda: iter(csv_source))
# stream = op.map("raw dispatch", stream, RawDispatcher.handle_mq_message)
# stream = op.map("clean dispatch", stream, CleaningDispatcher.dispatch_cleaner)
#####################
op.output(
    "cleaned data insert to qdrant",
    stream,
    QdrantOutput(connection=mongo_connection, sink_type="clean"),
)
stream = op.flat_map("chunk dispatch", stream, ChunkingDispatcher.dispatch_chunker)
stream = op.map(
    "embedded chunk dispatch", stream, EmbeddingDispatcher.dispatch_embedder
)
####################

op.output(
    "data is inserted to source database",
    stream,
    MongoOutput(connection=mongo_connection, sink_type="vector"),
)

