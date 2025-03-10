from bytewax.outputs import DynamicSink, StatelessSinkPartition
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid
from utils.mongo_data_load import MongoDataLoadConnector
import logging

logger = logging.getLogger(__name__)

mongo_dataload_connector = MongoDataLoadConnector()
class MongoOutput(DynamicSink):
    """
    Bytewax class that facilitates the connection to a MongoDB.
    Inherits DynamicSink because of the ability to create different sink sources (e.g., vector and non-vector collections)
    """

    def __init__(self, connection: MongoClient, sink_type: str):
        self._connection = connection
        self._sink_type = sink_type

        # try:
        #     self._connection["iGenius"].create_collection("cleaned_TicTokProfiles")
        # except CollectionInvalid:
        #     logger.info("Collection 'cleaned_TicTokProfiles' already exists.")

        try:
            self._connection["iGenius"].create_collection("knowledge_base")
        except CollectionInvalid:
            logger.info("Collection 'knowledge_base' already exists.")

    def build(self, worker_index: int, worker_count: int) -> StatelessSinkPartition:
        if self._sink_type == "clean":
            return MongoCleanedDataSink(connection=self._connection)
        elif self._sink_type == "vector":
            return MongoKGDataSink(connection=self._connection)
        else:
            raise ValueError(f"Unsupported sink type: {self._sink_type}")


class MongoCleanedDataSink(StatelessSinkPartition):
    def __init__(self, connection: MongoClient):
        self._client = connection

    def write_batch(self, items: list[dict]) -> None:
        collection = self._client["mydatabase"]["cleaned_TicTokProfiles"]
        collection.insert_many(items)
        logger.info("Successfully inserted cleaned data points", num=len(items))


class MongoKGDataSink(StatelessSinkPartition):
    def __init__(self, connection: MongoClient):
        self._client = connection

    def write_batch(self, items: list[dict]) -> None:
        collection = self._client["iGenius"]["knowledge_base"]
        index = mongo_dataload_connector.create_new_index(items, collection)
        logger.info("Successfully created knowlege graph - {index}")
