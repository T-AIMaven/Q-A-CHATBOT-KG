from bytewax.outputs import DynamicSink, StatelessSinkPartition
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid
from db.mongo import MongoDataLoadConnector
from db.memgraph import MemgraphDataLoadConnector
from neo4j import GraphDatabase
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
        try:
            self._connection["iGenius"].create_collection("knowledge_base")
        except CollectionInvalid:
            logger.info("Collection 'knowledge_base' already exists.")

    def build(self, worker_index: int, worker_count: int) -> StatelessSinkPartition:
        return MongoKGDataSink(connection=self._connection)
    

class MongoKGDataSink(StatelessSinkPartition):
    def __init__(self, connection: MongoClient):
        self._client = connection

    def write_batch(self, items: list[dict]) -> None:
        collection = self._client["iGenius"]["knowledge_base"]
        index = mongo_dataload_connector.create_new_index(items, collection)
        logger.info("Successfully created knowlege graph - {index}")


class MemgraphOutput(DynamicSink):
    """
    Bytewax class that facilitates the connection to Memgraph.
    Inherits DynamicSink because of the ability to create different sink sources (e.g., vector and non-vector collections)
    """
    def __init__(self, uri: str, auth: tuple, sink_type: str):
        self._uri = uri
        self._auth = auth
        self._sink_type = sink_type
        self._driver = GraphDatabase.driver(uri, auth=auth)

    def build(self, worker_index: int, worker_count: int) -> StatelessSinkPartition:
        return MemgraphKGDataSink(driver=self._driver)

    def close(self):
        self._driver.close()


class MemgraphKGDataSink(StatelessSinkPartition):
    def __init__(self, driver: GraphDatabase.driver):
        self._driver = driver

    def write_batch(self, items: list[dict]) -> None:
        index = MemgraphDataLoadConnector.import_csv_to_graph(items, self._driver)
        logger.info("Successfully created knowledge graph")
