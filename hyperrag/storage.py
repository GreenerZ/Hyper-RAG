import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import httpx
import numpy as np
from hyperdb import HypergraphDB
from nano_vectordb import NanoVectorDB

from .base import (
    BaseHypergraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)
from .prompt import GRAPH_FIELD_SEP
from .utils import load_json, logger, write_json


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()


@dataclass
class HypergraphStorage(BaseHypergraphStorage):
    backend: str = "hypergraph-db"

    def __post_init__(self):
        self._driver: HypergraphDriver = get_hypergraph_driver(self.backend)
        self._hgdb_file = None
        if self._driver.requires_local_file:
            self._hgdb_file = os.path.join(
                self.global_config["working_dir"], f"hypergraph_{self.namespace}.hgdb"
            )
        self._hg = self._driver.load_or_create(self._hgdb_file or "")

    async def index_done_callback(self):
        logger.info(
            f"Writing hypergraph with {await self.get_num_of_vertices()} vertices, {await self.get_num_of_hyperedges()} hyperedges"
        )
        if self._driver.requires_local_file and self._hgdb_file:
            self._driver.save(self._hg, self._hgdb_file)

    async def has_vertex(self, v_id: Any) -> bool:
        return self._hg.has_v(v_id)

    async def has_hyperedge(self, e_tuple: Union[List, Set, Tuple]) -> bool:
        return self._hg.has_e(e_tuple)

    async def get_vertex(self, v_id: str, default: Any = None) :
        return self._hg.v(v_id)

    async def get_hyperedge(self, e_tuple: Union[List, Set, Tuple], default: Any = None) :
        return self._hg.e(e_tuple)

    async def get_all_vertices(self):
        return self._hg.all_v

    async def get_all_hyperedges(self):
        return self._hg.all_e

    async def get_num_of_vertices(self):
        return self._hg.num_v

    async def get_num_of_hyperedges(self):
        return self._hg.num_e

    async def upsert_vertex(self, v_id: Any, v_data: Optional[Dict] = None) :
        return self._hg.add_v(v_id, v_data)

    async def upsert_hyperedge(self, e_tuple: Union[List, Set, Tuple], e_data: Optional[Dict] = None) :
        return self._hg.add_e(e_tuple, e_data)

    async def remove_vertex(self, v_id: Any) :
        return self._hg.remove_v(v_id)

    async def remove_hyperedge(self, e_tuple: Union[List, Set, Tuple]) :
        return self._hg.remove_e(e_tuple)

    async def vertex_degree(self, v_id: Any) -> int:
        return self._hg.degree_v(v_id)

    async def hyperedge_degree(self, e_tuple: Union[List, Set, Tuple]) -> int:
        return self._hg.degree_e(e_tuple)

    async def get_nbr_e_of_vertex(self, e_tuple: Union[List, Set, Tuple]) -> list:
        """
            Return the incident hyperedges of the vertex.
        """
        return self._hg.nbr_e_of_v(e_tuple)

    async def get_nbr_v_of_hyperedge(self, v_id: Any, exclude_self=True) -> list:
        """
            Return the incident vertices of the hyperedge.
        """
        return self._hg.nbr_v_of_e(v_id)

    async def get_nbr_v_of_vertex(self, v_id: Any, exclude_self=True) -> list:
        """
            Return the neighbors of the vertex.
        """
        return self._hg.nbr_v(v_id)


@dataclass
class BaseTuGraphStorage(BaseHypergraphStorage):
    """Shared TuGraph storage adapter logic for Cypher-based backends."""

    def __post_init__(self):
        self.server_url = self.global_config.get("tugraph_server_url", "http://localhost:7071")
        self.graph_name = self.global_config.get("tugraph_graph_name", "default")
        self.username = self.global_config.get("tugraph_user", "admin")
        self.password = self.global_config.get("tugraph_password", "73@TuGraph")
        self.auto_create_schema = self.global_config.get("tugraph_auto_create_schema", True)

        self._setup_client()

        if self.auto_create_schema:
            try:
                self._ensure_schema()
            except Exception as exc:  # pragma: no cover - best effort for remote service
                logger.warning(f"Failed to ensure TuGraph schema: {exc}")

    # ------------------------------------------------------------------
    # hooks implemented by concrete backends
    # ------------------------------------------------------------------
    def _setup_client(self):
        raise NotImplementedError

    def _run_cypher_sync(self, script: str, parameters: Optional[dict] = None):
        raise NotImplementedError

    async def _run_cypher(self, script: str, parameters: Optional[dict] = None):
        return await asyncio.to_thread(self._run_cypher_sync, script, parameters)

    def _ensure_schema(self):
        # Constraints are idempotent in Cypher when using IF NOT EXISTS
        schema_statements = [
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_name IS UNIQUE",
            "CREATE CONSTRAINT hyperedge_id_unique IF NOT EXISTS FOR (h:HyperEdge) REQUIRE h.id_set IS UNIQUE",
        ]
        for statement in schema_statements:
            try:
                self._run_cypher_sync(statement)
            except Exception as exc:  # pragma: no cover - best effort for remote service
                logger.debug(f"Schema statement failed (ignored): {exc}")

    @staticmethod
    def _normalize_hyperedge_key(e_tuple: Union[List, Set, Tuple]) -> Tuple[str, ...]:
        return tuple(sorted([str(x) for x in e_tuple]))

    @staticmethod
    def _hyperedge_id_set(e_tuple: Union[List, Set, Tuple]) -> str:
        return GRAPH_FIELD_SEP.join(BaseTuGraphStorage._normalize_hyperedge_key(e_tuple))

    # ------------------------------------------------------------------
    # Vertex helpers
    # ------------------------------------------------------------------
    async def has_vertex(self, v_id: Any) -> bool:
        script = "MATCH (e:Entity {entity_name: $id}) RETURN count(e) > 0"
        result = await self._run_cypher(script, {"id": v_id})
        data = result.get("data") or result.get("results", [{}])[0].get("data", [])
        return bool(data and (data[0][0] if isinstance(data[0], list) else data[0]))

    async def get_vertex(self, v_id: str, default: Any = None):
        script = "MATCH (e:Entity {entity_name: $id}) RETURN e"
        result = await self._run_cypher(script, {"id": v_id})
        data = result.get("data") or result.get("results", [{}])[0].get("data", [])
        if not data:
            return default
        node = data[0][0] if isinstance(data[0], list) else data[0]
        return node.get("properties", node)

    async def upsert_vertex(self, v_id: Any, v_data: Optional[Dict] = None):
        v_data = v_data or {}
        script = (
            "MERGE (e:Entity {entity_name: $id}) "
            "SET e += $payload RETURN e"
        )
        payload = {k: v for k, v in v_data.items()}
        await self._run_cypher(script, {"id": v_id, "payload": payload})
        return v_data

    async def remove_vertex(self, v_id: Any):
        script = "MATCH (e:Entity {entity_name: $id}) DETACH DELETE e"
        await self._run_cypher(script, {"id": v_id})

    async def get_all_vertices(self):
        script = "MATCH (e:Entity) RETURN e"
        result = await self._run_cypher(script)
        rows = result.get("data") or result.get("results", [{}])[0].get("data", [])
        return [
            (row[0] if isinstance(row, list) else row).get("properties", row[0] if isinstance(row, list) else row)
            for row in rows
        ]

    async def get_num_of_vertices(self):
        script = "MATCH (e:Entity) RETURN count(e)"
        result = await self._run_cypher(script)
        data = result.get("data") or result.get("results", [{}])[0].get("data", [])
        return int(data[0][0] if isinstance(data[0], list) else data[0]) if data else 0

    async def vertex_degree(self, v_id: Any) -> int:
        script = (
            "MATCH (e:Entity {entity_name: $id})<-[:CONNECTS]-(h:HyperEdge) "
            "RETURN count(h)"
        )
        result = await self._run_cypher(script, {"id": v_id})
        data = result.get("data") or result.get("results", [{}])[0].get("data", [])
        return int(data[0][0] if isinstance(data[0], list) else data[0]) if data else 0

    # ------------------------------------------------------------------
    # Hyperedge helpers
    # ------------------------------------------------------------------
    async def has_hyperedge(self, e_tuple: Union[List, Set, Tuple]) -> bool:
        id_set = self._hyperedge_id_set(e_tuple)
        script = "MATCH (h:HyperEdge {id_set: $id_set}) RETURN count(h) > 0"
        result = await self._run_cypher(script, {"id_set": id_set})
        data = result.get("data") or result.get("results", [{}])[0].get("data", [])
        return bool(data and (data[0][0] if isinstance(data[0], list) else data[0]))

    async def get_hyperedge(self, e_tuple: Union[List, Set, Tuple], default: Any = None):
        id_set = self._hyperedge_id_set(e_tuple)
        script = "MATCH (h:HyperEdge {id_set: $id_set}) RETURN h"
        result = await self._run_cypher(script, {"id_set": id_set})
        data = result.get("data") or result.get("results", [{}])[0].get("data", [])
        if not data:
            return default
        hyperedge = data[0][0] if isinstance(data[0], list) else data[0]
        return hyperedge.get("properties", hyperedge)

    async def upsert_hyperedge(self, e_tuple: Union[List, Set, Tuple], e_data: Optional[Dict] = None):
        e_data = e_data or {}
        normalized = self._normalize_hyperedge_key(e_tuple)
        id_set = self._hyperedge_id_set(normalized)
        payload = {**e_data, "id_set": id_set}

        # ensure entity nodes exist and link them to the hyperedge node
        entities_payload = [{"entity_name": ent} for ent in normalized]
        await asyncio.gather(*[
            self.upsert_vertex(ent["entity_name"], {}) for ent in entities_payload
        ])

        script = (
            "MERGE (h:HyperEdge {id_set: $id_set}) "
            "SET h += $payload "
            "WITH h UNWIND $entities AS ent "
            "MERGE (e:Entity {entity_name: ent.entity_name}) "
            "MERGE (h)-[r:CONNECTS]->(e) "
            "SET r.weight = $weight, r.source_id = $source_id, r.description = $description, r.keywords = $keywords "
            "RETURN h"
        )

        await self._run_cypher(
            script,
            {
                "id_set": id_set,
                "payload": payload,
                "entities": entities_payload,
                "weight": e_data.get("weight", 0),
                "source_id": e_data.get("source_id", ""),
                "description": e_data.get("description", ""),
                "keywords": e_data.get("keywords", ""),
            },
        )
        return payload

    async def remove_hyperedge(self, e_tuple: Union[List, Set, Tuple]):
        id_set = self._hyperedge_id_set(e_tuple)
        script = "MATCH (h:HyperEdge {id_set: $id_set}) DETACH DELETE h"
        await self._run_cypher(script, {"id_set": id_set})

    async def get_all_hyperedges(self):
        script = "MATCH (h:HyperEdge) RETURN h"
        result = await self._run_cypher(script)
        rows = result.get("data") or result.get("results", [{}])[0].get("data", [])
        return [
            (row[0] if isinstance(row, list) else row).get("properties", row[0] if isinstance(row, list) else row)
            for row in rows
        ]

    async def get_num_of_hyperedges(self):
        script = "MATCH (h:HyperEdge) RETURN count(h)"
        result = await self._run_cypher(script)
        data = result.get("data") or result.get("results", [{}])[0].get("data", [])
        return int(data[0][0] if isinstance(data[0], list) else data[0]) if data else 0

    async def hyperedge_degree(self, e_tuple: Union[List, Set, Tuple]) -> int:
        id_set = self._hyperedge_id_set(e_tuple)
        script = (
            "MATCH (h:HyperEdge {id_set: $id_set})-[r:CONNECTS]->(e:Entity) "
            "RETURN count(r)"
        )
        result = await self._run_cypher(script, {"id_set": id_set})
        data = result.get("data") or result.get("results", [{}])[0].get("data", [])
        return int(data[0][0] if isinstance(data[0], list) else data[0]) if data else 0

    async def get_nbr_e_of_vertex(self, e_tuple: Union[List, Set, Tuple]) -> list:
        script = (
            "MATCH (e:Entity {entity_name: $id})<-[:CONNECTS]-(h:HyperEdge) "
            "RETURN h.id_set"
        )
        result = await self._run_cypher(script, {"id": e_tuple})
        rows = result.get("data") or result.get("results", [{}])[0].get("data", [])
        return [row[0] if isinstance(row, list) else row for row in rows]

    async def get_nbr_v_of_hyperedge(self, v_id: Any, exclude_self=True) -> list:
        id_set = self._hyperedge_id_set(v_id if isinstance(v_id, (list, set, tuple)) else [v_id])
        script = (
            "MATCH (h:HyperEdge {id_set: $id_set})-[:CONNECTS]->(e:Entity) "
            "RETURN e.entity_name"
        )
        result = await self._run_cypher(script, {"id_set": id_set})
        rows = result.get("data") or result.get("results", [{}])[0].get("data", [])
        return [row[0] if isinstance(row, list) else row for row in rows]

    async def get_nbr_v_of_vertex(self, v_id: Any, exclude_self=True) -> list:
        script = (
            "MATCH (e:Entity {entity_name: $id})<-[:CONNECTS]-(h:HyperEdge)-[:CONNECTS]->(n:Entity) "
            "WHERE $exclude_self = false OR n.entity_name <> $id "
            "RETURN DISTINCT n.entity_name"
        )
        result = await self._run_cypher(script, {"id": v_id, "exclude_self": exclude_self})
        rows = result.get("data") or result.get("results", [{}])[0].get("data", [])
        return [row[0] if isinstance(row, list) else row for row in rows]

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    async def index_done_callback(self):
        # TuGraph persists data immediately; nothing to do.
        return None

    async def query_done_callback(self):
        return None


@dataclass
class TuGraphStorage(BaseTuGraphStorage):
    """TuGraph REST adapter using ``httpx``."""

    def _setup_client(self):
        self._session = httpx.Client(auth=(self.username, self.password))
        self._cypher_endpoint = f"{self.server_url.rstrip('/')}/cypher"

    def _run_cypher_sync(self, script: str, parameters: Optional[dict] = None):
        payload = {
            "graph": self.graph_name,
            "script": script,
            "parameters": parameters or {},
        }
        response = self._session.post(self._cypher_endpoint, json=payload)
        response.raise_for_status()
        return response.json()

