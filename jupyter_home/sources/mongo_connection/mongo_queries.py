import pickle
from typing import List

import numpy as np
import pymongo.cursor
from bson.binary import Binary

from sources.mongo_connection.mongo_connector import MongoDBConnection


# ITERATIONS
def save_iteration(dataset_id: str, settings_id: str, cost: float, solution: List[int], precision: List[float],
                   n_calls: int, duration: str, violations: int, historic: np.array) -> int:

    db = MongoDBConnection.get_iterations_db()

    result = db.insert_one({
        "dataset_id": dataset_id,
        "settings_id": settings_id,
        "cost": cost,
        "solution": solution,
        "precision": precision,
        "n_calls": n_calls,
        "duration": duration,
        "violations": violations,
        "historic": Binary(pickle.dumps(historic, protocol=2), subtype=128)
    })
    return result.inserted_id


def find_iteration(dataset_id: str, settings_id: str) -> pymongo.cursor.Cursor:
    db = MongoDBConnection.get_iterations_db()
    return db.find({'dataset_id': dataset_id, 'settings_id': settings_id})


def count_iterations(dataset_id: str, settings_id: str) -> int:
    db = MongoDBConnection.get_iterations_db()
    return db.count({'dataset_id': dataset_id, 'settings_id': settings_id})


def remove_all_iterations():
    db = MongoDBConnection.get_iterations_db()
    db.remove()


def remove_iterations(dataset_id: str, settings_id: str):
    db = MongoDBConnection.get_iterations_db()
    db.remove({'dataset_id': dataset_id, 'settings_id': settings_id})
