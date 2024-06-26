from typing import Dict, List
from logger import logger
import json


def create_json(file_json, annotations):
    logger.info(f"Create new {file_json}")
    try:
        with open(file_json, "w") as outfile:
            json.dump(annotations, outfile)
    except FileNotFoundError as e:
        logger.error(e)


def append_json(file_json, annotations):
    data = read_json(file_json)
    data.extend(annotations)
    try:
        with open(file_json, "w") as outfile:
            json.dump(data, outfile)
    except FileNotFoundError as e:
        logger.error(e)

    logger.info(f"{file_json} updated")


def read_json(file_json) -> List[Dict]:
    try:
        with open(file_json, "r") as infile:
            data = json.load(infile)
    except FileNotFoundError as e:
        logger.error(e)
    return data
