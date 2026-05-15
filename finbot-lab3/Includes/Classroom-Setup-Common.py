# Databricks notebook source
%run ./_common

# COMMAND ----------

import time
from pprint import pprint
from databricks.vector_search.client import VectorSearchClient

def is_vs_endpoint_ready(vs_endpoint_name):
    """Check if the vector search endpoint is online."""
    vsc = VectorSearchClient(disable_notice=True)
    for i in range(180):
        endpoint = vsc.get_endpoint(vs_endpoint_name)
        status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
        if "ONLINE" in status:
            print(f"Endpoint \'{vs_endpoint_name}\' is ready.")
            return True
        elif "PROVISIONING" in status or i < 6:
            if i % 20 == 0:
                print(f"Waiting for endpoint to be ready...")
            time.sleep(10)
        else:
            raise Exception(f"Error with endpoint {vs_endpoint_name}: {endpoint}")
    raise Exception(f"Timeout waiting for endpoint {vs_endpoint_name}")

def index_exists(vsc, endpoint_name, index_full_name):
    """Check if the vector search index exists."""
    try:
        dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
        return dict_vsindex.get("status").get("ready", False)
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" not in str(e) and "NOT_FOUND" not in str(e):
            print(f"Unexpected error describing the index.")
            raise e
    return False

print("Finance Bot classroom setup complete.")
