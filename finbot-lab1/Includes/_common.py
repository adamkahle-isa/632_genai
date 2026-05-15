# Databricks notebook source
from databricks.sdk import WorkspaceClient
import pyspark.sql.functions as F

class DBAcademyHelper:
    """Simplified helper for Finance Bot labs. No meta table required."""

    def __init__(self):
        self.workspace = WorkspaceClient()
        self.catalog_name = "isa632_7474656346303369"
        self.schema_name = "kahlear"
        self.vs_endpoint_name = "ai_agent_vs_endpoint_kahlear"
        self.vs_source_table_name = f"{self.catalog_name}.{self.schema_name}.pdf_docs"
        self.vs_index_name = f"{self.catalog_name}.{self.schema_name}.pdf"

        spark.sql(f"USE CATALOG {self.catalog_name}")
        spark.sql(f"USE SCHEMA {self.schema_name}")

    def init(self):
        """No-op for compatibility."""
        pass

    def validate_table(self, name):
        if spark.catalog.tableExists(name):
            print(f"Validation of table {name} complete. No errors found.")
            return True
        else:
            raise AssertionError(f"The table {name} does not exist")

    def unique_name(self, sep: str = "_") -> str:
        return "kahlear"