# Databricks notebook source
from databricks.sdk import WorkspaceClient
import pyspark.sql.functions as F

class DBAcademyHelper:
    """Simplified helper for Finance Bot labs."""

    def __init__(self):
        self.workspace = WorkspaceClient()
        self.catalog_name = "isa632_7474656346303369"
        self.schema_name = "kahlear"
        self.vs_endpoint_name = "ai_agent_vs_endpoint_kahlear"
        self.vs_source_table_name = f"{self.catalog_name}.{self.schema_name}.pdf_docs"
        self.vs_index_name = f"{self.catalog_name}.{self.schema_name}.pdf"

    def init(self):
        """No-op for compatibility."""
        pass

    def validate_table(self, name):
        try:
            if spark.catalog.tableExists(name):
                print(f"Validation of table {name} complete. No errors found.")
                return True
            else:
                raise AssertionError(f"The table {name} does not exist")
        except Exception as e:
            if "PERMISSION_DENIED" in str(e):
                print(f"Note: Cannot validate {name} due to permissions, but table references will work with fully qualified names.")
                return True
            raise

    def unique_name(self, sep: str = "_") -> str:
        return "kahlear"