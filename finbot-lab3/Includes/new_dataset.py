# Databricks notebook source
%run ./Classroom-Setup-Common

# COMMAND ----------

DA = DBAcademyHelper()
DA.init()

# COMMAND ----------

# Validate news table exists
DA.validate_table(f"{DA.catalog_name}.{DA.schema_name}.news")
print(f"News source table: {DA.vs_source_table_name}")
print(f"VS Index: {DA.vs_index_name}")
