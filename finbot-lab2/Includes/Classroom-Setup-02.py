# Databricks notebook source
%pip install -qqq -U databricks-sdk databricks-langchain databricks-vectorsearch langchain langgraph mlflow>=3.1
dbutils.library.restartPython()

# COMMAND ----------

%run ./new_dataset

# COMMAND ----------

DA = DBAcademyHelper()
DA.init()
