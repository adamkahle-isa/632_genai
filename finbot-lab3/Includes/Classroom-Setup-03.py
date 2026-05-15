# Databricks notebook source
%pip install -qqq -U databricks-sdk databricks-langchain databricks-vectorsearch langchain langgraph mlflow>=3.1
dbutils.library.restartPython()

# COMMAND ----------

%run ./Classroom-Setup-Common

# COMMAND ----------

DA = DBAcademyHelper()
DA.init()
