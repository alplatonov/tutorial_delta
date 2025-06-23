#!/usr/bin/env python
"""
Delta Lake Data Processing Script

This script performs operations on data using Delta Lake and PySpark.
It can create Delta tables from CSV files, create summary views, 
perform merge operations, and query data.
"""

import os
import sys
import argparse
from pathlib import Path

import delta
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import asc

# Constants
DELTA_JAR_VERSION = "2.4.0"
DELTA_JAR_PATH = f"delta-core_2.12-{DELTA_JAR_VERSION}.jar"

# Define schema for the CSV data
EMPLOYEE_SCHEMA = StructType([
    StructField("id", IntegerType(), False),
    StructField("firstName", StringType(), True),
    StructField("middleName", StringType(), True),
    StructField("lastName", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("birthDate", StringType(), True),  # Parse as string first
    StructField("ssn", StringType(), True),
    StructField("salary", IntegerType(), True)
])

# Path configuration
BASE_FOLDER = Path(os.getcwd()) / "testfolder"
BASE_FOLDER.mkdir(exist_ok=True)

CSV_PATH = Path(os.getcwd()) / "export.csv"
DELTA_TABLE_PATH = BASE_FOLDER / "employees_delta"

def get_spark() -> SparkSession:
    """Initialize and return a Spark session configured for Delta Lake.

    Returns:
        SparkSession: Configured Spark session
    """
    builder = (SparkSession.builder
        .master("local[*]")
        .appName("DeltaLakeProcessor")
        .config("spark.jars", "/Users/alplatonov/git/tutorial_delta/acryl-spark-lineage-0.2.18-rc1.jar")
        .config("spark.jars.packages", "io.acryl:acryl-spark-lineage:0.2.18-rc1")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.extraListeners", "datahub.spark.DatahubSparkListener")
        .config("spark.datahub.metadata.dataset.materialize", "true")
        .config("spark.datahub.rest.server", "https://data-api-test.sysops.xyz")
        .config("spark.datahub.rest.token", "eyJhbGciOiJIUzI1NiJ9.eyJhY3RvclR5cGUiOiJVU0VSIiwiYWN0b3JJZCI6ImFsZWtzYW5kci5wbGF0b25vdiIsInR5cGUiOiJQRVJTT05BTCIsInZlcnNpb24iOiIyIiwianRpIjoiNGQ3YmYzYjAtY2I4Yy00ZjliLTk5YmQtZjdjMzRhZmU0MjFiIiwic3ViIjoiYWxla3NhbmRyLnBsYXRvbm92IiwiZXhwIjoxNzUyOTE4NTY1LCJpc3MiOiJkYXRhaHViLW1ldGFkYXRhLXNlcnZpY2UifQ.2j_FAiD30u2rQlvPDbbHd_g38i962smmdxk86imqpwQ")
    )
    # Initialize with Delta Lake support
    spark = delta.configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("INFO")
    return spark


def create_delta_table(spark: SparkSession) -> bool:
    """Create a Delta table from a CSV file.

    Args:
        spark: Active Spark session

    Returns:
        bool: Success status of the operation
    """
    try:
        # Check if the CSV file exists
        if not CSV_PATH.exists():
            print(f"Error: CSV file '{CSV_PATH}' not found.")
            return False

        # Read the CSV file into a DataFrame
        print(f"Reading CSV file: {CSV_PATH}")
        df = (spark.read
            .option("header", "true")
            .option("inferSchema", "false")
            .schema(EMPLOYEE_SCHEMA)
            .csv(str(CSV_PATH)))

        # Write the DataFrame as a Delta table
        print(f"\nCreating Delta table at: {DELTA_TABLE_PATH}")
        (df.write
            .format("delta")
            .mode("overwrite")
            .save(str(DELTA_TABLE_PATH)))

        # Display sample data
        df.show()

        print("Delta table created successfully. You can query this table using Spark SQL.")
        return True

    except Exception as e:
        print(f"Error creating Delta table: {e}")
        return False

def create_employees_summary_view(spark: SparkSession) -> bool:
    """Create a summary view of employees data grouped by gender with salary statistics.

    Args:
        spark: Active Spark session

    Returns:
        bool: Success status of the operation
    """
    try:
        print("Creating employees summary view...")
        employees_summary_path = BASE_FOLDER / "employees_summary"

        # Create a DataFrame with the summary data using SQL query
        summary_df = spark.sql(f"""
            SELECT
                gender,
                COUNT(*) AS employee_count,
                AVG(salary) AS avg_salary,
                MIN(salary) AS min_salary,
                MAX(salary) AS max_salary
            FROM delta.`{str(DELTA_TABLE_PATH)}`
            GROUP BY gender
        """)

        # Write the DataFrame to a Delta table
        (summary_df.write
            .format("delta")
            .mode("overwrite")
            .save(str(employees_summary_path)))

        # Show the results
        summary_df.show(truncate=False)

        print(f"Employees summary view created at: {employees_summary_path}")
        return True

    except Exception as e:
        print(f"Error creating employees summary view: {e}")
        return False

def merge_employees_data(spark: SparkSession) -> bool:
    """Merge employee summary data into a target table with column transformations.

    This function demonstrates a Delta Lake MERGE operation with column value swapping,
    showcasing how to handle both matched and not matched conditions.

    Args:
        spark: Active Spark session

    Returns:
        bool: Success status of the operation
    """
    try:
        print("Performing merge operation...")

        # Define paths
        employees_summary_path = BASE_FOLDER / "employees_summary"
        employees_merged_path = BASE_FOLDER / "employees_merged"

        # Create target table if it doesn't exist
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS delta.`{str(employees_merged_path)}`
            (
                gender STRING,
                employee_count LONG,
                avg_salary DOUBLE,
                min_salary INT,
                max_salary INT
            )
            USING delta
        """)

        # Merge data with column transformations (intentionally swapping values)
        merge_result = spark.sql(f"""
            MERGE INTO delta.`{str(employees_merged_path)}` as target
            USING delta.`{str(employees_summary_path)}` as source
            ON target.gender = source.gender
            WHEN MATCHED THEN
                UPDATE SET 
                    target.employee_count = source.max_salary,
                    target.avg_salary = source.min_salary,
                    target.min_salary = source.avg_salary,
                    target.max_salary = source.employee_count
            WHEN NOT MATCHED THEN
                INSERT (gender, employee_count, avg_salary, min_salary, max_salary) 
                VALUES (gender, max_salary, min_salary, avg_salary, employee_count)
        """)

        # Display the merge result summary
        merge_result.show(truncate=False)

        # Show the contents of the merged table
        print("\nMerged table contents:")
        spark.read.format("delta").load(str(employees_merged_path)).show(truncate=False)

        print(f"Merge operation completed successfully to: {employees_merged_path}")
        return True

    except Exception as e:
        print(f"Error performing merge: {e}")
        return False

def query_data(spark: SparkSession) -> bool:
    """Query the original delta table data grouped by birthDate.

    Args:
        spark: Active Spark session

    Returns:
        bool: Success status of the operation
    """
    try:
        print("\nQuerying original data by birthDate:")
        # Read the Delta table into a DataFrame
        df = spark.read.format("delta").load(str(DELTA_TABLE_PATH))

        # Group by birthDate, count records, and sort by birthDate
        result_df = df.groupBy("birthDate").count().orderBy(asc("birthDate"))

        # Display the results
        result_df.show(truncate=False)
        return True

    except Exception as e:
        print(f"Error querying data: {e}")
        return False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='CSV to Delta Table Converter with multiple operations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
        Examples:
          python csv_to_delta.py create             # Create Delta table from default CSV
          python csv_to_delta.py create --csv=data.csv  # Create using specific CSV file
          python csv_to_delta.py view              # Create summary view by gender
          python csv_to_delta.py merge             # Perform merge operation with transformations
          python csv_to_delta.py query             # Query data grouped by birthDate
        '''
    )

    # Add commands as arguments
    parser.add_argument(
        'command',
        choices=['create', 'view', 'merge', 'query'],
        help=('Command to execute: create (delta table), view (create summary view), '
              'merge (data), query (existing data)')
    )

    # Optional arguments
    parser.add_argument(
        '--csv',
        help='Path to CSV file (for create command)',
        default=str(CSV_PATH)
    )

    return parser.parse_args()



def main() -> None:
    """Main function to process command line arguments and execute requested operations.

    Parses arguments, initializes Spark, executes the requested command,
    handles exceptions, and ensures proper cleanup of resources.

    Returns:
        None
    """
    global CSV_PATH  # Declare global at the start of the function

    # Parse command-line arguments
    args = parse_arguments()

    # Initialize Spark
    spark = get_spark()
    success = False

    try:
        # Execute the requested command
        if args.command == 'create':
            print("Starting CSV to Delta Table conversion...")
            success = create_delta_table(spark)

        elif args.command == 'view':
            success = create_employees_summary_view(spark)

        elif args.command == 'merge':
            success = merge_employees_data(spark)

        elif args.command == 'query':
            success = query_data(spark)

        else:
            print(f"Unknown command: {args.command}")
            success = False

    except Exception as e:
        print(f"Error executing command {args.command}: {e}")
        success = False

    finally:
        # Stop the Spark session
        if spark:
            print("Stopping Spark session...")
            spark.stop()

        # Return appropriate exit code
        sys.exit(0 if success else 1)


# Example commented code - kept for reference
'''
Additional examples of Delta Lake operations:

# Create a view using SQL
spark.sql("""
    INSERT OVERWRITE TABLE delta.`/path/to/summary_table`
    SELECT
        db,
        CAST(created_at AS DATE) AS create_date,
        COUNT(*) AS cnt,
        SUM(amount_cents) AS amount_cents
    FROM delta.`/path/to/source_table`
    GROUP BY db, CAST(created_at AS DATE)
""").show(truncate=False)

# Merge data between tables with column transformations
spark.sql("""
    MERGE INTO delta.`/path/to/target_table` AS target
    USING delta.`/path/to/source_table` AS source
    ON target.key_column = source.key_column
    WHEN MATCHED THEN
        UPDATE SET target.value_a = source.value_b, target.value_b = source.value_a
    WHEN NOT MATCHED THEN
        INSERT (key_column, value_a, value_b) VALUES (key_column, value_b, value_a)
""").show(truncate=False)
'''


if __name__ == "__main__":
    main()

