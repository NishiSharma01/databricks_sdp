# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # **Shuffle Reduction & collect() vs take()**
# MAGIC
# MAGIC This notebook demonstrates:
# MAGIC
# MAGIC ## Part 1: Shuffle Reduction Techniques
# MAGIC 1. **Filter Early** - Reduce data before expensive operations
# MAGIC 2. **Coalesce vs Repartition** - When to use each for reducing partitions
# MAGIC 3. **Partitioning Strategies** - Optimizing data distribution
# MAGIC 4. **Broadcast Joins** - Avoiding shuffle for small tables
# MAGIC 5. **Efficient Aggregations** - Using proper aggregation functions
# MAGIC
# MAGIC ## Part 2: collect() vs take()
# MAGIC - Memory implications
# MAGIC - Performance differences
# MAGIC - When to use each
# MAGIC - Best practices
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Filter Early Section
# MAGIC %md
# MAGIC ## **1. Filter Early - Reduce Data Before Shuffle**
# MAGIC
# MAGIC **Principle:** Apply filters as early as possible to reduce the amount of data being shuffled.
# MAGIC
# MAGIC **Why it matters:**
# MAGIC - Fewer rows = less data to shuffle across the network
# MAGIC - Reduces memory usage on executors
# MAGIC - Improves overall query performance
# MAGIC
# MAGIC **Rule:** Always filter BEFORE joins, aggregations, or window functions.

# COMMAND ----------

# DBTITLE 1,Create sample data for filter demo
# Create sample data
from pyspark.sql.functions import col, rand, when
import time

print("Creating sample sales dataset (10M rows)...")

# 10 million sales transactions
sales_df = spark.range(0, 10_000_000).select(
    col("id").alias("transaction_id"),
    (col("id") % 1000).alias("product_id"),
    (col("id") % 50).alias("region_id"),
    (rand() * 1000).cast("int").alias("amount"),
    when(col("id") % 10 < 7, "completed").otherwise("pending").alias("status")
)

print(f"Total transactions: {sales_df.count():,}")
print("\nSample data:")
display(sales_df.limit(5))

# COMMAND ----------

# DBTITLE 1,BAD - Filter AFTER aggregation
# ❌ BAD: Filter AFTER expensive operations
from pyspark.sql.functions import sum as spark_sum, count
import time

print("❌ BAD Practice: Filter AFTER aggregation")
print("Aggregates ALL data, then filters\n")

start = time.time()
result_filter_late = sales_df \
    .groupBy("region_id") \
    .agg(
        spark_sum("amount").alias("total_sales"),
        count("*").alias("transaction_count")
    ) \
    .filter(col("region_id") < 10)  # Filter AFTER aggregation

count_late = result_filter_late.count()
time_late = time.time() - start

print(f"Results: {count_late} regions")
print(f"Execution time: {time_late:.4f} seconds")
print("\n⚠️ Problem: Shuffles and aggregates ALL 10M rows")
print("   Then filters down to only 10 regions")

display(result_filter_late)

# COMMAND ----------

# DBTITLE 1,GOOD - Filter BEFORE aggregation
# ✅ GOOD: Filter BEFORE expensive operations
import time

print("✅ GOOD Practice: Filter BEFORE aggregation")
print("Filters data first, then aggregates\n")

start = time.time()
result_filter_early = sales_df \
    .filter(col("region_id") < 10) \
    .groupBy("region_id") \
    .agg(
        spark_sum("amount").alias("total_sales"),
        count("*").alias("transaction_count")
    )

count_early = result_filter_early.count()
time_early = time.time() - start

print(f"Results: {count_early} regions")
print(f"Execution time: {time_early:.4f} seconds")
print("\n✅ Advantage: Only shuffles ~2M rows (regions 0-9)")
print("   80% reduction in shuffle data!")

display(result_filter_early)

# COMMAND ----------

# DBTITLE 1,Filter early vs late comparison
# Performance comparison
print("=" * 60)
print("FILTER EARLY vs FILTER LATE")
print("=" * 60)

print(f"\n❌ Filter AFTER aggregation:  {time_late:.4f} seconds")
print(f"✅ Filter BEFORE aggregation: {time_early:.4f} seconds")

if time_late > time_early:
    improvement = ((time_late - time_early) / time_late) * 100
    speedup = time_late / time_early
    print(f"\n🚀 Performance Improvement: {improvement:.2f}%")
    print(f"   Speedup: {speedup:.2f}x faster")

print("\n💡 Key Takeaway: Filter early to reduce shuffle data!")

# COMMAND ----------

# DBTITLE 1,Coalesce vs Repartition Section
# MAGIC %md
# MAGIC ## **2. Coalesce vs Repartition**
# MAGIC
# MAGIC ### Key Differences:
# MAGIC
# MAGIC | Aspect | coalesce() | repartition() |
# MAGIC |--------|------------|---------------|
# MAGIC | **Shuffle** | No full shuffle (minimizes data movement) | Full shuffle |
# MAGIC | **Use case** | Reducing partitions | Increasing OR reducing partitions |
# MAGIC | **Performance** | Faster (less data movement) | Slower (full shuffle) |
# MAGIC | **Data distribution** | May be uneven | Even distribution |
# MAGIC | **Best for** | Writing fewer files, reducing parallelism | Rebalancing skewed data |
# MAGIC
# MAGIC ### When to Use:
# MAGIC - **coalesce()**: Reduce partitions after filtering (writing output files)
# MAGIC - **repartition()**: Fix data skew, increase parallelism, partition by column

# COMMAND ----------

# DBTITLE 1,Setup for coalesce vs repartition
# Create a filtered dataset with many partitions
from pyspark.sql.functions import col

print("Creating filtered dataset...\n")

# Filter to 10% of data, but still has many partitions
filtered_df = sales_df.filter(col("region_id") < 5).cache()

print(f"Filtered rows: {filtered_df.count():,}")
print("\nScenario: Want to reduce partitions for writing")

# COMMAND ----------

# DBTITLE 1,Using coalesce (no full shuffle)
# Method 1: coalesce() - minimizes data movement
import time

print("✅ Method 1: coalesce(4)")
print("Combines partitions without full shuffle\n")

start = time.time()
coalesced_df = filtered_df.coalesce(4)
count_coalesce = coalesced_df.count()
time_coalesce = time.time() - start

print(f"Result rows: {count_coalesce:,}")
print(f"Reduced to 4 partitions")
print(f"Execution time: {time_coalesce:.4f} seconds")
print("\n✅ Advantage: No full shuffle - just combines existing partitions")
print("   Fast and efficient for reducing partition count")

# COMMAND ----------

# DBTITLE 1,Using repartition (full shuffle)
# Method 2: repartition() - full shuffle
import time

print("⚠️ Method 2: repartition(4)")
print("Performs full shuffle for even distribution\n")

start = time.time()
repartitioned_df = filtered_df.repartition(4)
count_repartition = repartitioned_df.count()
time_repartition = time.time() - start

print(f"Result rows: {count_repartition:,}")
print(f"Repartitioned to 4 partitions")
print(f"Execution time: {time_repartition:.4f} seconds")
print("\n⚠️ Trade-off: Full shuffle but guarantees even distribution")
print("   Use when data skew is a problem")

# COMMAND ----------

# DBTITLE 1,Coalesce vs Repartition comparison
# Performance comparison
print("=" * 60)
print("COALESCE vs REPARTITION")
print("=" * 60)

print(f"\n✅ coalesce(4):    {time_coalesce:.4f} seconds")
print(f"⚠️ repartition(4): {time_repartition:.4f} seconds")

if time_repartition > time_coalesce:
    improvement = ((time_repartition - time_coalesce) / time_repartition) * 100
    speedup = time_repartition / time_coalesce
    print(f"\n🚀 coalesce is {improvement:.2f}% faster")
    print(f"   Speedup: {speedup:.2f}x")

print("\n" + "=" * 60)
print("DECISION GUIDE")
print("=" * 60)
print("\n✅ Use coalesce() when:")
print("   - Reducing partition count")
print("   - Writing fewer output files")
print("   - Data is already reasonably balanced")
print("\n⚠️ Use repartition() when:")
print("   - Increasing partition count")
print("   - Fixing data skew")
print("   - Need even distribution for downstream operations")
print("   - Partitioning by column: .repartition('column')")

# COMMAND ----------

# DBTITLE 1,Collect vs Take Section
# MAGIC %md
# MAGIC ## **3. collect() vs take()**
# MAGIC
# MAGIC ### Key Differences:
# MAGIC
# MAGIC | Aspect | collect() | take(n) |
# MAGIC |--------|-----------|----------|
# MAGIC | **Returns** | ALL rows | First n rows |
# MAGIC | **Memory** | Can cause OOM on large datasets | Safe - limited rows |
# MAGIC | **Performance** | Processes entire dataset | Stops after n rows |
# MAGIC | **Network** | Sends all data to driver | Sends only n rows |
# MAGIC | **Use case** | Small result sets | Preview, sampling, debugging |
# MAGIC
# MAGIC ### Memory Implications:
# MAGIC - **collect()**: Brings ALL data to driver memory - dangerous for large datasets!
# MAGIC - **take(n)**: Only brings n rows - safe for exploration
# MAGIC
# MAGIC ### When to Use:
# MAGIC - **collect()**: When you know the result is small (< 1000 rows)
# MAGIC - **take()**: Previewing data, debugging, sampling

# COMMAND ----------

# DBTITLE 1,Demonstrate collect() danger
# Create a moderately sized dataset for demonstration
from pyspark.sql.functions import col, rand
import time

print("Creating dataset (1M rows for demo)...\n")

demo_df = spark.range(0, 1_000_000).select(
    col("id"),
    (rand() * 100).alias("value")
)

print(f"Dataset size: {demo_df.count():,} rows")
print(f"Estimated memory if collected: ~{demo_df.count() * 16 / 1024 / 1024:.2f} MB")
print("\n⚠️ For large datasets (10M+ rows), collect() can crash the driver!")

# COMMAND ----------

# DBTITLE 1,Using take() - safe and fast
# Method 1: take() - safe and efficient
import time
import sys

print("✅ Method 1: take(10)")
print("Only retrieves 10 rows\n")

start = time.time()
result_take = demo_df.take(10)
time_take = time.time() - start

print(f"Rows retrieved: {len(result_take)}")
print(f"Memory usage: ~{sys.getsizeof(result_take) / 1024:.2f} KB")
print(f"Execution time: {time_take:.4f} seconds")
print("\nFirst 5 rows:")
for row in result_take[:5]:
    print(f"  {row}")

print("\n✅ Advantage: Fast, safe, predictable memory usage")

# COMMAND ----------

# DBTITLE 1,Using collect() - processes all data
# Method 2: collect() - processes ALL data
import time
import sys

print("⚠️ Method 2: collect()")
print("Retrieves ALL rows (limiting to 10K for demo safety)\n")

# Use a smaller subset for safe demonstration
small_df = demo_df.limit(10000)

start = time.time()
result_collect = small_df.collect()
time_collect = time.time() - start

print(f"Rows retrieved: {len(result_collect):,}")
print(f"Memory usage: ~{sys.getsizeof(result_collect) / 1024:.2f} KB")
print(f"Execution time: {time_collect:.4f} seconds")
print("\nFirst 5 rows:")
for row in result_collect[:5]:
    print(f"  {row}")

print("\n⚠️ Warning: collect() on full 1M rows would use much more memory!")
print("   Always prefer take() or limit() for exploration")

# COMMAND ----------

# DBTITLE 1,Collect vs Take comparison and best practices
# Comparison and Best Practices
print("=" * 60)
print("collect() vs take() COMPARISON")
print("=" * 60)

print(f"\n✅ take(10):           {time_take:.4f} seconds")
print(f"⚠️ collect() [10K rows]: {time_collect:.4f} seconds")

print("\n" + "=" * 60)
print("BEST PRACTICES")
print("=" * 60)

print("\n✅ SAFE - Use take() or limit():")
print("   df.take(100)              # First 100 rows")
print("   df.limit(100).collect()   # Same, but via limit")
print("   df.show(20)               # Display 20 rows (most common)")

print("\n⚠️ DANGEROUS - Avoid collect() on large datasets:")
print("   df.collect()              # ❌ Can crash driver!")
print("   df.toPandas()             # ❌ Also brings all data to driver")

print("\n💡 Memory Estimation:")
print(f"   1M rows ≈ 16 MB")
print(f"   10M rows ≈ 160 MB")
print(f"   100M rows ≈ 1.6 GB (driver crash risk!)")

print("\n" + "=" * 60)
print("WHEN TO USE EACH")
print("=" * 60)
print("\n✅ Use take(n) for:")
print("   - Data exploration")
print("   - Debugging")
print("   - Previewing results")
print("   - Unit tests")

print("\n⚠️ Use collect() only when:")
print("   - Result set is guaranteed small (< 1000 rows)")
print("   - After aggressive filtering/aggregation")
print("   - You verified the count first")

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## **Summary: Shuffle Reduction & Data Collection Best Practices**
# MAGIC
# MAGIC ### 🚀 Shuffle Reduction Techniques:
# MAGIC
# MAGIC 1. **Filter Early**
# MAGIC    - Apply filters before joins/aggregations
# MAGIC    - Reduces data volume early in the pipeline
# MAGIC    - Can reduce shuffle by 50-90%
# MAGIC
# MAGIC 2. **coalesce() vs repartition()**
# MAGIC    - Use `coalesce()` to reduce partitions (no full shuffle)
# MAGIC    - Use `repartition()` to fix skew or increase partitions
# MAGIC    - `coalesce()` is 2-3x faster for reducing partitions
# MAGIC
# MAGIC 3. **Other Techniques** (covered in other notebooks):
# MAGIC    - Broadcast joins for small tables
# MAGIC    - Direct aggregations instead of collect_list
# MAGIC    - Partition by column for future queries
# MAGIC
# MAGIC ### 💡 collect() vs take():
# MAGIC
# MAGIC | Operation | When to Use | Risk Level |
# MAGIC |-----------|-------------|------------|
# MAGIC | `take(n)` | Always for previewing | 🟢 Safe |
# MAGIC | `limit(n).collect()` | Alternative to take() | 🟢 Safe |
# MAGIC | `show(n)` | Display in notebook | 🟢 Safe |
# MAGIC | `collect()` | Small result sets only | 🔴 Dangerous |
# MAGIC | `toPandas()` | Small DataFrames only | 🔴 Dangerous |
# MAGIC
# MAGIC ### ✅ Golden Rules:
# MAGIC 1. Always filter early
# MAGIC 2. Use `coalesce()` when reducing partitions
# MAGIC 3. Never `collect()` without knowing the size
# MAGIC 4. Use `take()` for exploration
# MAGIC 5. Monitor shuffle in Spark UI