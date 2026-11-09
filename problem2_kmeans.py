import csv

from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


spark = (
    SparkSession.builder
    .appName("Assignment4-KMeans")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


INPUT_FILE = (
    "/home/pcherukuri3/"
    "Assignment4/"
    "kmeans_input_libsvm.txt"
)


# ---------------------------------------------------------
# Load required LIBSVM dataset.
# ---------------------------------------------------------

dataset = (
    spark.read
    .format("libsvm")
    .load(INPUT_FILE)
    .cache()
)


print(
    "\n=== K-MEANS INPUT DATASET ==="
)

dataset.show(
    20,
    truncate=False,
)


print(
    "\n=== DATASET SCHEMA ==="
)

dataset.printSchema()


total_points = dataset.count()


print(
    "Total data points:",
    total_points,
)


# ---------------------------------------------------------
# K-Means with required k = 2.
# ---------------------------------------------------------

kmeans = (
    KMeans()
    .setK(2)
    .setSeed(1)
)


model = kmeans.fit(
    dataset
)


predictions = (
    model
    .transform(dataset)
    .cache()
)


print(
    "\n=== SAMPLE CLUSTER ASSIGNMENTS ==="
)


predictions.select(
    "features",
    "prediction",
).show(
    20,
    truncate=False,
)


print(
    "\n=== NUMBER OF POINTS IN EACH CLUSTER ==="
)


cluster_counts = (
    predictions
    .groupBy("prediction")
    .count()
    .orderBy("prediction")
)


cluster_counts.show()


# ---------------------------------------------------------
# Evaluate clustering.
# ---------------------------------------------------------

evaluator = ClusteringEvaluator()


silhouette = float(
    evaluator.evaluate(predictions)
)


print(
    f"\nSilhouette with squared "
    f"euclidean distance = "
    f"{silhouette:.6f}"
)


# ---------------------------------------------------------
# Required cluster centers.
# ---------------------------------------------------------

centers = model.clusterCenters()


print(
    "\n=== CLUSTER CENTERS ==="
)


for index, center in enumerate(
    centers,
    start=1,
):

    print(
        f"Cluster {index} Center: "
        f"x = {float(center[0]):.6f}, "
        f"y = {float(center[1]):.6f}"
    )


# ---------------------------------------------------------
# Save all 200 points and calculated cluster labels.
# ---------------------------------------------------------

result_rows = predictions.select(
    "features",
    "prediction",
).collect()


with open(
    "kmeans_results.csv",
    "w",
    newline="",
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "x",
            "y",
            "cluster",
        ]
    )

    for row in result_rows:

        features = row["features"]

        # Spark uses cluster IDs 0 and 1.
        # Report/visualization uses Cluster 1 and Cluster 2.
        cluster_number = (
            int(
                row["prediction"]
            )
            + 1
        )

        writer.writerow(
            [
                float(features[0]),
                float(features[1]),
                cluster_number,
            ]
        )


with open(
    "kmeans_centers.csv",
    "w",
    newline="",
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "cluster",
            "x_center",
            "y_center",
        ]
    )

    for index, center in enumerate(
        centers,
        start=1,
    ):

        writer.writerow(
            [
                index,
                float(center[0]),
                float(center[1]),
            ]
        )


with open(
    "kmeans_metrics.txt",
    "w",
) as file:

    file.write(
        f"Total points: "
        f"{total_points}\n"
    )

    file.write(
        f"Silhouette score: "
        f"{silhouette:.6f}\n"
    )

    for index, center in enumerate(
        centers,
        start=1,
    ):

        file.write(
            f"Cluster {index} Center: "
            f"x = {float(center[0]):.6f}, "
            f"y = {float(center[1]):.6f}\n"
        )


print("\nFiles created:")
print("kmeans_results.csv")
print("kmeans_centers.csv")
print("kmeans_metrics.txt")


spark.stop()
