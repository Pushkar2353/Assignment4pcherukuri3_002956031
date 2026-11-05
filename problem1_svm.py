import csv

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.classification import (
    LinearSVC,
    OneVsRest,
)

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
)

from pyspark.ml.tuning import (
    CrossValidator,
    ParamGridBuilder,
)


# ---------------------------------------------------------
# Create the Spark session
# ---------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("Assignment4-Iris-SVM")
    .getOrCreate()
)

# Reduce unnecessary Spark INFO messages.
spark.sparkContext.setLogLevel("WARN")


# ---------------------------------------------------------
# Input file
# ---------------------------------------------------------

INPUT_FILE = (
    "/home/pcherukuri3/"
    "Assignment4/"
    "iris.csv"
)

SPECIES_NAMES = [
    "setosa",
    "versicolor",
    "virginica",
]


# ---------------------------------------------------------
# Read the Iris CSV file
# ---------------------------------------------------------

iris = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(INPUT_FILE)
)


print("\n=== IRIS DATASET ===")

iris.show(
    10,
    truncate=False,
)

print("\n=== IRIS SCHEMA ===")

iris.printSchema()

print(
    "Total Iris rows:",
    iris.count(),
)


print("\n=== NUMBER OF RECORDS PER SPECIES ===")

iris.groupBy(
    "species"
).count().orderBy(
    "species"
).show()


# ---------------------------------------------------------
# Convert text species into numeric labels
#
# setosa      = 0
# versicolor  = 1
# virginica   = 2
# ---------------------------------------------------------

labeled = iris.withColumn(
    "label",

    when(
        col("species") == "setosa",
        0.0
    )

    .when(
        col("species") == "versicolor",
        1.0
    )

    .when(
        col("species") == "virginica",
        2.0
    )
)


# ---------------------------------------------------------
# Combine the 4 measurements into the features column
# ---------------------------------------------------------

assembler = VectorAssembler(

    inputCols=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ],

    outputCol="features",
)


data = (
    assembler
    .transform(labeled)

    .select(
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
        "label",
        "features",
    )

    .cache()
)


print("\n=== PREPARED SVM DATA ===")

data.show(
    10,
    truncate=False,
)


# ---------------------------------------------------------
# Split data into training and testing sets
# ---------------------------------------------------------

training, test = data.randomSplit(

    [0.8, 0.2],

    seed=42,
)


training = training.cache()
test = test.cache()


print(
    "Training rows:",
    training.count(),
)

print(
    "Test rows:",
    test.count(),
)


# ---------------------------------------------------------
# Create the Linear Support Vector Machine
# ---------------------------------------------------------

lsvc = LinearSVC(

    featuresCol="features",

    labelCol="label",

    maxIter=100,

    regParam=0.1,
)


# ---------------------------------------------------------
# Iris has 3 classes.
# LinearSVC is binary.
# Therefore use One-vs-Rest.
# ---------------------------------------------------------

ovr = OneVsRest(

    classifier=lsvc,

    featuresCol="features",

    labelCol="label",

    predictionCol="prediction",

    parallelism=2,
)


# ---------------------------------------------------------
# Accuracy evaluator
# ---------------------------------------------------------

evaluator = MulticlassClassificationEvaluator(

    labelCol="label",

    predictionCol="prediction",

    metricName="accuracy",
)


# ---------------------------------------------------------
# Parameter grid
#
# We are mainly using CrossValidator for the required
# 10-fold evaluation.
# ---------------------------------------------------------

param_grid = (
    ParamGridBuilder()
    .build()
)


# ---------------------------------------------------------
# 10-fold Cross Validation
# ---------------------------------------------------------

crossval = CrossValidator(

    estimator=ovr,

    estimatorParamMaps=param_grid,

    evaluator=evaluator,

    numFolds=10,

    parallelism=2,

    seed=42,
)


print(
    "\nRunning Linear SVM + One-vs-Rest "
    "with 10-fold cross validation..."
)


# ---------------------------------------------------------
# Train using cross validation
# ---------------------------------------------------------

cv_model = crossval.fit(
    training
)


# Average validation accuracy across the folds
cv_accuracy = cv_model.avgMetrics[0]


# ---------------------------------------------------------
# Test the best model
# ---------------------------------------------------------

predictions = (
    cv_model
    .transform(test)
    .cache()
)


test_accuracy = evaluator.evaluate(
    predictions
)


print(
    f"\nSVM 10-fold CV accuracy = "
    f"{cv_accuracy:.6f}"
)

print(
    f"SVM test accuracy       = "
    f"{test_accuracy:.6f}"
)


# ---------------------------------------------------------
# Display predictions
# ---------------------------------------------------------

print(
    "\n=== SVM PREDICTIONS ==="
)


predictions.select(

    "species",

    "label",

    "prediction",

).orderBy(

    "label",

    "prediction",

).show(

    test.count(),

    truncate=False,
)


# ---------------------------------------------------------
# Confusion Matrix
#
# The professor's lecture uses:
# groupBy("label", "prediction").count()
# ---------------------------------------------------------

print(
    "\n=== SVM CONFUSION MATRIX COUNTS ==="
)


confusion_counts = (

    predictions

    .groupBy(
        "label",
        "prediction"
    )

    .count()

    .orderBy(
        "label",
        "prediction"
    )
)


confusion_counts.show()


# ---------------------------------------------------------
# Convert confusion counts into a 3 x 3 matrix
# ---------------------------------------------------------

matrix = [

    [0, 0, 0],

    [0, 0, 0],

    [0, 0, 0],
]


for row in confusion_counts.collect():

    actual = int(
        row["label"]
    )

    predicted = int(
        row["prediction"]
    )

    matrix[
        actual
    ][
        predicted
    ] = int(
        row["count"]
    )


print(
    "\nSVM Confusion Matrix:"
)

print(
    "Actual / Predicted    "
    "setosa    versicolor    virginica"
)


for i, row in enumerate(
    matrix
):

    print(

        f"{SPECIES_NAMES[i]:18}"

        f"{row[0]:8}"

        f"{row[1]:13}"

        f"{row[2]:12}"
    )


# ---------------------------------------------------------
# Save individual predictions to CSV
# ---------------------------------------------------------

with open(
    "svm_predictions.csv",
    "w",
    newline="",
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow(
        [
            "actual_species",
            "predicted_species",
            "correct",
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    )


    for row in predictions.select(

        "species",

        "prediction",

        "sepal_length",

        "sepal_width",

        "petal_length",

        "petal_width",

    ).collect():


        predicted_species = (
            SPECIES_NAMES[
                int(
                    row["prediction"]
                )
            ]
        )


        writer.writerow(
            [
                row["species"],

                predicted_species,

                row["species"]
                == predicted_species,

                row["sepal_length"],

                row["sepal_width"],

                row["petal_length"],

                row["petal_width"],
            ]
        )


# ---------------------------------------------------------
# Save confusion matrix
# ---------------------------------------------------------

with open(
    "svm_confusion_matrix.csv",
    "w",
    newline="",
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow(
        [
            "actual_predicted",
            "setosa",
            "versicolor",
            "virginica",
        ]
    )


    for i, row in enumerate(
        matrix
    ):

        writer.writerow(
            [
                SPECIES_NAMES[i],
                row[0],
                row[1],
                row[2],
            ]
        )


# ---------------------------------------------------------
# Save accuracy results
# ---------------------------------------------------------

with open(
    "svm_metrics.txt",
    "w",
) as file:

    file.write(
        f"10-fold CV accuracy: "
        f"{cv_accuracy:.6f}\n"
    )

    file.write(
        f"Test accuracy: "
        f"{test_accuracy:.6f}\n"
    )


print(
    "\nCreated:"
)

print(
    "svm_predictions.csv"
)

print(
    "svm_confusion_matrix.csv"
)

print(
    "svm_metrics.txt"
)


spark.stop()
