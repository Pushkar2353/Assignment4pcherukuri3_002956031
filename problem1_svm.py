import csv

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


spark = (
    SparkSession.builder
    .appName("Assignment4-Iris-SVM")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


INPUT_FILE = "/home/pcherukuri3/Assignment4/iris.csv"

SPECIES_NAMES = [
    "setosa",
    "versicolor",
    "virginica",
]


# ---------------------------------------------------------
# Load Iris data
# ---------------------------------------------------------

iris = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(INPUT_FILE)
)


print("\n=== IRIS DATASET ===")
iris.show(10, truncate=False)

print("\n=== IRIS SCHEMA ===")
iris.printSchema()

print("Total Iris records:", iris.count())

print("\n=== RECORDS PER SPECIES ===")

iris.groupBy(
    "species"
).count().orderBy(
    "species"
).show()


# ---------------------------------------------------------
# Convert species to numeric label
# ---------------------------------------------------------

labeled = iris.withColumn(
    "label",
    when(
        col("species") == "setosa",
        0.0,
    )
    .when(
        col("species") == "versicolor",
        1.0,
    )
    .when(
        col("species") == "virginica",
        2.0,
    )
)


# ---------------------------------------------------------
# Create feature vector
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
        "species",
        "label",
        "features",
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    )
    .cache()
)


print("\n=== PREPARED DATA ===")

data.show(
    10,
    truncate=False,
)


# ---------------------------------------------------------
# Independent test set
#
# Cross validation is performed on the training portion.
# The held-out test portion provides an additional
# independent evaluation and confusion matrix.
# ---------------------------------------------------------

training, test = data.randomSplit(
    [0.8, 0.2],
    seed=42,
)

training = training.cache()
test = test.cache()


training_count = training.count()
test_count = test.count()


print("Training records:", training_count)
print("Test records:", test_count)


# ---------------------------------------------------------
# Binary Linear SVM
# ---------------------------------------------------------

svm = LinearSVC(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.1,
)


# ---------------------------------------------------------
# Convert binary LinearSVC to multiclass classifier.
# ---------------------------------------------------------

ovr = OneVsRest(
    classifier=svm,
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    parallelism=2,
)


evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy",
)


# One fixed SVM configuration evaluated through
# the required ten folds.
param_grid = (
    ParamGridBuilder()
    .build()
)


cross_validator = CrossValidator(
    estimator=ovr,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=10,
    parallelism=2,
    seed=42,
)


print(
    "\nRunning Linear SVM + One-vs-Rest "
    "with 10-fold cross-validation..."
)


cv_model = cross_validator.fit(
    training
)


cv_accuracy = float(
    cv_model.avgMetrics[0]
)


predictions = (
    cv_model
    .transform(test)
    .cache()
)


test_accuracy = float(
    evaluator.evaluate(predictions)
)


print(
    f"\nSVM 10-fold CV accuracy = "
    f"{cv_accuracy:.6f}"
)

print(
    f"SVM test accuracy       = "
    f"{test_accuracy:.6f}"
)


print("\n=== SVM PREDICTIONS ===")

predictions.select(
    "species",
    "label",
    "prediction",
).orderBy(
    "label",
    "prediction",
).show(
    test_count,
    truncate=False,
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

confusion = (
    predictions
    .groupBy(
        "label",
        "prediction",
    )
    .count()
    .orderBy(
        "label",
        "prediction",
    )
)


print(
    "\n=== SVM CONFUSION MATRIX COUNTS ==="
)

confusion.show()


matrix = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]


for row in confusion.collect():

    actual = int(row["label"])
    predicted = int(row["prediction"])

    matrix[actual][predicted] = int(
        row["count"]
    )


print("\nSVM Confusion Matrix")

print(
    "Actual / Predicted    "
    "setosa    versicolor    virginica"
)


for index, values in enumerate(matrix):

    print(
        f"{SPECIES_NAMES[index]:18}"
        f"{values[0]:8}"
        f"{values[1]:13}"
        f"{values[2]:12}"
    )


# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------

prediction_rows = predictions.select(
    "species",
    "prediction",
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
).collect()


with open(
    "svm_predictions.csv",
    "w",
    newline="",
) as file:

    writer = csv.writer(file)

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

    for row in prediction_rows:

        predicted_species = (
            SPECIES_NAMES[
                int(row["prediction"])
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


with open(
    "svm_confusion_matrix.csv",
    "w",
    newline="",
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "actual_predicted",
            "setosa",
            "versicolor",
            "virginica",
        ]
    )

    for index, values in enumerate(matrix):

        writer.writerow(
            [
                SPECIES_NAMES[index],
                values[0],
                values[1],
                values[2],
            ]
        )


with open(
    "svm_metrics.txt",
    "w",
) as file:

    file.write(
        f"Training records: "
        f"{training_count}\n"
    )

    file.write(
        f"Test records: "
        f"{test_count}\n"
    )

    file.write(
        f"10-fold CV accuracy: "
        f"{cv_accuracy:.6f}\n"
    )

    file.write(
        f"Test accuracy: "
        f"{test_accuracy:.6f}\n"
    )


print("\nFiles created:")
print("svm_predictions.csv")
print("svm_confusion_matrix.csv")
print("svm_metrics.txt")


spark.stop()
