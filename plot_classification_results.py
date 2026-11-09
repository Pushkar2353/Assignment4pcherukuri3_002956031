import csv

import matplotlib

# Allows plotting on Ubuntu without a graphical display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


SPECIES = [
    "setosa",
    "versicolor",
    "virginica",
]


def load_confusion_matrix(filename):

    matrix = []

    with open(filename, "r") as file:

        reader = csv.reader(file)

        next(reader)

        for row in reader:

            matrix.append(
                [
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                ]
            )

    return matrix


def plot_confusion_matrix(
    filename,
    title,
    output_file,
):

    matrix = load_confusion_matrix(
        filename
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        matrix,
        cmap="Blues",
    )

    ax.set_xticks(
        range(3),
        SPECIES,
    )

    ax.set_yticks(
        range(3),
        SPECIES,
    )

    ax.set_xlabel(
        "Predicted Species"
    )

    ax.set_ylabel(
        "Actual Species"
    )

    ax.set_title(title)

    for actual in range(3):

        for predicted in range(3):

            value = matrix[
                actual
            ][
                predicted
            ]

            color = (
                "red"
                if actual != predicted
                and value > 0
                else "black"
            )

            ax.text(
                predicted,
                actual,
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                fontweight="bold",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def load_predictions(filename):

    rows = []

    with open(filename, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            rows.append(
                {
                    "actual":
                    row[
                        "actual_species"
                    ],

                    "predicted":
                    row[
                        "predicted_species"
                    ],

                    "petal_length":
                    float(
                        row[
                            "petal_length"
                        ]
                    ),

                    "petal_width":
                    float(
                        row[
                            "petal_width"
                        ]
                    ),
                }
            )

    return rows


def plot_predictions(
    filename,
    title,
    output_file,
):

    rows = load_predictions(filename)

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    markers = {
        "setosa": "o",
        "versicolor": "s",
        "virginica": "^",
    }

    for species in SPECIES:

        correct = [
            row
            for row in rows
            if row["actual"] == species
            and row["actual"]
            == row["predicted"]
        ]

        ax.scatter(
            [
                row["petal_length"]
                for row in correct
            ],
            [
                row["petal_width"]
                for row in correct
            ],
            marker=markers[species],
            label=
            f"{species} - correct",
        )

    errors = [
        row
        for row in rows
        if row["actual"]
        != row["predicted"]
    ]

    if errors:

        ax.scatter(
            [
                row["petal_length"]
                for row in errors
            ],
            [
                row["petal_width"]
                for row in errors
            ],
            c="red",
            marker="x",
            s=100,
            linewidths=2,
            label="Misclassified",
        )

    ax.set_xlabel(
        "Petal Length"
    )

    ax.set_ylabel(
        "Petal Width"
    )

    ax.set_title(title)

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


plot_confusion_matrix(
    "svm_confusion_matrix.csv",
    "Linear SVM Confusion Matrix",
    "svm_confusion_matrix.png",
)


plot_confusion_matrix(
    "mlp_confusion_matrix.csv",
    "Multilayer Perceptron Confusion Matrix",
    "mlp_confusion_matrix.png",
)


plot_predictions(
    "svm_predictions.csv",
    "Linear SVM Classification Results",
    "svm_classification_results.png",
)


plot_predictions(
    "mlp_predictions.csv",
    "Multilayer Perceptron Classification Results",
    "mlp_classification_results.png",
)


print("Created:")
print("svm_confusion_matrix.png")
print("svm_classification_results.png")
print("mlp_confusion_matrix.png")
print("mlp_classification_results.png")
