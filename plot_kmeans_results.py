import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


points = []


with open(
    "kmeans_results.csv",
    "r",
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        points.append(
            {
                "x":
                float(row["x"]),

                "y":
                float(row["y"]),

                "cluster":
                int(row["cluster"]),
            }
        )


centers = {}


with open(
    "kmeans_centers.csv",
    "r",
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        centers[
            int(row["cluster"])
        ] = (
            float(row["x_center"]),
            float(row["y_center"]),
        )


# ---------------------------------------------------------
# Original input-data figure
# ---------------------------------------------------------

plt.figure(
    figsize=(7, 6)
)


plt.scatter(
    [
        point["x"]
        for point in points
    ],
    [
        point["y"]
        for point in points
    ],
    facecolors="none",
    edgecolors="black",
    marker="o",
)


plt.xlabel("x")
plt.ylabel("y")

plt.title(
    "K-Means Input Dataset"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "kmeans_input_data.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


cluster1 = [
    point
    for point in points
    if point["cluster"] == 1
]


cluster2 = [
    point
    for point in points
    if point["cluster"] == 2
]


# ---------------------------------------------------------
# Required clustering-result figure
# ---------------------------------------------------------

plt.figure(
    figsize=(7, 6)
)


# Cluster 1 = RED CROSS

plt.scatter(
    [
        point["x"]
        for point in cluster1
    ],
    [
        point["y"]
        for point in cluster1
    ],
    c="red",
    marker="x",
    label="Cluster 1",
)


# Cluster 2 = BLUE CIRCLE

plt.scatter(
    [
        point["x"]
        for point in cluster2
    ],
    [
        point["y"]
        for point in cluster2
    ],
    c="blue",
    marker="o",
    label="Cluster 2",
)


# Cluster 1 center = RED TRIANGLE

plt.scatter(
    centers[1][0],
    centers[1][1],
    c="red",
    marker="^",
    s=220,
    edgecolors="black",
    label="Cluster 1 Center",
)


# Cluster 2 center = BLUE SQUARE

plt.scatter(
    centers[2][0],
    centers[2][1],
    c="blue",
    marker="s",
    s=220,
    edgecolors="black",
    label="Cluster 2 Center",
)


plt.xlabel("x")
plt.ylabel("y")

plt.title(
    "Spark ML K-Means Clustering Results"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "kmeans_clusters.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print("Created:")
print("kmeans_input_data.png")
print("kmeans_clusters.png")
