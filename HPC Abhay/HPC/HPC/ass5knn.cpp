#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <limits> // Include for numeric_limits

using namespace std;

// Function to compute the Euclidean distance between two points
double distance(const vector<double>& point1, const vector<double>& point2) {
    double dist = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        dist += pow(point1[i] - point2[i], 2);
    }
    return sqrt(dist);
}

// Function to assign each data point to the nearest cluster
void assign_clusters(const vector<vector<double>>& data, const vector<vector<double>>& centroids, vector<int>& cluster_assignments) {
    int num_data_points = data.size();
    int num_clusters = centroids.size();

    #pragma omp parallel for
    for (int i = 0; i < num_data_points; ++i) {
        double min_dist = numeric_limits<double>::max();
        int closest_cluster = -1;

        // Find the nearest cluster for the current data point
        for (int j = 0; j < num_clusters; ++j) {
            double dist = distance(data[i], centroids[j]);
            if (dist < min_dist) {
                min_dist = dist;
                closest_cluster = j;
            }
        }

        // Assign the data point to the nearest cluster
        cluster_assignments[i] = closest_cluster;
    }
}

int main() {
    // Example data and centroids
    vector<vector<double>> data = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    vector<vector<double>> centroids = {{2, 3}, {6, 7}};
    // Initialize cluster assignments
    vector<int> cluster_assignments(data.size());

    // Assign data points to clusters
    assign_clusters(data, centroids, cluster_assignments);

    // Output cluster assignments
    for (size_t i = 0; i < cluster_assignments.size(); ++i) {
        cout << "Data point " << i << " belongs to cluster " << cluster_assignments[i] << endl;
    }

    return 0;
}
