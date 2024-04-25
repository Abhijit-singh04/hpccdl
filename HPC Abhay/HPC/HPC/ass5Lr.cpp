#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Function to perform linear regression using gradient descent
void linear_regression_parallel(vector<double>& X, vector<double>& y, double& theta0, double& theta1, double alpha, int num_iterations) {
    int m = X.size();
    double cost;

    for (int iter = 0; iter < num_iterations; ++iter) {
        double sum_error0 = 0.0, sum_error1 = 0.0;

        // Parallelize the computation of gradient updates
        #pragma omp parallel for reduction(+:sum_error0, sum_error1)
        for (int i = 0; i < m; ++i) {
            double prediction = theta0 + theta1 * X[i];
            double error = prediction - y[i];

            sum_error0 += error;
            sum_error1 += error * X[i];
        }

        // Update parameters theta0 and theta1
        theta0 -= (alpha / m) * sum_error0;
        theta1 -= (alpha / m) * sum_error1;

        // Calculate cost (optional)
        cost = 0.0;
        for (int i = 0; i < m; ++i) {
            double prediction = theta0 + theta1 * X[i];
            cost += (prediction - y[i]) * (prediction - y[i]);
        }
        cost /= (2 * m);

        // Output cost for monitoring convergence (optional)
        cout << "Iteration " << iter + 1 << ", Cost: " << cost <<" theta0 "<<theta0<<" theta1 "<<theta1<< endl;
    }
}

int main() {
    // Example dataset
    vector<double> X = {1, 2, 3, 4, 5};
    vector<double> y = {3, 5, 7, 9, 11};

    // Initial parameters
    double theta0 = 0.0;
    double theta1 = 0.0;

    // Learning rate and number of iterations
    double alpha = 0.01;
    int num_iterations = 1000;

    // Perform linear regression using gradient descent
    linear_regression_parallel(X, y, theta0, theta1, alpha, num_iterations);

    // Output final parameters
    cout << "Final parameters: theta0 = " << theta0 << ", theta1 = " << theta1 << endl;

    return 0;
}
