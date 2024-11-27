#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>

using namespace std;

class NeuralNetwork {
private:
    int inputLayerSize = 2;
    int hiddenLayerSize = 3;
    int outputLayerSize = 1;

    vector<vector<double>> W1; // Weights between input and hidden layer
    vector<vector<double>> W2; // Weights between hidden and output layer

    vector<double> sigmoid(const vector<double>& z) {
        vector<double> result(z.size());
        for (size_t i = 0; i < z.size(); i++) {
            result[i] = 1.0 / (1.0 + exp(-z[i]));
        }
        return result;
    }

    vector<double> sigmoidPrime(const vector<double>& z) {
        vector<double> result(z.size());
        for (size_t i = 0; i < z.size(); i++) {
            double sig = 1.0 / (1.0 + exp(-z[i]));
            result[i] = sig * (1.0 - sig);
        }
        return result;
    }

    vector<double> dotProduct(const vector<vector<double>>& matrix, const vector<double>& vec) {
        vector<double> result(matrix.size(), 0.0);
        for (size_t i = 0; i < matrix.size(); i++) {
            for (size_t j = 0; j < vec.size(); j++) {
                result[i] += matrix[i][j] * vec[j];
            }
        }
        return result;
    }

    vector<vector<double>> transpose(const vector<vector<double>>& matrix) {
        vector<vector<double>> result(matrix[0].size(), vector<double>(matrix.size()));
        for (size_t i = 0; i < matrix.size(); i++) {
            for (size_t j = 0; j < matrix[0].size(); j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    vector<vector<double>> randomMatrix(int rows, int cols) {
        vector<vector<double>> matrix(rows, vector<double>(cols));
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> dis(0.0, 1.0);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = dis(gen);
            }
        }
        return matrix;
    }

public:
    NeuralNetwork() {
        // Initialize weights randomly
        W1 = randomMatrix(inputLayerSize, hiddenLayerSize);
        W2 = randomMatrix(hiddenLayerSize, outputLayerSize);
    }

    vector<double> forward(const vector<double>& X) {
        z2 = dotProduct(W1, X);
        a2 = sigmoid(z2);
        z3 = dotProduct(W2, a2);
        yHat = sigmoid(z3);
        return yHat;
    }

    double costFunction(const vector<double>& X, const vector<double>& y) {
        vector<double> yHat = forward(X);
        double cost = 0.0;
        for (size_t i = 0; i < y.size(); i++) {
            cost += 0.5 * pow(y[i] - yHat[i], 2);
        }
        return cost;
    }

    pair<vector<vector<double>>, vector<vector<double>>> costFunctionPrime(const vector<double>& X, const vector<double>& y) {
        forward(X);

        // Compute gradients
        vector<double> delta3(outputLayerSize);
        for (size_t i = 0; i < outputLayerSize; i++) {
            delta3[i] = -(y[i] - yHat[i]) * sigmoidPrime(z3)[i];
        }

        vector<vector<double>> dJdW2(hiddenLayerSize, vector<double>(outputLayerSize));
        vector<vector<double>> a2_T = transpose({a2});
        for (size_t i = 0; i < hiddenLayerSize; i++) {
            for (size_t j = 0; j < outputLayerSize; j++) {
                dJdW2[i][j] = a2_T[i][0] * delta3[j];
            }
        }

        vector<double> delta2(hiddenLayerSize);
        vector<vector<double>> W2_T = transpose(W2);
        for (size_t i = 0; i < hiddenLayerSize; i++) {
            for (size_t j = 0; j < outputLayerSize; j++) {
                delta2[i] += delta3[j] * W2_T[i][j];
            }
            delta2[i] *= sigmoidPrime(z2)[i];
        }

        vector<vector<double>> dJdW1(inputLayerSize, vector<double>(hiddenLayerSize));
        vector<vector<double>> X_T = transpose({X});
        for (size_t i = 0; i < inputLayerSize; i++) {
            for (size_t j = 0; j < hiddenLayerSize; j++) {
                dJdW1[i][j] = X_T[i][0] * delta2[j];
            }
        }

        return {dJdW1, dJdW2};
    }

private:
    vector<double> z2, a2, z3, yHat;
};
