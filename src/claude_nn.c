#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    int input_size;
    int output_size;
    double **weights;
    double *bias;
    double *activations;
    // Add fields for backpropagation
    double *deltas;
    double **weight_gradients;
    double *bias_gradients;
} Layer;

typedef struct {
    int num_layers;
    int *layer_sizes;
    Layer **layers;
    double learning_rate;
} NeuralNetwork;

typedef struct {
    int num_samples;
    int input_size;
    int num_classes;
    double **inputs;
    double **targets;
} Dataset;

// Initialize a single layer
Layer* init_layer(int input_size, int output_size) {
    Layer *layer = (Layer*)malloc(sizeof(Layer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    
    // Allocate and initialize weights with random values
    layer->weights = (double**)malloc(output_size * sizeof(double*));
    layer->weight_gradients = (double**)malloc(output_size * sizeof(double*));
    for(int i = 0; i < output_size; i++) {
        layer->weights[i] = (double*)malloc(input_size * sizeof(double));
        layer->weight_gradients[i] = (double*)malloc(input_size * sizeof(double));
        for(int j = 0; j < input_size; j++) {
            layer->weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            layer->weight_gradients[i][j] = 0.0;
        }
    }
    
    // Allocate and initialize other arrays
    layer->bias = (double*)calloc(output_size, sizeof(double));
    layer->bias_gradients = (double*)calloc(output_size, sizeof(double));
    layer->activations = (double*)calloc(output_size, sizeof(double));
    layer->deltas = (double*)calloc(output_size, sizeof(double));
    
    return layer;
}

// Create network with specified topology
NeuralNetwork* create_network(int* sizes, int num_layers, double learning_rate) {
    if (num_layers < 2) {
        printf("Error: Network must have at least input and output layers\n");
        return NULL;
    }
    
    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->learning_rate = learning_rate;
    
    // Copy layer sizes
    nn->layer_sizes = (int*)malloc(num_layers * sizeof(int));
    for(int i = 0; i < num_layers; i++) {
        nn->layer_sizes[i] = sizes[i];
    }
    
    // Create layers
    nn->layers = (Layer**)malloc((num_layers - 1) * sizeof(Layer*));
    for(int i = 0; i < num_layers - 1; i++) {
        nn->layers[i] = init_layer(sizes[i], sizes[i + 1]);
    }
    
    return nn;
}

// Activation function and its derivative
double tanh_activation(double x) {
    return tanh(x);
}

double tanh_derivative(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}

// Forward propagation through a single layer
void forward_layer(Layer *layer, double *input) {
    for(int i = 0; i < layer->output_size; i++) {
        double sum = layer->bias[i];
        for(int j = 0; j < layer->input_size; j++) {
            sum += input[j] * layer->weights[i][j];
        }
        layer->activations[i] = tanh_activation(sum);
    }
}

// Forward propagation through entire network
void forward_prop(NeuralNetwork *nn, double *input) {
    double *current_input = input;
    
    for(int i = 0; i < nn->num_layers - 1; i++) {
        forward_layer(nn->layers[i], current_input);
        current_input = nn->layers[i]->activations;
    }
}

// Compute error for output layer
void compute_output_error(Layer *layer, double *target) {
    for(int i = 0; i < layer->output_size; i++) {
        double error = target[i] - layer->activations[i];
        layer->deltas[i] = error * tanh_derivative(layer->activations[i]);
    }
}

// Compute error for hidden layer
void compute_hidden_error(Layer *current, Layer *next) {
    for(int i = 0; i < current->output_size; i++) {
        double error = 0.0;
        for(int j = 0; j < next->output_size; j++) {
            error += next->deltas[j] * next->weights[j][i];
        }
        current->deltas[i] = error * tanh_derivative(current->activations[i]);
    }
}

// Update weights and biases for a layer
void update_layer(Layer *layer, double *input, double learning_rate) {
    for(int i = 0; i < layer->output_size; i++) {
        for(int j = 0; j < layer->input_size; j++) {
            layer->weights[i][j] += learning_rate * layer->deltas[i] * input[j];
        }
        layer->bias[i] += learning_rate * layer->deltas[i];
    }
}

// Backpropagation
void backprop(NeuralNetwork *nn, double *input, double *target) {
    // Compute output layer error
    Layer *output_layer = nn->layers[nn->num_layers - 2];
    compute_output_error(output_layer, target);
    
    // Compute hidden layer errors
    for(int i = nn->num_layers - 3; i >= 0; i--) {
        compute_hidden_error(nn->layers[i], nn->layers[i + 1]);
    }
    
    // Update weights and biases
    double *current_input = input;
    for(int i = 0; i < nn->num_layers - 1; i++) {
        update_layer(nn->layers[i], current_input, nn->learning_rate);
        current_input = nn->layers[i]->activations;
    }
}

// Calculate mean squared error
double compute_mse(double *output, double *target, int size) {
    double mse = 0.0;
    for(int i = 0; i < size; i++) {
        double error = target[i] - output[i];
        mse += error * error;
    }
    return mse / size;
}

// Train network using batch gradient descent
void train_gd(NeuralNetwork *nn, double **inputs, double **targets, 
              int num_samples, int epochs, double error_threshold) {
    printf("\nTraining with Batch Gradient Descent:\n");
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0.0;
        
        // Process all samples in batch
        for(int i = 0; i < num_samples; i++) {
            forward_prop(nn, inputs[i]);
            backprop(nn, inputs[i], targets[i]);
            
            // Compute error for this sample
            Layer *output_layer = nn->layers[nn->num_layers - 2];
            total_error += compute_mse(output_layer->activations, targets[i], 
                                     output_layer->output_size);
        }
        
        double avg_error = total_error / num_samples;
        if(epoch % 10 == 0) {
            printf("Epoch %d: Average MSE = %.6f\n", epoch, avg_error);
        }
        
        if(avg_error < error_threshold) {
            printf("Reached error threshold after %d epochs\n", epoch);
            break;
        }
    }
}

// Train network using stochastic gradient descent
void train_sgd(NeuralNetwork *nn, double **inputs, double **targets, 
               int num_samples, int epochs, double error_threshold) {
    printf("\nTraining with Stochastic Gradient Descent:\n");
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0.0;
        
        // Process samples one at a time in random order
        int *indices = (int*)malloc(num_samples * sizeof(int));
        for(int i = 0; i < num_samples; i++) indices[i] = i;
        
        // Fisher-Yates shuffle
        for(int i = num_samples - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        for(int i = 0; i < num_samples; i++) {
            int idx = indices[i];
            forward_prop(nn, inputs[idx]);
            backprop(nn, inputs[idx], targets[idx]);
            
            // Compute error for this sample
            Layer *output_layer = nn->layers[nn->num_layers - 2];
            total_error += compute_mse(output_layer->activations, targets[idx],
                                     output_layer->output_size);
        }
        
        free(indices);
        
        double avg_error = total_error / num_samples;
        if(epoch % 10 == 0) {
            printf("Epoch %d: Average MSE = %.6f\n", epoch, avg_error);
        }
        
        if(avg_error < error_threshold) {
            printf("Reached error threshold after %d epochs\n", epoch);
            break;
        }
    }
}

// Free memory
void free_layer(Layer *layer) {
    for(int i = 0; i < layer->output_size; i++) {
        free(layer->weights[i]);
        free(layer->weight_gradients[i]);
    }
    free(layer->weights);
    free(layer->weight_gradients);
    free(layer->bias);
    free(layer->bias_gradients);
    free(layer->activations);
    free(layer->deltas);
    free(layer);
}

void free_network(NeuralNetwork *nn) {
    for(int i = 0; i < nn->num_layers - 1; i++) {
        free_layer(nn->layers[i]);
    }
    free(nn->layers);
    free(nn->layer_sizes);
    free(nn);
}

void free_dataset(Dataset *dataset) {
    if(dataset) {
        for(int i = 0; i < dataset->num_samples; i++) {
            free(dataset->inputs[i]);
            free(dataset->targets[i]);
        }
        free(dataset->inputs);
        free(dataset->targets);
        free(dataset);
    }
}

Dataset* read_dataset(const char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }

    Dataset *dataset = (Dataset*)malloc(sizeof(Dataset));
    
    // Read header information
    fscanf(file, "%d %d %d", &dataset->num_samples, &dataset->input_size, &dataset->num_classes);
    
    // Allocate memory for inputs and targets
    dataset->inputs = (double**)malloc(dataset->num_samples * sizeof(double*));
    dataset->targets = (double**)malloc(dataset->num_samples * sizeof(double*));
    
    for(int i = 0; i < dataset->num_samples; i++) {
        dataset->inputs[i] = (double*)malloc(dataset->input_size * sizeof(double));
        dataset->targets[i] = (double*)calloc(dataset->num_classes, sizeof(double));
    }
    
    // Read samples
    for(int i = 0; i < dataset->num_samples; i++) {
        int class_label;
        fscanf(file, "%d", &class_label);
        
        // Set the target (one-hot encoding)
        dataset->targets[i][class_label] = 1.0;
        
        // Read input values
        for(int j = 0; j < dataset->input_size; j++) {
            fscanf(file, "%lf", &dataset->inputs[i][j]);
        }
    }
    
    fclose(file);
    return dataset;
}

int* get_topology(int input_size, int num_classes, int *num_layers) {
    printf("\nEnter the number of layers (including input and output layers): ");
    scanf("%d", num_layers);
    
    if(*num_layers < 2) {
        printf("Error: Network must have at least input and output layers\n");
        return NULL;
    }
    
    int *topology = (int*)malloc(*num_layers * sizeof(int));
    
    // Set input layer size
    topology[0] = input_size;
    printf("Input layer size: %d (automatically set)\n", input_size);
    
    // Get hidden layer sizes
    for(int i = 1; i < *num_layers - 1; i++) {
        printf("Enter size for hidden layer %d: ", i);
        scanf("%d", &topology[i]);
        
        if(topology[i] <= 0) {
            printf("Error: Layer size must be positive\n");
            free(topology);
            return NULL;
        }
    }
    
    // Set output layer size
    topology[*num_layers - 1] = num_classes;
    printf("Output layer size: %d (automatically set)\n", num_classes);
    
    return topology;
}
// Function to print dataset information
void print_dataset(Dataset *dataset) {
    printf("\nDataset Information:\n");
    printf("Number of samples: %d\n", dataset->num_samples);
    printf("Input size: %d\n", dataset->input_size);
    printf("Number of classes: %d\n", dataset->num_classes);
    
    printf("\nFirst few samples:\n");
    int samples_to_show = dataset->num_samples < 5 ? dataset->num_samples : 5;
    
    for(int i = 0; i < samples_to_show; i++) {
        printf("Sample %d: Class [", i + 1);
        for(int k = 0; k < dataset->num_classes; k++) {
            printf("%.0f", dataset->targets[i][k]);
            if(k < dataset->num_classes - 1) printf(" ");
        }
        printf("] Values [");
        for(int j = 0; j < dataset->input_size; j++) {
            printf("%.2f", dataset->inputs[i][j]);
            if(j < dataset->input_size - 1) printf(" ");
        }
        printf("]\n");
    }
}

// Example usage
int main() {
    srand(time(NULL));

    // Read dataset
    Dataset *dataset = read_dataset("../data/train_scaled.txt");
    if(!dataset) {
        printf("Failed to read dataset\n");
        return 1;
    }
    
    // Print dataset information
    printf("\nDataset Information:\n");
    printf("Number of samples: %d\n", dataset->num_samples);
    printf("Input size: %d\n", dataset->input_size);
    printf("Number of classes: %d\n", dataset->num_classes);

    
    // Get network topology from user
    int num_layers;
    int *topology = get_topology(dataset->input_size, dataset->num_classes, &num_layers);
    if(!topology) {
        printf("Failed to create topology\n");
        free_dataset(dataset);
        return 1;
    }
    
    
    // Get learning rate from user
    double learning_rate = 0.001;
    // printf("\nEnter learning rate (recommended range 0.001-0.1): ");
    // scanf("%lf", &learning_rate);
    
    // Create neural networks
    NeuralNetwork *nn = create_network(topology, num_layers, learning_rate);
    NeuralNetwork *nn_sgd = create_network(topology, num_layers, learning_rate);
    
    // Get training parameters
    int epochs = 10000;
    double error_threshold = 0.001;
    // printf("Enter number of epochs: ");
    // scanf("%d", &epochs);
    // printf("Enter error threshold (recommended range 0.001-0.01): ");
    // scanf("%lf", &error_threshold);
    
    // Train using both methods
    // printf("\nTraining using Batch Gradient Descent:\n");
    // train_gd(nn, dataset->inputs, dataset->targets, dataset->num_samples, epochs, error_threshold);
    
    printf("\nTraining using Stochastic Gradient Descent:\n");
    train_sgd(nn_sgd, dataset->inputs, dataset->targets, dataset->num_samples, epochs, error_threshold);
    
    // Test both networks
    // printf("\nTesting Batch GD Network:\n");
    // for(int i = 0; i < (dataset->num_samples < 5 ? dataset->num_samples : 5); i++) {
    //     forward_prop(nn, dataset->inputs[i]);
    //     printf("Sample %d - Predicted: [", i + 1);
    //     for(int j = 0; j < dataset->num_classes; j++) {
    //         printf("%.3f", nn->layers[nn->num_layers-2]->activations[j]);
    //         if(j < dataset->num_classes - 1) printf(" ");
    //     }
    //     printf("], Target: [");
    //     for(int j = 0; j < dataset->num_classes; j++) {
    //         printf("%.0f", dataset->targets[i][j]);
    //         if(j < dataset->num_classes - 1) printf(" ");
    //     }
    //     printf("]\n");
    // }
    
    printf("\nTesting Stochastic GD Network:\n");
    for(int i = 0; i < (dataset->num_samples < 5 ? dataset->num_samples : 5); i++) {
        forward_prop(nn_sgd, dataset->inputs[i]);
        printf("Sample %d - Predicted: [", i + 1);
        for(int j = 0; j < dataset->num_classes; j++) {
            printf("%.3f", nn_sgd->layers[nn_sgd->num_layers-2]->activations[j]);
            if(j < dataset->num_classes - 1) printf(" ");
        }
        printf("], Target: [");
        for(int j = 0; j < dataset->num_classes; j++) {
            printf("%.0f", dataset->targets[i][j]);
            if(j < dataset->num_classes - 1) printf(" ");
        }
        printf("]\n");
    }
    
    // Clean up
    free(topology);
    free_dataset(dataset);
    free_network(nn);
    free_network(nn_sgd);
    
    return 0;
}