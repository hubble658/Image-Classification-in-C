#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// #define _CRTDBG_MAP_ALLOC //Memoryleak test etmek icindir. Memory Leak bulunmamaktadır.
// #include <crtdbg.h>
#define PI 3.14159265358979323846
#define EPS 0.000001

const double clip_threshold = 10.0;

double getRandom();
double activation(double x);
double derivActivation(double x);
typedef enum {
    INPUT_LAYER,
    HIDDEN_LAYER,
    OUTPUT_LAYER
} LayerType;

typedef struct D {
    double** inputVals;
    double** targetVals;
    int rowNumber;
    int colNumber;
    int numOfClasses;
}Data;
typedef struct N
{
    double* weights; // Neurons Connections
    double m_output_val; // Calculated Value that will be multiplied with weights
    double m_gradient;

    int weightNum;
    int m_myIndex;
} Neuron;

typedef struct L
{
    int neuronNum;
    Neuron** neurons; // Pointer to array of neuron pointers
    LayerType type;

} Layer;

typedef struct net
{
    int* topology;
    int layerNum;

    double m_error;
    double m_recentAverageError;
    double m_errorSmoothingFactor;
    Layer** layers;
}Net;

//Neuron Codes
Neuron* newNeuron(int weightNum, int myIndex) {
    int i;

    // Allocate memory for the Neuron
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
    if (!neuron) {
        fprintf(stderr, "Memory allocation failed for Neuron.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize default values
    neuron->m_output_val = 0.0f;
    neuron->m_gradient = 0.0f;
    neuron->m_myIndex = myIndex;
    neuron->weightNum = weightNum;
    neuron->weights = NULL; // Set to NULL for output layer

    // Allocate memory for weights if necessary
    if (weightNum != 0) {
        neuron->weights = (double*)malloc(sizeof(double) * weightNum);
        if (!neuron->weights) {
            fprintf(stderr, "Memory allocation failed for Neuron weights.\n");
            free(neuron); // Clean up previously allocated memory
            exit(EXIT_FAILURE);
        }

        // Initialize weights with random values
        for (i = 0; i < weightNum; i++) {
            neuron->weights[i] = getRandom();
        }
    }

    return neuron;
}

void freeNeuron(Neuron* neuron) {
    if (neuron) {
        free(neuron->weights);
        free(neuron);
    }
}

int a = 0;
void feedForwardNeuron(Neuron* neuron, Layer* prevLayer) {
    int i;
    double sum = 0;
    for (i = 0; i < prevLayer->neuronNum; i++)
    {
        sum += prevLayer->neurons[i]->weights[neuron->m_myIndex] *
            prevLayer->neurons[i]->m_output_val;

        if(isnan(sum)){
            printf("feedForward nan index: %d , %d : \n",neuron->m_myIndex, i);
            printf("feedForward sum %lf\n",sum);
            printf("feedForward weight %lf , neuron indexi: %d\n",prevLayer->neurons[i]->weights[neuron->m_myIndex],neuron->m_myIndex);
            printf("feedForward output_val %lf\n",prevLayer->neurons[i]->m_output_val);

            exit(0);
        }
        // if(neuron->m_myIndex>53){

        //     printf("feedForward sum %lf\n",sum);
        //     printf("feedForward weight %lf , neuron indexi: %d\n",prevLayer->neurons[i]->weights[neuron->m_myIndex],neuron->m_myIndex);
        //     printf("feedForward output_val %lf\n",prevLayer->neurons[i]->m_output_val);
        // }
        // Clip gradient if too large
        if (sum > clip_threshold){ 
            // printf("%lf\n",sum);
            sum = clip_threshold;
            }
        if (sum < -clip_threshold) sum = -clip_threshold;
    }
    a++;
    // printf("%d %lf\n",a,sum);
    neuron->m_output_val = activation(sum);

}

//Layer Codes
Layer* newLayer(int neuronNum1, int nextNeuronNum, LayerType type) {
    int neuronNum = neuronNum1+1; // Add bias neuron (except for the last layer)

    // Allocate memory for the layer
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    if (!layer) {
        fprintf(stderr, "Memory allocation failed for Layer.\n");
        exit(EXIT_FAILURE);
    }

    layer->type = type;
    layer->neuronNum = neuronNum;
    // Allocate memory for the neurons array
    layer->neurons = (Neuron**)malloc(sizeof(Neuron*) * neuronNum);
    if (!layer->neurons) {
        fprintf(stderr, "Memory allocation failed for Layer neurons array.\n");
        free(layer);
        exit(EXIT_FAILURE);
    }


    // Allocate memory for each neuron
    int i;
    for (i = 0; i < neuronNum; i++) {
        layer->neurons[i] = newNeuron(nextNeuronNum, i);
        if (!layer->neurons[i]) {
            fprintf(stderr, "Memory allocation failed for Neuron %d.\n", i);
            // Free previously allocated neurons and layer
            for (int j = 0; j < i; j++) {
                freeNeuron(layer->neurons[j]);
            }
            free(layer->neurons);
            free(layer);
            exit(EXIT_FAILURE);
        }
    }

    // Set bias neuron's output value to 1.0
    layer->neurons[neuronNum - 1]->m_output_val = 1.0f;

    return layer;
}


void freeLayer(Layer* layer) {

    if (layer) {
        for (int i = 0; i < layer->neuronNum; i++)
        {
            freeNeuron(layer->neurons[i]);
        }
        free(layer->neurons);
        free(layer);
    }

}

void feedForwardLayer(Layer* prevLayer, Layer* currLayer) {
    int i;
    for (i = 0; i < currLayer->neuronNum - 1; i++)
    {
        feedForwardNeuron(currLayer->neurons[i], prevLayer);
    }

}

//Net Codes

Net* newNet(int* topology, int layerNum) {
    int i;

    // Allocate memory for the Net structure
    Net* net = (Net*)malloc(sizeof(Net));
    if (!net) {
        fprintf(stderr, "Memory allocation failed for Net.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize fields
    net->topology = topology;
    net->layerNum = layerNum;
    net->m_errorSmoothingFactor = 100.0f;
    net->m_recentAverageError = 0;
    net->m_error = 0;

    // Allocate memory for the layers array
    net->layers = (Layer**)malloc(sizeof(Layer*) * layerNum);
    if (!net->layers) {
        fprintf(stderr, "Memory allocation failed for Net layers array.\n");
        free(net);
        exit(EXIT_FAILURE);
    }

    // Allocate and initialize layers
    for (i = 0; i < layerNum; i++) {
        LayerType type = (i == 0) ? INPUT_LAYER : (i == layerNum - 1) ? OUTPUT_LAYER : HIDDEN_LAYER;
        int nextNeuronNum = (i < layerNum - 1) ? topology[i + 1] : 0;

        net->layers[i] = newLayer(topology[i], nextNeuronNum, type);
        if (!net->layers[i]) {
            fprintf(stderr, "Memory allocation failed for Layer %d.\n", i);
            // Free previously allocated layers and the Net structure
            for (int j = 0; j < i; j++) {
                freeLayer(net->layers[j]); // Assuming freeLayer properly cleans up a Layer
            }
            free(net->layers);
            free(net);
            exit(EXIT_FAILURE);
        }
    }

    return net;
}


void freeNet(Net* net) {
    int i;
    if (net) {
        for (i = 0; i < net->layerNum; i++)
        {
            freeLayer(net->layers[i]);
        }
        free(net->layers);
        free(net->topology);
        free(net);
    }
}

void feedForwardNet(Net* net, double* inputVals, int inputSize) {

    int i;

    // Change when you add bias
    if (net->topology[0] != inputSize) {
        printf("Couldnt Feed Forward !! \n Input Size DON'T match.\n");
        exit(0);
        return;
    }
    //Set first layers value with input
    for (i = 0; i < net->topology[0]; i++)
    {
        net->layers[0]->neurons[i]->m_output_val = inputVals[i];
    }
    //Set first layers value with input
    for (i = 1; i < net->layerNum; i++)
    {
        feedForwardLayer(net->layers[i - 1], net->layers[i]);
    }

}

void updateWeightsNeuron(Neuron* neuron, Layer* prevLayer) {

    //Kontrol et sonra
    int i;
    for (i = 0; i < prevLayer->neuronNum; i++)
    {
        Neuron* prevNeuron = prevLayer->neurons[i];
        double gradient = 0.1f * prevNeuron->m_output_val * neuron->m_gradient;

        // Clip gradient if too large
        if (gradient > clip_threshold) gradient = clip_threshold;
        if (gradient < -clip_threshold) gradient = -clip_threshold;

        prevNeuron->weights[neuron->m_myIndex] += gradient;

        if (prevNeuron->weights[neuron->m_myIndex] > clip_threshold) 
            prevNeuron->weights[neuron->m_myIndex] = clip_threshold;
        if (prevNeuron->weights[neuron->m_myIndex] < -clip_threshold) 
            prevNeuron->weights[neuron->m_myIndex] = -clip_threshold;
    }

}

double sumDOW(Layer* nextLayer, Neuron* neuron) { // sum of Derivate Of Weigths
    int i;
    double sum = 0.0;

    for (i = 0; i < nextLayer->neuronNum - 1; i++) // update this when added BIAS
    {
        sum += neuron->weights[i] * nextLayer->neurons[i]->m_gradient;
        if (sum > clip_threshold) {
            sum = clip_threshold;
        } else if (sum < -clip_threshold) {
            sum = -clip_threshold;
        }

        if(isnan(sum)){
            printf("Nan in sumDow : %d\n",i);
            exit(0);
        }
    }
    return sum;

}

void calculateHiddenGrad(Layer* nextLayer, Neuron* neuron) {
    double dow = sumDOW(nextLayer, neuron);
    double gradient = dow * derivActivation(neuron->m_output_val);

    // Clip gradients to a reasonable range (e.g., [-1.0, 1.0])
    if (gradient > clip_threshold) {
        gradient = clip_threshold;
    } else if (gradient < -clip_threshold) {
        gradient = -clip_threshold;
    }

    neuron->m_gradient = gradient;
    // printf("%lf\n",neuron->m_gradient);
}
void calculateOutputGrad(double* targetVals, Layer* outputLayer) {
    int i;
    double delta;
    for (i = 0; i < outputLayer->neuronNum - 1; i++) { // Bias dan dolayı neuronNum-1 e kadar gidiyor
        //Gradient Descent
        delta = targetVals[i] - outputLayer->neurons[i]->m_output_val;
        outputLayer->neurons[i]->m_gradient = delta * derivActivation(outputLayer->neurons[i]->m_output_val);
    }
}
void calculateErr(Net* net,double* targetVals){
    int i;
    Layer* outputLayer = net->layers[net->layerNum - 1];

    double m_error = 0.0f;
    double delta = 0.0f;
    for (i = 0; i < outputLayer->neuronNum - 1; i++)
    {
        delta = targetVals[i] - outputLayer->neurons[i]->m_output_val;
        m_error += delta * delta;
    }

    m_error = sqrtf(m_error); //RMS
    net->m_error += m_error;

    net->m_recentAverageError += (net->m_recentAverageError * net->m_errorSmoothingFactor + net->m_error)
        / (net->m_errorSmoothingFactor + 1.0);

}

void backPropagation(Net* net, double* targetVals, int targetSize) {

    Layer* outputLayer = net->layers[net->layerNum - 1];

    if (outputLayer->neuronNum - 1 != targetSize) {
        printf("\n!!! OUTPUT SIZE DONT MATCH !!!\nCouldnt use backpropagation");
        return;
    }

    //Calculate gradient for output layer
    calculateOutputGrad(targetVals, outputLayer);

    //Calculate gradient for hidden
    int layerNum;
    int n;
    for (layerNum = net->layerNum - 2; layerNum > 0; --layerNum)
    {
        Layer* hiddenLayer = net->layers[layerNum];
        Layer* nextLayer = net->layers[layerNum + 1];

        for (n = 0; n < hiddenLayer->neuronNum; ++n) {
            calculateHiddenGrad(nextLayer, hiddenLayer->neurons[n]);
        }
    }

    //Update Weights
    for (layerNum = net->layerNum - 1; layerNum > 0; --layerNum)
    {
        for (n = 0; n < net->layers[layerNum]->neuronNum - 1; n++)
        {
            updateWeightsNeuron(net->layers[layerNum]->neurons[n], net->layers[layerNum - 1]);
        }
    }

}

void gradientDescent(Net* net, int epoch){
    int i,j;

    for (i = 0; i < epoch; i++)
    {
        myNet->m_error = 0;
        myNet->m_recentAverageError = 0;
        for (j = 0; j < data.rowNumber; j++)
        {
            // int inputSize = sizeof(data.inputVals[i]) / sizeof(data.inputVals[i][0]); 
            feedForwardNet(myNet,data.inputVals[i],data.colNumber);
            backPropagation(myNet,data.targetVals[i],data.numOfClasses);
            // printf("---------------\n");
            calculateErr(myNet,data.targetVals[i]);
        }
        if(j%10==0)
        printf("Error :%lf \n",(myNet->m_error/data.rowNumber));

        
    }

}

void printOutputNet(Net* net) {

    int i, j, k;

    int lastLayer = net->layerNum - 1;
    for (i = 0; i < net->layers[lastLayer]->neuronNum - 1; i++)
    {
        printf("%d.Output :%f\n", i , net->layers[lastLayer]->neurons[i]->m_output_val);
    }

}

void saveNet(Net* net) {

    int i, j, k;
    FILE* fp;
    fp = fopen("weights.txt", "w");

    //Write layer num
    fprintf(fp, "%d\n", net->layerNum);
    //Write topology
    for (i = 0; i < net->layerNum; i++)
    {
        fprintf(fp, "%d ", net->topology[i]);
    }
    fprintf(fp, "\n");
    //Write weigths
    for (i = 0; i < net->layerNum - 1; i++)
    {
        for (j = 0; j < net->layers[i]->neuronNum; j++)
        {
            for (k = 0; k < net->layers[i]->neurons[j]->weightNum; k++)
            {
                fprintf(fp, "%f ", net->layers[i]->neurons[j]->weights[k]);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }
}

Net* readNet() {

    int i, j, k;
    FILE* fp;
    fp = fopen("weights.txt", "r");

    //Read Layer num
    int layerNum;
    fscanf_s(fp, "%d\n", &layerNum);

    //Read topology
    int* topology;
    topology = (int*)malloc(sizeof(int) * layerNum);
    for (i = 0; i < layerNum; i++)
    {
        fscanf_s(fp, "%d ", &topology[i]);
    }
    fscanf_s(fp, "\n");

    //Create net
    Net* net = newNet(topology, layerNum);

    //Read weights from file
    for (i = 0; i < net->layerNum - 1; i++)
    {
        for (j = 0; j < net->layers[i]->neuronNum; j++)
        {
            for (k = 0; k < net->layers[i]->neurons[j]->weightNum; k++)
            {
                fscanf_s(fp, "%f ", &net->layers[i]->neurons[j]->weights[k]);
            }
            fscanf_s(fp, "\n");
        }
        fscanf_s(fp, "\n");
    }

    return net;
}

void printNet(Net* net) {

    int i, j, k;
    for (i = 0; i < net->layerNum - 1; i++)
    {
        for (j = 0; j < net->layers[i]->neuronNum; j++)
        {
            for (k = 0; k < net->layers[i]->neurons[j]->weightNum; k++)
            {
                printf("%lf ", net->layers[i]->neurons[j]->weights[k]);
            }
            printf("\n");
        }
        printf("\n");
    }

}
void readData(FILE* fp, double*** data, double*** targetVal, int* rowNum, int* colNum, int* numOfClasses) {
    int i, j, targetIndex;

    // Read rowNumber, colNumber, and numOfClasses from the file
    fscanf(fp, "%d %d %d\n", rowNum, colNum, numOfClasses);
    printf("Rows: %d, Columns: %d, Classes: %d\n", *rowNum, *colNum, *numOfClasses);

    // Allocate memory for inputVals
    *data = (double**)malloc(sizeof(double*) * (*rowNum));
    for (i = 0; i < (*rowNum); i++) {
        (*data)[i] = (double*)malloc(sizeof(double) * (*colNum));
    }

    // Allocate memory for targetVals
    *targetVal = (double**)malloc(sizeof(double*) * (*rowNum));
    for (i = 0; i < (*rowNum); i++) {
        (*targetVal)[i] = (double*)malloc(sizeof(double) * (*numOfClasses));
        for (j = 0; j < (*numOfClasses); j++) {
            (*targetVal)[i][j] = 0;  // Initialize target values to 0
        }
    }

    // Read data and target values from the file
    for (i = 0; i < (*rowNum); i++) {

        fscanf(fp, "%d ", &targetIndex); // Read target index
        if (targetIndex >= 0 && targetIndex < *numOfClasses) {
            (*targetVal)[i][targetIndex] = 1; // Set target value to 1 for the correct class
        }

        // Read the input values
        for (j = 0; j < (*colNum); j++) {
            fscanf(fp, "%lf ", &(*data)[i][j]);
            // (*data)[i][j] = (*data)[i][j]/255.0f;
        }
        fscanf(fp, "\n");
    }
}
void writeData(double** data, double** targetVal, int* rowNum, int* colNum, int* numOfClasses) {

    int i, j, k;
    //Read Layer num

    printf("%d ", *rowNum);
    printf("%d ", *colNum);
    printf("%d ", *numOfClasses);

    printf("\n");


    for (i = 0; i < (*rowNum); i++)
    {

        for (j = 0; j < (*numOfClasses); j++)
        {
            if(targetVal[i][j]!=0){
                printf("%d ",j);
            }
        }
        // printf("\n");

        for (j = 0; j < (*colNum); j++)
        {
            printf("%.0f ", data[i][j]);
        }
        printf("\n");
    }

}
void freeData(Data* data) {
    int i;

    // Free inputVals
    for (i = 0; i < data->rowNumber; i++) {
        free(data->inputVals[i]);
    }
    free(data->inputVals);

    // Free targetVals
    for (i = 0; i < data->rowNumber; i++) {
        free(data->targetVals[i]);
    }
    free(data->targetVals);
}

double getRandom() {
    double result;
    double u = ((double)rand() / RAND_MAX);
    double v = ((double)rand() / RAND_MAX);

    result = sqrt(-2.0f * log(u)) * cos(2.0f * PI * v);
    if (result>3){
        result = 3.0f;
    }else if(result<-3.0){
        result = -3.0f;
    }
    return result;
}
double activation(double x) { return tanh(x); }
double derivActivation(double x) { return 1 - tanh(x) * tanh(x); }


// double activation(double x) { return 1 / (1 + exp(-x)); }  // Sigmoid function
// double derivActivation(double x) { return activation(x) * (1 - activation(x)); }  // Derivative of sigmoid


int main() {

    //Random comment out below ,if u want random seed
    // srand(time(NULL)); 

    printf("New net: 1 , Read Net : 2\nOption:");

    Net* myNet;

    int option;
    scanf_s("%d",&option);

    if(option==1){

        int* topology;
        int layerNum;
        printf("Layer sayisi girin:");
        scanf_s("%d", &layerNum);
        printf("Topolojiyi girin:");
        topology = (int*)malloc(sizeof(int) * layerNum);

        for (int i = 0; i < layerNum; i++){
            scanf_s(" %d", &topology[i]);
        }
        

        myNet = newNet(topology, layerNum);

        // printNet(myNet);
        // saveNet(myNet);

    }else if(option == 2){

        // myNet = readNet();
        // printNet(myNet);

    }


    FILE *fp = fopen("../data/train_scaled.txt", "r"); // Open file containing the data
    // FILE *fp = fopen("../data/test.txt", "r"); // Open file containing the data
    // FILE *fp = fopen("D:/Codes/Projects/Image_Classification_in_C/data/test.txt", "r"); // Open file containing the data

    if (fp == NULL) {
        printf("Error opening file!\n");
        return -1;
    }
    Data data;
    readData(fp, &data.inputVals, &data.targetVals, &data.rowNumber, &data.colNumber, &data.numOfClasses);
    // writeData(data.inputVals, data.targetVals, &data.rowNumber, &data.colNumber, &data.numOfClasses);

    fclose(fp);
    // printf("Feed Forward\n");

    int i,j,k;

    // feedForwardNet(myNet,data.inputVals[0],data.colNumber);
    // calculateErr(myNet,data.targetVals[0]);
    // backPropagation(myNet,data.targetVals[0],data.numOfClasses);
    // printf("---------------\n");
    // printOutputNet(myNet);
    // printNet(myNet);



    for (i = 0; i <4; i++)
    {
        // int inputSize = sizeof(data.inputVals[i]) / sizeof(data.inputVals[i][0]); 
        printf("---------------\n");
        feedForwardNet(myNet,data.inputVals[i],data.colNumber);
        printOutputNet(myNet);
        for (j = 0; j < data.numOfClasses; j++)
        {
            if(data.targetVals[i][j]>0){
                printf("target val was :%d\n",j);
            }
        }
        
    }


    // freeNet(myNet);
    // _CrtDumpMemoryLeaks();
    return 0;
}