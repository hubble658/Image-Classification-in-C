#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// #define _CRTDBG_MAP_ALLOC //Memoryleak test etmek icindir. Memory Leak bulunmamaktadır.
// #include <crtdbg.h>
#define PI 3.14159265358979323846
#define EPS 0.000001
#define ETA 0.01

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
    double m_gradient_sum;

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
Neuron* newNuron(int weightNum, int myIndex) {
    int i;
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));

    neuron->m_output_val = 0.0f; // default value
    neuron->m_gradient = 0.0f; // default value
    neuron->m_gradient_sum = 0.0f; // default value
    neuron->m_myIndex = myIndex;
    neuron->weightNum = weightNum;
    neuron->weights = NULL; // null for output layer

    if (weightNum != 0) {
        neuron->weights = (double*)malloc(sizeof(double) * weightNum);
        for (i = 0; i < weightNum; i++)
        {
            neuron->weights[i] = getRandom();
        }
    }
    return neuron;
}
void freeNuron(Neuron* neuron) {
    if (neuron) {
        free(neuron->weights);
        free(neuron);
    }
}

void feedForwardNeuron(Neuron* neuron, Layer* prevLayer) {
    int i;
    double sum = 0;
    for (i = 0; i < prevLayer->neuronNum; i++)
    {
        sum += prevLayer->neurons[i]->weights[neuron->m_myIndex] *
            prevLayer->neurons[i]->m_output_val;
    }
    neuron->m_output_val = activation(sum);

}

//Layer Codes
Layer* newLayer(int neuronNum, int nextNeuronNum, LayerType type) {

    neuronNum = neuronNum + 1; // neuronNum+1 because we add bias as a neuron execpt last layer.

    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->neurons = (Neuron**)malloc(sizeof(Neuron*) * neuronNum);
    layer->type = type;
    layer->neuronNum = neuronNum;

    int i;
    if (neuronNum > 0) {
        for (i = 0; i < neuronNum; i++)
        {
            layer->neurons[i] = newNuron(nextNeuronNum, i);
            if (!layer->neurons[i]) {
                printf("Memory allocation failed for Neuron %d\n", i);
                // Free previously allocated neurons and the layer
                for (int j = 0; j < i; j++) {
                    freeNuron(layer->neurons[j]);
                }
                free(layer->neurons);
                free(layer);
                return NULL;
            }
        }
    }

    //Set 1 to bias value. It will be ajdusted via weight
    layer->neurons[neuronNum - 1]->m_output_val = 1.0f;

    return layer;
}

void freeLayer(Layer* layer) {

    if (layer) {
        for (int i = 0; i < layer->neuronNum; i++)
        {
            freeNuron(layer->neurons[i]);
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
    Net* net = (Net*)malloc(sizeof(Net));
    net->topology = topology;
    net->layerNum = layerNum;
    net->layers = (Layer**)malloc(sizeof(Layer*) * layerNum);
    net->m_errorSmoothingFactor = 100.0f;
    net->m_recentAverageError = 0;
    net->m_error = 0;

    //First input layer
    net->layers[0] = newLayer(topology[0], topology[1], INPUT_LAYER);
    //Hidden layers
    for (i = 1; i < layerNum - 1; i++)
    {
        net->layers[i] = newLayer(topology[i], topology[i + 1], HIDDEN_LAYER);
    }
    //Output layer
    net->layers[layerNum - 1] = newLayer(topology[layerNum - 1], 0, OUTPUT_LAYER);

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
        return;
    }
    //Set first layers value with input
    for (i = 0; i < net->layers[0]->neuronNum - 1; i++){
        net->layers[0]->neurons[i]->m_output_val = inputVals[i];
    }
    //Set first layers value with input
    for (i = 1; i < net->layerNum; i++){
        feedForwardLayer(net->layers[i - 1], net->layers[i]);
    }

}

void updateWeightsNeuron(Neuron* neuron, Layer* prevLayer , double learningRate) {

    //Kontrol et sonra
    int i;
    for (i = 0; i < prevLayer->neuronNum; i++)
    {
        Neuron* prevNeuron = prevLayer->neurons[i];
        prevNeuron->weights[neuron->m_myIndex] += learningRate * (prevNeuron->m_output_val) * neuron->m_gradient;
    }

}

double sumDOW(Layer* nextLayer, Neuron* neuron) { // sum of Derivate Of Weigths
    int i;
    double sum = 0.0;
    for (i = 0; i < nextLayer->neuronNum - 1; i++) // update this when added BIAS
    {
        sum += neuron->weights[i] * nextLayer->neurons[i]->m_gradient;
    }
    return sum;

}

void calculateHiddenGrad(Layer* nextLayer, Neuron* neuron) {
    //Gradient Descent
    double dow = sumDOW(nextLayer, neuron);
    neuron->m_gradient = dow * derivActivation(neuron->m_output_val);
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

    m_error = m_error / outputLayer->neuronNum + EPS;
    m_error = sqrtf(m_error); //RMS
    net->m_error = m_error;

    net->m_recentAverageError = (net->m_recentAverageError * net->m_errorSmoothingFactor + net->m_error)
        / (net->m_errorSmoothingFactor + 1.0);

}

void backPropagation(Net* net, double* targetVals, int targetSize, double learningRate) {

    Layer* outputLayer = net->layers[net->layerNum - 1];

    //Dont count bias at last layer
    if (outputLayer->neuronNum - 1 != targetSize) {
        printf("\n!!! OUTPUT SIZE DONT MATCH !!!\nCouldnt use backpropagation");
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
            updateWeightsNeuron(net->layers[layerNum]->neurons[n], net->layers[layerNum - 1] ,learningRate);
        }
    }

}

void softmaxOutputNet(Net* net) {

    int i;
    double sum = 0.0;
    double temprature = 2.0;

    int lastLayer = net->layerNum - 1;
    for (i = 0; i < net->layers[lastLayer]->neuronNum - 1; i++){
        sum += exp(net->layers[lastLayer]->neurons[i]->m_output_val/temprature);
    }
    for (i = 0; i < net->layers[lastLayer]->neuronNum - 1; i++){
        double subAns = exp(net->layers[lastLayer]->neurons[i]->m_output_val/temprature)/sum;
        printf("%d.Output :%.3f\n", i , subAns);
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
            (*targetVal)[i][j] = -1;  // Initialize target values to 0
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
    if (result > 1) {
        result = 1.0f;
    }
    else if (result < -1.0) {
        result = -1.0f;
    }
    return result;
}

double activation(double x) { return tanh(x); }
double derivActivation(double x) { return 1 - tanh(x) * tanh(x); }

// double activation(double x) { return 1 / (1 + exp(-x)); }  // Sigmoid function
// double derivActivation(double x) { return activation(x) * (1 - activation(x)); }  // Derivative of sigmoid


//TO DO

//add extra attributes to struct (needed for Adam)
//add gd
//add sgd 

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

    double learningRate = 0.01f;
    int epoch = 17;
    for (j = 0; j < epoch; j++)
    {
        // if(j == epoch*(7.0/10.0) ){
        //     printf("LearningRate has decreased\n");
        //     learningRate/=10;
        // }
        for (i =  data.rowNumber/10; i < data.rowNumber; i++)
        {
            // int inputSize = sizeof(data.inputVals[i]) / sizeof(data.inputVals[i][0]); 
            feedForwardNet(myNet,data.inputVals[i],data.colNumber);
            backPropagation(myNet,data.targetVals[i],data.numOfClasses ,learningRate);
            // printf("---------------\n");
        }
        for (i =  0; i < data.rowNumber/10; i++)
        {
            // int inputSize = sizeof(data.inputVals[i]) / sizeof(data.inputVals[i][0]); 
            feedForwardNet(myNet,data.inputVals[i],data.colNumber);
            calculateErr(myNet,data.targetVals[i]);
            // printf("---------------\n");
        }

        printf("Error :%lf \n",myNet->m_recentAverageError);
    }


    int correct = 0;
    int wrong = 0;
    for (i = 0 ; i < data.rowNumber/10; i++)
    {
        // int inputSize = sizeof(data.inputVals[i]) / sizeof(data.inputVals[i][0]); 
        // printf("---------------\n");
        feedForwardNet(myNet,data.inputVals[i],data.colNumber);
        // softmaxOutputNet(myNet);
        // calculateErr(myNet,data.targetVals[i]);


        int target = 0;
        for (j = 0; j < data.numOfClasses; j++)
        {
            if(data.targetVals[i][j]>0){
                target = j;
            }
        }
        // printf("target val was :%d\n",target);

        double ans = -1;
        int ansIndex = 0;
        for (j = 0; j < data.numOfClasses; j++)
        {
            if(myNet->layers[myNet->layerNum-1]->neurons[j]->m_output_val > ans){
                ans = myNet->layers[myNet->layerNum-1]->neurons[j]->m_output_val;
                ansIndex = j;
            }
        }
        if(target == ansIndex){
            correct++;
        }else{
            // printf("Ans :%d , Real:%d\n",target,ansIndex);
            wrong++;
        }

        
    }
    printf("Correct : %d \nWrong : %d",correct,wrong);


    // freeNet(myNet);
    // _CrtDumpMemoryLeaks();
    return 0;
}