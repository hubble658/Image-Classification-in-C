#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// #define _CRTDBG_MAP_ALLOC //Memoryleak test etmek icindir. Memory Leak bulunmamaktadır.
// #include <crtdbg.h>
#define PI 3.14159265358979323846
#define EPS 0.000001
#define ETA 0.001
#define GRAD_CLIP 100
#define WEIGHT_CLIP 100

double getRandom();
double activation(double x);
double derivActivation(double x);
typedef enum {
    INPUT_LAYER,
    HIDDEN_LAYER,
    OUTPUT_LAYER
} LayerType;

typedef struct  {
    double** inputVals;
    double** targetVals;
    int rowCount;
    int colCount;
    int numOfClasses;
}Data;

typedef struct 
{
    double* weights; // Neurons Connections
    double m_output_val; // Calculated Value that will be multiplied with weights
    double m_gradient;
    double m_gradient_sum;

    int weightCount;
    int m_myIndex;
} Neuron;

typedef struct 
{
    int neuronNum;
    Neuron** neurons; // Pointer to array of neuron pointers
    LayerType type;

} Layer;

typedef struct 
{
    int* topology;
    int layerNum;

    double m_error;
    double m_recentAverageError;
    double m_errorSmoothingFactor;
    Layer** layers;
}Net;

//------------------- Neuron Codes -------------------
Neuron* newNuron(int weightCount, int myIndex) {
    int i;
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));

    neuron->m_output_val = 0.0f; // default value
    neuron->m_gradient = 0.0f; // default value
    neuron->m_gradient_sum = 0.0f; // default value
    neuron->m_myIndex = myIndex;
    neuron->weightCount = weightCount;
    neuron->weights = NULL; // null for output layer

    if (weightCount != 0) {
        neuron->weights = (double*)malloc(sizeof(double) * weightCount);
        for (i = 0; i < weightCount; i++)
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

void updateWeightsNeuron(Neuron* neuron, Layer* prevLayer , double learningRate, int batchSize) {
    int i;
    for (i = 0; i < prevLayer->neuronNum; i++)
    {
        Neuron* prevNeuron = prevLayer->neurons[i];
        //Gradient clipping
        double avgGradient = neuron->m_gradient_sum / batchSize;
        // printf("%lf \n",val);
        if(avgGradient >  GRAD_CLIP) avgGradient =  GRAD_CLIP;
        if(avgGradient < -GRAD_CLIP) avgGradient = -GRAD_CLIP;

        // prevNeuron->weights[neuron->m_myIndex] += learningRate  * avgGradient;
        prevNeuron->weights[neuron->m_myIndex] += learningRate * (prevNeuron->m_output_val) * avgGradient;
    }
    // printf("SUm : %lf\n",neuron->m_gradient_sum);
}

void updateWeightsNeuronNew(Neuron* neuron , double learningRate, int batchSize) {
    int i;
    for (i = 0; i < neuron->weightCount; i++)
    {
        double avgGradient = neuron->m_gradient_sum / batchSize;
        // printf("%lf \n",val);
        if(avgGradient >  GRAD_CLIP) avgGradient =  GRAD_CLIP;
        if(avgGradient < -GRAD_CLIP) avgGradient = -GRAD_CLIP;

        neuron->weights[i] += learningRate  * avgGradient;
        // prevNeuron->weights[neuron->m_myIndex] += learningRate * (prevNeuron->m_output_val) * avgGradient;
    }
    // printf("SUm : %lf\n",neuron->m_gradient_sum);
}

// ------------------- Layer Codes -------------------

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
    for (i = 0; i < currLayer->neuronNum -1; i++)
    {
        feedForwardNeuron(currLayer->neurons[i], prevLayer);
    }

}

// ------------------- Net Codes -------------------

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
    net->layers[0] = newLayer(net->topology[0], net->topology[1], INPUT_LAYER);
    //Hidden layers
    for (i = 1; i < layerNum - 1; i++)
    {
        net->layers[i] = newLayer(net->topology[i], net->topology[i + 1], HIDDEN_LAYER);
    }
    //Output layer
    net->layers[layerNum - 1] = newLayer(net->topology[layerNum - 1], 0, OUTPUT_LAYER);

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
        printf("%d",net->topology[0]);
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

void updateWeightsNet(Net* net, double learningRate,int batchSize){

    int layerNum;
    int n;
    // Go backwards while updating

    for (layerNum = net->layerNum - 1; layerNum >= 0; --layerNum)
    {
        for (n = 0; n < net->layers[layerNum]->neuronNum - 1; n++)
        {
            // updateWeightsNeuron(net->layers[layerNum]->neurons[n], net->layers[layerNum - 1] ,learningRate, batchSize);
            updateWeightsNeuronNew(net->layers[layerNum]->neurons[n],learningRate, batchSize);
        }
    }
    resetGradientsSum(net);
}

// ------------------- Calculatinons -------------------
void resetGradientsSum(Net* net) {
    int i, j, k;
    //Read weights from file
    for (i = 0; i < net->layerNum - 1; i++)
    {
        for (j = 0; j < net->layers[i]->neuronNum; j++)
        {
            net->layers[i]->neurons[j]->m_gradient_sum = 0;
            //Weight Clipping
            for (k = 0; k < net->layers[i]->neurons[j]->weightCount; k++)
            {
                if(net->layers[i]->neurons[j]->weights[k] >  WEIGHT_CLIP) net->layers[i]->neurons[j]->weights[k] =  WEIGHT_CLIP;
                if(net->layers[i]->neurons[j]->weights[k] < -WEIGHT_CLIP) net->layers[i]->neurons[j]->weights[k] = -WEIGHT_CLIP;
            }
            
        }
    }
}

double sumDOW(Neuron* neuron,Layer* nextLayer) { // sum of Derivate Of Weigths
    int i;
    double sum = 0.0;
    for (i = 0; i < nextLayer->neuronNum - 1; i++) 
    {
        sum += neuron->weights[i] * nextLayer->neurons[i]->m_gradient;
    }
    return sum;

}

void calculateHiddenGrad(Neuron* neuron , Layer* nextLayer) {
    //Gradient Descent
    double dow = sumDOW(nextLayer, neuron);
    neuron->m_gradient = dow * derivActivation(neuron->m_output_val); 
    neuron->m_gradient_sum += neuron->m_gradient;
}

void calculateOutputGrad(double* targetVals, Layer* outputLayer) {
    int i;
    double delta;
    for (i = 0; i < outputLayer->neuronNum - 1; i++) { // Bias dan dolayı neuronNum-1 e kadar gidiyor
        //Gradient Descent -(y[i] - y_hat[i]) * derivative
        delta = (targetVals[i] - outputLayer->neurons[i]->m_output_val );
        outputLayer->neurons[i]->m_gradient = delta * derivActivation(outputLayer->neurons[i]->m_output_val);
        outputLayer->neurons[i]->m_gradient_sum += delta * derivActivation(outputLayer->neurons[i]->m_output_val);
    }
}

double calculateErr(Net* net,double* targetVals){
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
    // net->m_error = m_error;
    // net->m_recentAverageError = (net->m_recentAverageError * net->m_errorSmoothingFactor + net->m_error)
    //     / (net->m_errorSmoothingFactor + 1.0);
    return m_error;
}

void backPropagation(Net* net, double* targetVals, int targetSize, double learningRate) {

    Layer* outputLayer = net->layers[net->layerNum - 1];

    //Dont count bias at last layer
    if (outputLayer->neuronNum - 1 != targetSize) {
        printf("\n!!! OUTPUT SIZE DONT MATCH !!!\nCouldnt use backpropagation");
        return;
    }

    //Calculate gradient for output layer
    calculateOutputGrad(targetVals, outputLayer);


    //Calculate gradient for hidden
    int layerNum;
    int n;
    for (layerNum = net->layerNum - 2; layerNum >= 0; --layerNum)
    {
        Layer* hiddenLayer = net->layers[layerNum];
        Layer* nextLayer = net->layers[layerNum + 1];

        for (n = 0; n < hiddenLayer->neuronNum; ++n) {
            calculateHiddenGrad( hiddenLayer->neurons[n] ,nextLayer);
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
// double activation(double x) { return x > 0 ? x : 0.01 * x; }
// double derivActivation(double x) { return x > 0 ? 1 : 0.01; }
// double activation(double x) { return 1 / (1 + exp(-x)); }  // Sigmoid function
// double derivActivation(double x) { return activation(x) * (1 - activation(x)); }  // Derivative of sigmoid

// ------------------- Save-Read Neural Net -------------------

void getNetProperties(Net** net, Data* data) {
    int layerNum = 0;
    printf("Enter Hidden layer number: ");
    scanf_s("%d", &layerNum);
    layerNum += 2; // Add input and output layers

    int* topology = (int*)malloc(sizeof(int) * layerNum);
    if (topology == NULL) {
        printf("Memory allocation failed for topology\n");
        return;
    }

    printf("Enter the neuron number for Hidden layers:\n");
    for (int i = 1; i < layerNum - 1; i++) {
        scanf_s("%d", &topology[i]);
    }

    // Set input and output layer sizes
    topology[0] = data->colCount;
    topology[layerNum - 1] = data->numOfClasses;
    
    *net = newNet(topology, layerNum);
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
            for (k = 0; k < net->layers[i]->neurons[j]->weightCount; k++)
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
            for (k = 0; k < net->layers[i]->neurons[j]->weightCount; k++)
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
            for (k = 0; k < net->layers[i]->neurons[j]->weightCount; k++)
            {
                printf("%lf ", net->layers[i]->neurons[j]->weights[k]);
            }
            printf("\n");
        }
        printf("\n");
    }

}

// ------------------- Save-Read Data -------------------
void readData(FILE* fp, Data* data) {
    int i, j, targetIndex;

    // Read rowCount, colCount, and numOfClasses from the file
    fscanf(fp, "%d ", &data->rowCount);
    fscanf(fp, "%d ", &data->colCount);
    fscanf(fp, "%d \n", &data->numOfClasses);

    printf("Rows: %d, Columns: %d, Classes: %d\n", data->rowCount, data->colCount, data->numOfClasses);

    // Allocate memory for inputVals
    data->inputVals = (double**)malloc(sizeof(double*) * (data->rowCount));
    for (i = 0; i < (data->rowCount); i++) {
        data->inputVals[i] = (double*)malloc(sizeof(double) * (data->colCount));
    }

    // Allocate memory for targetVals
    data->targetVals = (double**)malloc(sizeof(double*) * (data->rowCount));
    for (i = 0; i < (data->rowCount); i++) {
        data->targetVals[i] = (double*)malloc(sizeof(double) * (data->numOfClasses));
        for (j = 0; j < (data->numOfClasses); j++) {
            data->targetVals[i][j] = -1;  // Initialize target values to -1
        }
    }

    // Read data and target values from the file
    for (i = 0; i < (data->rowCount); i++) {

        fscanf(fp, "%d ", &targetIndex); // Read target index
        if (targetIndex >= 0 && targetIndex < data->numOfClasses) {
            data->targetVals[i][targetIndex] = 1; // Set target value to 1 for the correct class
        }

        // Read the input values
        for (j = 0; j < (data->colCount); j++) {
            fscanf(fp, "%lf ", &data->inputVals[i][j]);
            // (*data)[i][j] = (*data)[i][j]/255.0f;
        }
        fscanf(fp, "\n");
    }
}

void printData(Data* data) {

    int i, j, k;
    //Read Layer num

    printf("%d ", data->rowCount);
    printf("%d ", data->colCount);
    printf("%d ", data->numOfClasses);

    printf("\n");


    for (i = 0; i < data->rowCount; i++)
    {

        for (j = 0; j < data->numOfClasses; j++)
        {
            if(data->targetVals[i][j]!=0){
                printf("%d ",j);
            }
        }
        // printf("\n");

        for (j = 0; j < data->colCount; j++)
        {
            printf("%.0f ", data->inputVals[i][j]);
        }
        printf("\n");
    }

}

void freeData(Data* data) {
    int i;

    // Free inputVals
    for (i = 0; i < data->rowCount; i++) {
        free(data->inputVals[i]);
    }
    free(data->inputVals);

    // Free targetVals
    for (i = 0; i < data->rowCount; i++) {
        free(data->targetVals[i]);
    }
    free(data->targetVals);
}

// ------------------- Optimizers -------------------

void trainGD(Net* net , Data* data , int epoch , double learningRate){

    int epochIdx, dataIdx , i;
    int dataCount = data->rowCount; // 80-20 ratio. First 20 percent is for testing the model 
    // int splitIdx = (2 * dataCount) /10; // 80-20 ratio. First 20 percent is for testing the model 
    double loss = 0;
    double valError = 0;

    //Train
    for (epochIdx = 0; epochIdx < epoch; epochIdx++)
    {
        valError = 0;
        loss = 0;
        for (dataIdx = 0; dataIdx < dataCount; dataIdx++)
        {
            feedForwardNet(net ,data->inputVals[dataIdx] , data->colCount);
            backPropagation(net ,data->targetVals[dataIdx] ,data->numOfClasses , learningRate);
            // printf("---------------\n");
            loss += calculateErr(net , data->targetVals[dataIdx]);
        }
        updateWeightsNet(net, learningRate, dataCount);

        // for (dataIdx =  0; dataIdx <splitIdx; dataIdx++)
        // {
        //     feedForwardNet(net , data->inputVals[dataIdx], data->colCount);
        //     valError += calculateErr(net , data->targetVals[dataIdx]);
        // }
        loss = loss/ dataCount;
        // valError = valError/ splitIdx;
        printf("Loss :%lf ,\n", loss);
        // printf(" Validation Error:%lf\n",valError);
    }

    // //Test how many correct answers
    // int correct = 0;
    // int wrong = 0;
    // for (dataIdx =  0; dataIdx <splitIdx; dataIdx++)
    // {
    //     feedForwardNet(net , data->inputVals[dataIdx], data->colCount);

    //     int target = 0;
    //     int ansIdx = 0;
    //     for (i = 0; i < data->numOfClasses; i++){
    //         if(data->targetVals[dataIdx][i]>0){
    //             target = i;
    //         }            
    //     }
    //     for (i = 0; i < data->numOfClasses; i++){
    //         if(net->layers[net->layerNum-1]->neurons[i]->m_output_val > net->layers[net->layerNum-1]->neurons[ansIdx]->m_output_val ){
    //             ansIdx = i;
    //         }            
    //     }

    //     if(target == ansIdx){
    //         correct++;
    //     }else{
    //         wrong++;
    //     }
    // }
    // printf("Correct :%d  \nWrong :%d\n",correct,wrong);

}


int main() {

    srand(time(NULL)); 

    // FILE *fp = fopen("../data/train_0_1.txt", "r"); // Open file containing the data
    FILE *fp = fopen("D:/Codes/Projects/Image_Classification_in_C/data/test.txt", "r"); // Open file containing the data
    // FILE *fp = fopen("D:/Codes/Projects/Image_Classification_in_C/data/train_8816.txt", "r"); // Open file containing the data
    // FILE *fp = fopen("D:/Codes/Projects/Image_Classification_in_C/data/train_full.txt", "r"); // Open file containing the data

    if (fp == NULL) {
        printf("Error opening file!\n");
        return -1;
    }
    Data trainData;
    readData(fp, &trainData);
    fclose(fp);


    printf("New Net: 1 , Load Net from file : 2\nOption: \n");
    Net* myNet;
    int option = 1;
    // scanf_s("%d",&option);

    if(option==1){
        getNetProperties(&myNet,&trainData);
    }
    else{
        myNet = readNet();
        printNet(myNet);
    }

    trainGD(myNet, &trainData , 1000 , ETA);

    

    freeNet(myNet);
    // _CrtDumpMemoryLeaks();
    return 0;
}