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

    int weightNum;
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
}NN;

//------------------- Neuron Codes -------------------
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

void updateWeightsNeuron(Neuron* neuron, Layer* prevLayer , double learningRate, int batchSize) {
    int i;
    for (i = 0; i < prevLayer->neuronNum; i++)
    {
        Neuron* prevNeuron = prevLayer->neurons[i];
        prevNeuron->weights[neuron->m_myIndex] += learningRate * (prevNeuron->m_output_val) * (neuron->m_gradient_sum/batchSize);
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
    for (i = 0; i < currLayer->neuronNum - 1; i++)
    {
        feedForwardNeuron(currLayer->neurons[i], prevLayer);
    }

}

// ------------------- NN Codes -------------------

NN* newNet(int* topology, int layerNum) {

    int i;
    NN* NN = (NN*)malloc(sizeof(NN));
    NN->topology = topology;
    NN->layerNum = layerNum;
    NN->layers = (Layer**)malloc(sizeof(Layer*) * layerNum);
    NN->m_errorSmoothingFactor = 100.0f;
    NN->m_recentAverageError = 0;
    NN->m_error = 0;

    //First input layer
    NN->layers[0] = newLayer(topology[0], topology[1], INPUT_LAYER);
    //Hidden layers
    for (i = 1; i < layerNum - 1; i++)
    {
        NN->layers[i] = newLayer(topology[i], topology[i + 1], HIDDEN_LAYER);
    }
    //Output layer
    NN->layers[layerNum - 1] = newLayer(topology[layerNum - 1], 0, OUTPUT_LAYER);

    return NN;
}

void freeNet(NN* NN) {
    int i;
    if (NN) {
        for (i = 0; i < NN->layerNum; i++)
        {
            freeLayer(NN->layers[i]);
        }
        free(NN->layers);
        free(NN->topology);
        free(NN);
    }
}

void feedForwardNet(NN* NN, double* inputVals, int inputSize) {

    int i;

    // Change when you add bias
    if (NN->topology[0] != inputSize) {
        printf("Couldnt Feed Forward !! \n Input Size DON'T match.\n");
        return;
    }
    //Set first layers value with input
    for (i = 0; i < NN->layers[0]->neuronNum - 1; i++){
        NN->layers[0]->neurons[i]->m_output_val = inputVals[i];
    }
    //Set first layers value with input
    for (i = 1; i < NN->layerNum; i++){
        feedForwardLayer(NN->layers[i - 1], NN->layers[i]);
    }

}

void updateWeightsNet(NN* NN, double learningRate,int batchSize){

    int layerNum;
    int n;
    // Go backwards while updating

    for (layerNum = NN->layerNum - 1; layerNum > 0; --layerNum)
    {
        for (n = 0; n < NN->layers[layerNum]->neuronNum - 1; n++)
        {
            updateWeightsNeuron(NN->layers[layerNum]->neurons[n], NN->layers[layerNum - 1] ,learningRate, batchSize);
        }
    }
    resetGradientsSum(NN);
}

// ------------------- 
void resetGradientsSum(NN* NN) {
    int i,j;
    //Read weights from file
    for (i = 0; i < NN->layerNum - 1; i++)
    {
        for (j = 0; j < NN->layers[i]->neuronNum; j++)
        {
            NN->layers[i]->neurons[j]->m_gradient_sum = 0;
        }
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
    neuron->m_gradient_sum += dow * derivActivation(neuron->m_output_val);
}

void calculateOutputGrad(double* targetVals, Layer* outputLayer) {
    int i;
    double delta;
    for (i = 0; i < outputLayer->neuronNum - 1; i++) { // Bias dan dolayı neuronNum-1 e kadar gidiyor
        //Gradient Descent
        delta = targetVals[i] - outputLayer->neurons[i]->m_output_val;
        outputLayer->neurons[i]->m_gradient = delta * derivActivation(outputLayer->neurons[i]->m_output_val);
        outputLayer->neurons[i]->m_gradient_sum += delta * derivActivation(outputLayer->neurons[i]->m_output_val);
    }
}

double calculateErr(NN* NN,double* targetVals){
    int i;
    Layer* outputLayer = NN->layers[NN->layerNum - 1];

    double m_error = 0.0f;
    double delta = 0.0f;
    for (i = 0; i < outputLayer->neuronNum - 1; i++)
    {
        delta = targetVals[i] - outputLayer->neurons[i]->m_output_val;
        m_error += delta * delta;
    }

    m_error = m_error / outputLayer->neuronNum + EPS;
    m_error = sqrtf(m_error); //RMS
    // NN->m_error = m_error;
    // NN->m_recentAverageError = (NN->m_recentAverageError * NN->m_errorSmoothingFactor + NN->m_error)
    //     / (NN->m_errorSmoothingFactor + 1.0);
    return m_error;
}

void backPropagation(NN* NN, double* targetVals, int targetSize, double learningRate) {

    Layer* outputLayer = NN->layers[NN->layerNum - 1];

    //Dont count bias at last layer
    if (outputLayer->neuronNum - 1 != targetSize) {
        printf("\n!!! OUTPUT SIZE DONT MATCH !!!\nCouldnt use backpropagation");
    }

    //Calculate gradient for output layer
    calculateOutputGrad(targetVals, outputLayer);


    //Calculate gradient for hidden
    int layerNum;
    int n;
    for (layerNum = NN->layerNum - 2; layerNum >= 0; --layerNum)
    {
        Layer* hiddenLayer = NN->layers[layerNum];
        Layer* nextLayer = NN->layers[layerNum + 1];

        for (n = 0; n < hiddenLayer->neuronNum; ++n) {
            calculateHiddenGrad(nextLayer, hiddenLayer->neurons[n]);
        }
    }

}

void softmaxOutputNet(NN* NN) {

    int i;
    double sum = 0.0;
    double temprature = 2.0;

    int lastLayer = NN->layerNum - 1;
    for (i = 0; i < NN->layers[lastLayer]->neuronNum - 1; i++){
        sum += exp(NN->layers[lastLayer]->neurons[i]->m_output_val/temprature);
    }
    for (i = 0; i < NN->layers[lastLayer]->neuronNum - 1; i++){
        double subAns = exp(NN->layers[lastLayer]->neurons[i]->m_output_val/temprature)/sum;
        printf("%d.Output :%.3f\n", i , subAns);
    }

}

void saveNet(NN* NN) {

    int i, j, k;
    FILE* fp;
    fp = fopen("weights.txt", "w");

    //Write layer num
    fprintf(fp, "%d\n", NN->layerNum);
    //Write topology
    for (i = 0; i < NN->layerNum; i++)
    {
        fprintf(fp, "%d ", NN->topology[i]);
    }
    fprintf(fp, "\n");
    //Write weigths
    for (i = 0; i < NN->layerNum - 1; i++)
    {
        for (j = 0; j < NN->layers[i]->neuronNum; j++)
        {
            for (k = 0; k < NN->layers[i]->neurons[j]->weightNum; k++)
            {
                fprintf(fp, "%f ", NN->layers[i]->neurons[j]->weights[k]);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }
}

NN* readNet() {

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

    //Create NN
    NN* NN = newNet(topology, layerNum);

    //Read weights from file
    for (i = 0; i < NN->layerNum - 1; i++)
    {
        for (j = 0; j < NN->layers[i]->neuronNum; j++)
        {
            for (k = 0; k < NN->layers[i]->neurons[j]->weightNum; k++)
            {
                fscanf_s(fp, "%f ", &NN->layers[i]->neurons[j]->weights[k]);
            }
            fscanf_s(fp, "\n");
        }
        fscanf_s(fp, "\n");
    }

    return NN;
}

void printNet(NN* NN) {

    int i, j, k;
    for (i = 0; i < NN->layerNum - 1; i++)
    {
        for (j = 0; j < NN->layers[i]->neuronNum; j++)
        {
            for (k = 0; k < NN->layers[i]->neurons[j]->weightNum; k++)
            {
                printf("%lf ", NN->layers[i]->neurons[j]->weights[k]);
            }
            printf("\n");
        }
        printf("\n");
    }

}
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
    // srand(time(NULL)); 1

    // FILE *fp = fopen("../data/train_scaled.txt", "r"); // Open file containing the data
    // FILE *fp = fopen("../data/train_0_1.txt", "r"); // Open file containing the data
    FILE *fp = fopen("D:/Codes/Projects/Image_Classification_in_C/data/train_0_1.txt", "r"); // Open file containing the data
    // FILE *fp = fopen("../data/test.txt", "r"); // Open file containing the data

    if (fp == NULL) {
        printf("Error opening file!\n");
        return -1;
    }
    Data trainData;
    readData(fp, &trainData);
    // printData(&trainData);

    fclose(fp);


    printf("New NN: 1 , Read NN : 2\nOption: \n");
    NN* myNet;
    int option = 1;
    // scanf_s("%d",&option);

    if(option==1){

        int* topology;
        int layerNum = 0;
        printf("Enter Hidden layer number :");
        scanf_s("%d", &layerNum);
        layerNum += 2;//Input and output layer is added

        printf("Enter the neruron number for each layer:");
        topology = (int*)malloc(sizeof(int) * layerNum);

        for (int i = 1; i < layerNum-1; i++){
            scanf_s(" %d", &topology[i]);
        }
        topology[0] = trainData.colCount;
        topology[layerNum-1] =trainData.numOfClasses;

        myNet = newNet(topology, layerNum);

        // printNet(myNet);
        // saveNet(myNet);

    }
    // else if(option == 2){
        // myNet = readNet();
        // printNet(myNet);
    // }


    int i,j,k;

    double learningRate = ETA;
    int epoch = 20;
    for (j = 0; j < epoch; j++)
    {
        // if(j == epoch*(7.0/10.0) ){
        //     printf("LearningRate has decreased\n");
        //     learningRate/=10;
        // }
        for (i =  trainData.rowCount/10; i < trainData.rowCount; i++)
        {
            feedForwardNet(myNet ,trainData.inputVals[i] , trainData.colCount);
            backPropagation(myNet ,trainData.targetVals[i] ,trainData.numOfClasses , learningRate);
            // printf("---------------\n");
        }
        updateWeightsNet( myNet, learningRate, trainData.rowCount);
        double err2 = 0;
        for (i =  0; i < trainData.rowCount/10; i++)
        {
            feedForwardNet(myNet , trainData.inputVals[i], trainData.colCount);
            err2 += calculateErr(myNet , trainData.targetVals[i]);
            // printf("---------------\n");
        }

        printf("Error :%lf \n",err2/(trainData.rowCount/10));
    }


    int correct = 0;
    int wrong = 0;
    double err = 0;
    for (i = 0 ; i < trainData.rowCount/10; i++)
    {
        // int inputSize = sizeof(data.inputVals[i]) / sizeof(data.inputVals[i][0]); 
        // printf("---------------\n");
        feedForwardNet(myNet, trainData.inputVals[i], trainData.colCount);
        // softmaxOutputNet(myNet);

        err += calculateErr(myNet, trainData.targetVals[i]);

        int target = 0;
        for (j = 0; j < trainData.numOfClasses; j++)
        {
            if(trainData.targetVals[i][j]>0){
                target = j;
            }
        }
        // printf("target val was :%d\n",target);

        double ans = -1;
        int ansIndex = 0;
        for (j = 0; j < trainData.numOfClasses; j++)
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
    printf("Correct : %d \nWrong : %d\n",correct,wrong);
    printf("Err: %lf  ",(err/(trainData.rowCount/10)));
    


    freeNet(myNet);
    // _CrtDumpMemoryLeaks();
    return 0;
}