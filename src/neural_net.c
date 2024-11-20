#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// #define _CRTDBG_MAP_ALLOC //Memoryleak test etmek icindir. Memory Leak bulunmamaktadır.
// #include <crtdbg.h>
#define PI 3.14159265358979323846

float getRandomF();
float activationF(float x);
float derivActivationF(float x);
typedef enum {
    INPUT_LAYER,
    HIDDEN_LAYER,
    OUTPUT_LAYER
} LayerType;

typedef struct D {
    float** inputVals;
    float** targetVals;
    int rowNumber;
    int colNumber;
    int numOfClasses;
}Data;
typedef struct N
{
    float* weights; // Neurons Connections
    float m_output_val; // Calculated Value that will be multiplied with weights
    float m_gradient;

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

    float m_error;
    float m_recentAverageError;
    float m_errorSmoothingFactor;
    Layer** layers;
}Net;

//Neuron Codes
Neuron* newNuron(int weightNum, int myIndex) {
    int i;
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));

    neuron->m_output_val = 0.0f; // default value
    neuron->m_gradient = 0.0f; // default value
    neuron->m_myIndex = myIndex;
    neuron->weightNum = weightNum;
    neuron->weights = NULL; // null for output layer

    if (weightNum != 0) {
        neuron->weights = (float*)malloc(sizeof(float) * weightNum);
        for (i = 0; i < weightNum; i++)
        {
            neuron->weights[i] = getRandomF();
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
    float sum = 0;
    for (i = 0; i < prevLayer->neuronNum; i++)
    {
        sum += prevLayer->neurons[i]->weights[neuron->m_myIndex] *
            prevLayer->neurons[i]->m_output_val;
    }
    neuron->m_output_val = activationF(sum);

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

void feedForwardNet(Net* net, float* inputVals, int inputSize) {

    int i;

    // Change when you add bias
    if (net->topology[0] != inputSize) {
        printf("Couldnt Feed Forward !! \n Input Size DON'T match.\n");
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

float* getResultsNet(Net* net) {// no usage

    int i, j, k;
    int lastLayer = net->layerNum - 1;
    int outputNum = net->layers[lastLayer]->neuronNum;

    float* results = (float*)malloc(sizeof(float) * outputNum);

    for (i = 0; i < net->layers[lastLayer]->neuronNum; i++)
    {
        results[i] = net->layers[lastLayer]->neurons[i]->m_output_val;
    }
    return results;
}

void updateWeightsNeuron(Neuron* neuron, Layer* prevLayer) {

    //Kontrol et sonra
    int i;
    for (i = 0; i < prevLayer->neuronNum; i++)
    {
        Neuron* prevNeuron = prevLayer->neurons[i];
        prevNeuron->weights[neuron->m_myIndex] += 0.1f * prevNeuron->m_output_val * neuron->m_gradient;
    }

}

float sumDOW(Layer* nextLayer, Neuron* neuron) { // sum of Derivate Of Weigths
    int i;
    float sum = 0.0f;
    for (i = 0; i < nextLayer->neuronNum - 1; i++) // update this when added BIAS
    {
        sum += neuron->weights[i] * nextLayer->neurons[i]->m_gradient;
    }
    return sum;

}

void calculateHiddenGrad(Layer* nextLayer, Neuron* neuron) {
    //Gradient Descent
    float dow = sumDOW(nextLayer, neuron);
    neuron->m_gradient = dow * derivActivationF(neuron->m_output_val);
}
void calculateOutputGrad(float* targetVals, Layer* outputLayer) {
    int i;
    float delta;
    for (i = 0; i < outputLayer->neuronNum - 1; i++) { // Bias dan dolayı neuronNum-1 e kadar gidiyor
        //Gradient Descent
        delta = targetVals[i] - outputLayer->neurons[i]->m_output_val;
        outputLayer->neurons[i]->m_gradient = delta * derivActivationF(outputLayer->neurons[i]->m_output_val);
    }
}
void calculateErr(Net* net,float* targetVals){
    int i;
    Layer* outputLayer = net->layers[net->layerNum - 1];

    float m_error = 0.0f;
    float delta = 0.0f;
    for (i = 0; i < outputLayer->neuronNum - 1; i++)
    {
        delta = targetVals[i] - outputLayer->neurons[i]->m_output_val;
        m_error += delta * delta;
    }

    m_error = m_error / outputLayer->neuronNum;
    m_error = sqrtf(m_error); //RMS
    net->m_error = m_error;

    net->m_recentAverageError = (net->m_recentAverageError * net->m_errorSmoothingFactor + net->m_error)
        / (net->m_errorSmoothingFactor + 1.0);

}

void backPropagation(Net* net, float* targetVals, int targetSize) {

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
            updateWeightsNeuron(net->layers[layerNum]->neurons[n], net->layers[layerNum - 1]);
        }
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
                printf("%f ", net->layers[i]->neurons[j]->weights[k]);
            }
            printf("\n");
        }
        printf("\n");
    }

}
void readData(FILE* fp, float*** data, float*** targetVal, int* rowNum, int* colNum, int* numOfClasses) {
    int i, j, targetIndex;

    // Read rowNumber, colNumber, and numOfClasses from the file
    fscanf(fp, "%d %d %d\n", rowNum, colNum, numOfClasses);
    printf("Rows: %d, Columns: %d, Classes: %d\n", *rowNum, *colNum, *numOfClasses);

    // Allocate memory for inputVals
    *data = (float**)malloc(sizeof(float*) * (*rowNum));
    for (i = 0; i < (*rowNum); i++) {
        (*data)[i] = (float*)malloc(sizeof(float) * (*colNum));
    }

    // Allocate memory for targetVals
    *targetVal = (float**)malloc(sizeof(float*) * (*rowNum));
    for (i = 0; i < (*rowNum); i++) {
        (*targetVal)[i] = (float*)malloc(sizeof(float) * (*numOfClasses));
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
            fscanf(fp, "%f ", &(*data)[i][j]);
            // (*data)[i][j] = (*data)[i][j]/255.0f;
        }
        fscanf(fp, "\n");
    }
}
void writeData(float** data, float** targetVal, int* rowNum, int* colNum, int* numOfClasses) {

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

float getRandomF() {
    float result;
    do {
        // Generate uniform random numbers in the range (0, 1)
        float u = ((float)rand() / RAND_MAX);
        float v = ((float)rand() / RAND_MAX);

        // Transform to normal distribution (mean 0, std dev 1)
        result = sqrt(-2.0f * log(u)) * cos(2.0f * PI * v);

        // Increase spread (e.g., standard deviation of 3)
        result *= 3.0f;
    } while (result == 0.0f); // Regenerate if result is 0

    return result;
}
float activationF(float x) { return tanhf(x); }
float derivActivationF(float x) { return 1 - tanhf(x) * tanhf(x); }

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

    // float data[][2] = {
    //     {0,0},
    //     {1,0},
    //     {0,1},
    //     {1,1}            
    // };
    // float targetVal[][1] = {
    //     {0},
    //     {1},
    //     {1},
    //     {0}            
    // };


    FILE *fp = fopen("../data/test.txt", "r"); // Open file containing the data
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

    for (j = 0; j < 10000; j++)
    {
        for (i = 0; i < data.rowNumber; i++)
        {
            // int inputSize = sizeof(data.inputVals[i]) / sizeof(data.inputVals[i][0]); 
            feedForwardNet(myNet,data.inputVals[i],data.colNumber);
            calculateErr(myNet,data.targetVals[i]);
            backPropagation(myNet,data.targetVals[i],data.numOfClasses);
            // printf("---------------\n");
        }
        printf("Error :%f \n",myNet->m_recentAverageError);
    }

    for (i = 0; i <4; i++)
    {
        // int inputSize = sizeof(data.inputVals[i]) / sizeof(data.inputVals[i][0]); 
        printf("---------------\n");
        feedForwardNet(myNet,data.inputVals[i],data.colNumber);
        printOutputNet(myNet);
        // for ( j = 0; j < 28; j++)
        // {
        //     for (k = 0; k < 28; k++)
        //     {
        //         if(data.inputVals[i][j*28+k]>0.5f){
        //             printf("#");
        //         }else{
        //             printf(" ");
        //         }
        //     }printf("\n");
            
        // }
        // for (j = 0; j < 10; j++)
        // {
        //     if(data.targetVals[i][j]>0){
        //         printf("target val was :%d\n",j);
        //     }
        // }
        
    }


    // freeNet(myNet);
    // _CrtDumpMemoryLeaks();
    return 0;
}