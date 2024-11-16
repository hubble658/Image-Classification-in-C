#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// #define _CRTDBG_MAP_ALLOC //Memoryleak test etmek icindir. Memory Leak bulunmamaktadır.
// #include <crtdbg.h>

float getRadnomF();
float activationF(float x);
float derivActivationF(float x);
typedef enum {
    INPUT_LAYER,
    HIDDEN_LAYER,
    OUTPUT_LAYER
} LayerType;

typedef struct N
{
    float* weights; // Neurons Connections
    int weightNum;
    float m_output_val; // Calculated Value that will be multiplied with weights
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
    Layer** layers;
}Net;

//Neuron Codes
Neuron* newNuron(int weightNum, int myIndex) {
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));

    neuron->m_output_val = 0.0f; // default value
    neuron->m_myIndex = myIndex;
    neuron->weightNum = weightNum;
    neuron->weights = NULL; // null for output layer

    if (weightNum != 0) {
        neuron->weights = (float*)malloc(sizeof(float) * weightNum);
        for (int i = 0; i < weightNum; i++)
        {
            neuron->weights[i] = getRadnomF();
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

void feedForwardNeuron(Neuron* neuron,Layer* prevLayer){
    int i;
    float sum = 0;
    for ( i = 0; i < prevLayer->neuronNum; i++)
    {
        sum += prevLayer->neurons[i]->weights[neuron->m_myIndex]*
        prevLayer->neurons[i]->m_output_val;
    }
    neuron->m_output_val = activationF(sum);
    
}

//Layer Codes
Layer* newLayer(int neuronNum, int nextNeuronNum, LayerType type) {

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

void feedForwardLayer(Layer* prevLayer, Layer* currLayer){
    int i;

    for ( i = 0; i < currLayer->neuronNum; i++)
    {
        feedForwardNeuron(currLayer->neurons[i],prevLayer);
    }
    
}

//Net Codes

Net* newNet(int* topology, int layerNum) {

    Net* net = (Net*)malloc(sizeof(Net));
    net->topology = topology;
    net->layerNum = layerNum;
    net->layers = (Layer**)malloc(sizeof(Layer*) * layerNum);

    //First input layer
    net->layers[0] = newLayer(topology[0], topology[1], INPUT_LAYER);
    //Hidden layers
    for (int i = 1; i < layerNum - 1; i++)
    {
        net->layers[i] = newLayer(topology[i], topology[i + 1], HIDDEN_LAYER);
    }
    //Output layer
    net->layers[layerNum - 1] = newLayer(topology[layerNum - 1], 0, OUTPUT_LAYER);

    return net;
}

void freeNet(Net* net) {

    if (net) {
        for (int i = 0; i < net->layerNum; i++)
        {
            freeLayer(net->layers[i]);
        }
        free(net->layers);
        free(net->topology);
        free(net);
    }
}

void feedForwardNet(Net* net,float * inputVals,int inputSize){
    
    int i,j,k;

    // Change when you add bias
    if(net->topology[0] != inputSize){
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
        feedForwardLayer(net->layers[i-1],net->layers[i]);
    }
    
}



void printOutputNet(Net* net){
    
    int i,j,k;

    int lastLayer = net->layerNum-1;
    for (i = 0; i < net->layers[lastLayer]->neuronNum; i++)
    {
        printf("%d.Output :%f\n",i+1,net->layers[lastLayer]->neurons[i]->m_output_val);
    }
    
}


void saveNet(Net* net){

    int i,j,k;
    FILE* fp;
    fp = fopen("weights.txt","w");

    //Write layer num
    fprintf(fp,"%d\n", net->layerNum);
    //Write topology
    for (i = 0; i < net->layerNum; i++)
    {
        fprintf(fp,"%d ", net->topology[i]);
    }
    fprintf(fp,"\n");
    //Write weigths
    for (i = 0; i < net->layerNum-1; i++)
    {
        for (j = 0; j < net->layers[i]->neuronNum; j++)
        {
            for (k = 0; k < net->layers[i]->neurons[j]->weightNum; k++)
            {
                fprintf(fp,"%f ", net->layers[i]->neurons[j]->weights[k]);
            }
            fprintf(fp,"\n");
        }
        fprintf(fp,"\n");
    }
}

Net* readNet(){

    int i,j,k;
    FILE* fp;
    fp = fopen("weights.txt","r");

    //Read Layer num
    int layerNum;
    fscanf_s(fp,"%d\n", &layerNum);

    //Read topology
    int* topology;
    topology = (int*)malloc(sizeof(int) * layerNum);
    for (i = 0; i < layerNum; i++)
    {
        fscanf_s(fp,"%d ", &topology[i]);
    }
    fscanf_s(fp,"\n");

    //Create net
    Net* net = newNet(topology, layerNum);
    
    //Read weights from file
    for (i = 0; i < net->layerNum-1; i++)
    {
        for (j = 0; j < net->layers[i]->neuronNum; j++)
        {
            for (k = 0; k < net->layers[i]->neurons[j]->weightNum; k++)
            {
                fscanf_s(fp,"%f ", &net->layers[i]->neurons[j]->weights[k]);
            }
            fscanf_s(fp,"\n");
        }
        fscanf_s(fp,"\n");
    }

    return net;
}

void printNet(Net* net){

    int i,j,k;
    for (i = 0; i < net->layerNum-1; i++)
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


float getRadnomF() {
    return (float)rand() / (float)RAND_MAX;
}
float activationF(float x){return tanhf(x);}
float derivActivationF(float x){return 1 - x * x;}


int main() {

    //Random comment out below ,if u want random seed
    srand(time(NULL)); 

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

        printNet(myNet);
        saveNet(myNet);

    }else{
        
        myNet = readNet();
        printNet(myNet);
        
    }

    float data[][2] = {
        {0,0},
        {1,0},
        {0,1},
        {1,1}            
    };
    float targetVal[][1] = {
        {0},
        {1},
        {1},
        {0}            
    };

    printf("Feed Forward\n");

    int i,j,k;

    for (i = 0; i < 4; i++)
    {
        int inputSize = sizeof(data[i]) / sizeof(data[i][0]); 
        feedForwardNet(myNet,&data[i],inputSize);
        printOutputNet(myNet);
        printf("---------------\n");
    }
    




    // for (int i = 0; i < outLayer.neuronNum; i++)
    // {
    //     Neuron* currNeuron = &outLayer.neurons[i];
    //     Layer* previousLayer = &inLayer;

    //     float sum = 0.0f;
    //     for (int i = 0; i < previousLayer->neuronNum; i++)
    //     {
    //         sum += previousLayer->neurons[i].weights[currNeuron->m_myIndex] * previousLayer->neurons[i].m_output_val;
    //     }
    //     currNeuron->m_output_val = sum;
    //     printf("sum : %f\n",sum);
    // }


    freeNet(myNet);
    // _CrtDumpMemoryLeaks();
    return 0;
}