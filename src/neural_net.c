#include <stdio.h>
#include <time.h>
#include <stdlib.h>


float getRadnomF();
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

Neuron* newNuron(int weightNum, int myIndex){
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
    neuron->m_myIndex = myIndex;
    neuron->weightNum = weightNum;
    neuron->weights = (float*)malloc(sizeof(float)*weightNum);
    for (int i = 0; i < weightNum; i++)
    {
        neuron->weights[i] = getRadnomF();
    }
    return neuron;
}
void freeNuron(Neuron* neuron){
    if (neuron) {
        free(neuron->weights);
        free(neuron);
    }
}

typedef struct L
{
    int neuronNum;
    Neuron** neurons; // Pointer to array of neuron pointers
    LayerType type;
    
} Layer;

Layer* newLayer(int neuronNum,int nextNeuronNum, LayerType type){

    Layer* layer = (Layer*) malloc(sizeof(Layer));
    layer->neurons = (Neuron**) malloc(sizeof(Neuron*)*neuronNum);
    layer->type = type;

    for (int i = 0; i < neuronNum; i++)
    {
        layer->neurons[i] = newNuron(nextNeuronNum,i);
    }
}

void freeLayer(Layer* layer){

    if(layer){
        for (int i = 0; i < layer->neuronNum; i++)
        {
            freeNuron(layer->neurons[i]);
        }
        free(layer->neurons);
        free(layer);
    }
    
}

typedef struct net
{
    Layer** layers;
}Net;

Net* newNet(int){
    
    
}



float getRadnomF(){
    return (float)rand()/(float)RAND_MAX;
}
int main(){

    //Random comment out below ,if u want random seed
    // srand(time(NULL)); 


    int* topology;
    int layerNum;
    printf("Layer sayisi girin:");
    scanf("%d",&layerNum);
    printf("Topolojiyi girin:");
    topology = (int*)malloc(sizeof(int)*layerNum);

    for (int i = 0; i < layerNum; i++)
    {
        scanf(" %d",&topology[i]);
        // printf(" ");
    }
    printf("\n");

    for (int i = 0; i < layerNum; i++)
    {
        printf("%d ",topology[i]);
    }
    

    // Net* myNet = newNet();


    
    




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
    return 0;
}