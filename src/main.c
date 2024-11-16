#include <stdio.h>
#include <time.h>
#include <stdlib.h>

typedef enum {
    INPUT_LAYER,
    HIDDEN_LAYER,
    OUTPUT_LAYER
} LayerType;


typedef struct N
{
    float* weights; // Neurons Connections
    float m_output_val; // Calculated Value that will be multiplied with weights
    int weightNum;
    int myIndex;
} Neuron;

typedef struct L
{
    int neuronNum;
    Neuron* neurons;
    LayerType type;
    
} Layer;

typedef struct net
{
    Layer* layers;
}Net;

float getRadnomF(){
    return (float)rand()/(float)RAND_MAX;
}
int main(){

    //Random comment out below ,if u want random seed
    // srand(time(NULL)); 

    Net myNet;

    Neuron n1;
    Neuron n2;
    Neuron o1;
    n1.m_output_val=getRadnomF();
    n2.m_output_val=getRadnomF();
    o1.m_output_val=getRadnomF();

    n1.weightNum=1;
    n2.weightNum=1;
    o1.weightNum=1;

    n1.myIndex=0;
    n2.myIndex=1;
    o1.myIndex=0;

    n1.weights = (float*)malloc(1 * sizeof(float));
    n2.weights = (float*)malloc(1 * sizeof(float));
    o1.weights = (float*)malloc(1 * sizeof(float));
    *n1.weights = getRadnomF();
    *n2.weights = getRadnomF();
    *o1.weights = getRadnomF();

    Layer inLayer;
    inLayer.neurons = (Neuron*)malloc(2 * sizeof(Neuron));
    inLayer.neurons[0] = n1;
    inLayer.neurons[1] = n2;
    inLayer.type = INPUT_LAYER;
    inLayer.neuronNum = 2;

    Layer outLayer;
    outLayer.neurons = (Neuron*)malloc(1 * sizeof(Neuron));
    outLayer.neurons[0] = o1;
    outLayer.type = OUTPUT_LAYER;
    outLayer.neuronNum = 1;


    // printf("%f\n",n1.weights[0]);
    // printf("%f\n",n2.weights[0]);
    // printf("%f\n",o1.weights[0]);

    for (int i = 0; i < inLayer.neuronNum; i++)
    {
        printf("W : %f\n",inLayer.neurons[i].weights[0]);
        printf("V : %f\n",inLayer.neurons[i].m_output_val);
        
    }

    for (int i = 0; i < outLayer.neuronNum; i++)
    {
        Neuron* currNeuron = &outLayer.neurons[i];
        Layer* previousLayer = &inLayer;

        float sum = 0.0f;
        for (int i = 0; i < previousLayer->neuronNum; i++)
        {
            sum += previousLayer->neurons[i].weights[currNeuron->myIndex] * previousLayer->neurons[i].m_output_val;
        }
        currNeuron->m_output_val = sum;
        printf("sum : %f\n",sum);
    }
    
    


    for (int i = 0; i < outLayer.neuronNum; i++)
    {
        printf("Output layer %f\n",outLayer.neurons[i].m_output_val);
        
    }


    return 0;
}