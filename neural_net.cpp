#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>


#define PI 3.14159265358979323846
#define ETA 0.001
#define ALPHA 0.9


double sin_deg(double degrees) {
    return sin(degrees *  PI / 180.0);
}

static double randomWeightGlobal(void) { return rand() / double(RAND_MAX); }

using namespace std;


struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer& prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer& nextLayer);
    void updateInputWeights(Layer& prevLayer);
    vector<Connection> m_outputWeights;

private:
    static double eta;
    static double alpha;
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return 2*(rand() / double(RAND_MAX) -0.5); }
    double sumDOW(const Layer& nextLayer) const;
    double m_outputVal;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = ETA;
double Neuron::alpha = ALPHA;


void Neuron::updateInputWeights(Layer& prevLayer)
{

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron& neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient +
            alpha * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;

    }
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
    double sum = 0.0;
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer& nextLayer){
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal){
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x){return tanh(x);}
double Neuron::transferFunctionDerivative(double x){return 1.0 - x * x;}

void Neuron::feedForward(const Layer& prevLayer)
{
    double sum = 0.0;

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
            prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}


class Net
{
public:
    Net(const vector<unsigned>& topology);
    void feedForward(const vector<double>& inputVals);
    void backProp(const vector<double>& targetVals);
    void getResults(vector<double>& resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
private:
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};


double Net::m_recentAverageSmoothingFactor = 100.0;


void Net::getResults(vector<double>& resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const vector<double>& targetVals)
{

    Layer& outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error); // RMS

    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
        / (m_recentAverageSmoothingFactor + 1.0);

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer& hiddenLayer = m_layers[layerNum];
        Layer& nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer& layer = m_layers[layerNum];
        Layer& prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const vector<double>& inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);

    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    //m_layers[0][inputVals.size() - 1].setOutputVal(randomWeightGlobal());

    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer& prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const vector<unsigned>& topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a Neuron!" << endl;
        }

        m_layers.back().back().setOutputVal(1.0);
    }
}


void showVectorVals(string label, vector<double>& v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

int main() {

   srand(static_cast<unsigned int>(time(0)));

//    fstream myFile;
//    myFile.open("weights.txt", ios::in);

//    if (myFile.is_open()) {

//        string line;
//        while (getline(myFile, line))
//        {
//            if (line == "1 10 10 10 1") {
//                cout << "yes";
//            }
//        }

//    }

   vector<unsigned> topology = { 1, 10 ,10 ,1 }; // 1 10 10 10 1
   Net myNet(topology);

   vector<double> inputVals, targetVals, resultVals;
   vector<vector<double>> inputData;
   vector<double> targetData;


   //inputData = {
   //    {0.0, 0.0},
   //    {0.0, 1.0},
   //    {1.0, 0.0},
   //    {1.0, 1.0}
   //};
   //targetData = { 0.0, 1.0, 1.0, 0.0 };

   for (int i = 0; i < 1000; i++)
   {
       double data = (2 * PI) * rand() / (double(RAND_MAX)) - PI;
       inputData.push_back({ data });
       targetData.push_back({ sin(data) });
   }


   cout << "Loading :";
   int counter=0;
   int trainTime = 1000;

   for (int i = 0; i < trainTime; ++i) { // Train for 10,000 epochs

       for (int j = inputData.size()/10; j < inputData.size(); ++j) {
           inputVals = inputData[j];
           targetVals = { targetData[j] };

           myNet.feedForward(inputVals);
           myNet.getResults(resultVals);
           myNet.backProp(targetVals);


           //cout << "Input: ";
           //showVectorVals("", inputVals);
           //cout << "Expected Output: ";
           //showVectorVals("", targetVals);
           //cout << "Actual Output: ";
           //showVectorVals("", resultVals);
           //cout << "----------------------" << endl;
       }
       counter++;
       if (counter > trainTime / 10 ) {
           cout << "*";
           counter = 0;
       }
   }

   // Test after training
   cout << "\nTesting after training:" << endl;
   double topErr = 0;
   for (int j = 0; j < inputData.size() / 10; ++j) {
       double err;
       inputVals = inputData[j];
       targetVals = { targetData[j] };

       myNet.feedForward(inputVals);
       myNet.getResults(resultVals);

       err = (targetVals[0] - resultVals[0]);
       err = err * err;

       topErr += err;
       cout << "Input: ";
       showVectorVals("", (inputVals));
       cout << "Expected Output: ";
       showVectorVals("", targetVals);
       cout << "Actual Output  : ";
       showVectorVals("", resultVals);
       cout << "----------------------" << endl;
   }

//  To see weights
//
//    cout << "--------" << endl;
//    for (int i = 0; i < myNet.m_layers.size()-1; i++)
//    {
//        cout << "-----------Layer" << i << "----------" << endl;
//        for (int j = 0; j < myNet.m_layers[i].size(); j++)
//        {
//            for (int k = 0; k < myNet.m_layers[i][j].m_outputWeights.size(); k++)
//            {
//                cout << myNet.m_layers[i][j].m_outputWeights[k].weight<<"  ";

//            }
//            cout << endl;
//        }
//    }
//    cout << "------------------"<<endl;

   cout << "Top Error :" << topErr<<endl;

   return 0;
}
