#include <stdlib.h>
//#include <algorithm>
#include <memory.h>
#include <vector>
#include <iostream>
#include <time.h>
#include <math.h>
#include <fstream>
#include <string>
#include <unordered_map>
#include <functional>

using namespace std;

int LEARN_START = 0;
int GLO = 0;
int DIRE = 0;
double DIRE_PH1 = 0;
double  DIRE_PH2 = 0;
double DIRE_PH3 = 0;
double STEP_SIZE = 1;
int SAMPLE_SIZE = 0;    
int ROUND = 1;
int START_TIME, END_TIME;

unordered_map<string,int> table;
hash<string> str_hash;

int W_SIZE = 1 << 21;

void outputModel(double* w, int size) {
    cout << "model" << endl;
    for(int i = 0; i < size; ++i) {
        printf("%f\t", w[i]);
    }
    printf("\n");
}

double dot(double* x, double* y) {
    double res = 0;
    for(int i = 0; i < W_SIZE; ++i) {
        res += x[i] * y[i];
        //cout << "dot res  " << res << endl;
    }
    //cout << "dot res  " << res << endl;
    return res;
}

float norm(double* x) {
    return sqrt(dot(x, x));
}

void scal(double* direction, double stepSize) {
    for(int i = 0; i < W_SIZE; ++i) {
        direction[i] *= stepSize;
    }
}

void ypax(double* y, double alpha, double* x) {
    for(int i = 0; i < W_SIZE; ++i) {
        y[i] += alpha * x[i];
    }
}

double dot(vector<int>& x, double* w) {
    double res = 0;
    for(int i = 0 ; i < x.size(); ++i) {
        res += w[x[i]];
    }
    return res;
}

class Example {
    public:
        double prediction;
        double label;
        vector<int> features;
        Example(vector<int>& _features): features(_features), prediction(0) {}
};

class LoopArray {
	private:
		int start, end;
		int realSize;
		int maxSize;
		double** array;
	public:
		LoopArray(int _size) : maxSize(_size) {
			realSize = 0;
			start = end = 0;
			array = (double**) malloc(sizeof(double*) * _size);
		}
		double* operator[](const int _index) const {
    		return array[(_index + start) % maxSize];
		}
		void appendAndRemoveFirstIfFull(double* element) {
			if (realSize == maxSize) {
				free(array[start]);
				array[start] = element;
				start = (start + 1) % maxSize;
			} else {
				array[realSize++] = element;
			}
		}
		int size() {
			return realSize;
		}
        ~LoopArray() {
            free(array);
        }
};

class Loss {
	public:
        virtual double getLoss(double prediction, double _y) = 0;
		virtual double getLoss(double* w, vector<int>& _x, double _y) = 0;
        virtual double getVal(double* w, vector<int>& _x) = 0;
        virtual double getFirstDeri(double* w, vector<int>& _x, double _y) = 0;
        virtual double getFirstDeri(double prediction, double _y) = 0;
		virtual double* getGradient(double* w, vector<int>& _x, double _y) = 0;
        virtual double* getGradient(double prediction, vector<int>& _x, double _y) = 0;
        virtual void updateGradient(double prediction, vector<int>& _x, double _y, double* t) = 0;
};

class LogLoss : public Loss {
    public:
        double getLoss(double prediction, double _y);
        double getLoss(double* w, vector<int>& _x, double _y);
        double getVal(double* w, vector<int>& _x);
        double getFirstDeri(double* w, vector<int>& _x, double _y);
        double getFirstDeri(double prediction, double _y);
        double* getGradient(double* w, vector<int>& _x, double _y);
        double* getGradient(double prediction, vector<int>& _x, double _y);
        void updateGradient(double prediction, vector<int>& _x, double _y, double* t);
};

double LogLoss::getVal(double* w, vector<int>& _x) {
    return dot(_x, w);
}

double LogLoss::getLoss(double* w, vector<int>& _x, double _y) {
	return log(1 + exp((1 - 2 *_y) * dot(_x, w)));
}


double LogLoss::getLoss(double prediction, double _y) {
    return log(1 + exp((1 - 2 *_y) * prediction));
}

double LogLoss::getFirstDeri(double* w, vector<int>& _x, double _y) {
    return (1 - 2 *_y) / (1.0 + exp((2 *_y - 1) * dot(_x, w)));
}

double LogLoss::getFirstDeri(double prediction, double _y) {
    return (1 - 2 *_y) / (1.0 + exp((2 *_y - 1) * prediction));
}

double* LogLoss::getGradient(double* w, vector<int>& _x, double _y) {
	double* t = (double*) calloc(W_SIZE, sizeof(double));
	for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] = getFirstDeri(w, _x, _y);
	}
	return t;
}

double* LogLoss::getGradient(double prediction, vector<int>& _x, double _y) {
    double* t = (double*) calloc(W_SIZE, sizeof(double));
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] = getFirstDeri(prediction, _y);
    }
    return t;
}

void LogLoss::updateGradient(double prediction, vector<int>& _x, double _y, double* t) {
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] += getFirstDeri(prediction, _y) / SAMPLE_SIZE;
    }
}

class LogLoss2 : public Loss {
    public:
        double getLoss(double prediction, double _y);
        double getLoss(double* w, vector<int>& _x, double _y);
        double getVal(double* w, vector<int>& _x);
        double getFirstDeri(double* w, vector<int>& _x, double _y);
        double getFirstDeri(double prediction, double _y);
        double* getGradient(double* w, vector<int>& _x, double _y);
        double* getGradient(double prediction, vector<int>& _x, double _y);
        void updateGradient(double prediction, vector<int>& _x, double _y, double* t);
};

double LogLoss2::getVal(double* w, vector<int>& _x) {
    return 1 / (1 + exp(-dot(_x, w)));
}

double LogLoss2::getLoss(double* w, vector<int>& _x, double _y) {
    return getLoss(getVal(w, _x), _y);
}

double LogLoss2::getLoss(double prediction, double _y) {
    return 0 - (_y == 1 ? log(prediction) : log(1 - prediction));
}

double LogLoss2::getFirstDeri(double* w, vector<int>& _x, double _y) {
    return (1 - 2 *_y) / (1.0 + exp((2 *_y - 1) * dot(_x, w)));
}

double LogLoss2::getFirstDeri(double prediction, double _y) {
    return (1 - 2 *_y) / (1.0 + exp((2 *_y - 1) * prediction));
}

double* LogLoss2::getGradient(double* w, vector<int>& _x, double _y) {
    double* t = (double*) calloc(W_SIZE, sizeof(double));
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] = getFirstDeri(w, _x, _y);
    }
    return t;
}

double* LogLoss2::getGradient(double prediction, vector<int>& _x, double _y) {
    double* t = (double*) calloc(W_SIZE, sizeof(double));
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] = getFirstDeri(prediction, _y);
    }
    return t;
}

void LogLoss2::updateGradient(double prediction, vector<int>& _x, double _y, double* t) {
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] += (prediction - _y) / SAMPLE_SIZE;
    }
}

class LBFGS {
	private:
		double stopGrad;
		Loss& loss;
        vector<Example*>& examples;
		int m;
        double wolfe1Bound =  0.5;
		double* alpha;
		double* rho;
		double H = 1.0;
		LoopArray* s;
		LoopArray* t;
		void cal_and_save_ST();
		void updateST(double* _s, double* _t);
		double* getDirection(double* q);
        void predict();
        double* getGradient();
        void stepForward();
        void stepBackward();
        bool evalWolfe();
        void updateGradientWithLambda2(double* grad);
        bool hasBack = false;
        double lambda2;
	public:
		double* weight;
        double* lastGrad;
        double* grad;
        double* direction;
		double stepSize;
        double lossSum;
        double preLossSum;
        double lossBound;
		LBFGS(Loss& _loss, vector<Example*>& _examples, double* _weight, double _lambda2, double _lossBound): loss(_loss),  examples(_examples), weight(_weight), lambda2(_lambda2), lossBound(_lossBound) {
            m = 15;
			alpha = (double*) malloc(sizeof(double) * m);
			rho = (double*) malloc(sizeof(double) * m);
			s = new LoopArray(m);
			t = new LoopArray(m);
			stepSize = 0.01;
			stopGrad = 0;
		};
		bool learn();
        void init();
};

bool LBFGS::evalWolfe() {
    //wolfe1 = (loss_sum - previous_loss_sum) / (step_size * g0_d);
    return lossSum - preLossSum  <= 0.0001 * stepSize * dot(lastGrad, direction);
}

void LBFGS::predict() {
    preLossSum = lossSum;
    lossSum = 0;
    double reg = lambda2 * dot(weight, weight);
    for(Example* example : examples) {
        // example->prediction = dot(example->features, weight);
        example->prediction = loss.getVal(weight, example->features);
        double l = loss.getLoss(example->prediction, example->label);
        //if(l > 1) {
        //    printf("ERROR: %f\t %f\n", example->prediction, example->label);
        //    printf("\tERROR: %f\n", dot(example->features, weight));
        //    printf("\t\ttERROR: %f\n", l);
        //}
        lossSum += l;
    }
    lossSum += reg;
}

void LBFGS::updateGradientWithLambda2(double* grad) {
    for(int i = 0; i < W_SIZE; ++i) {
        grad[i] += lambda2 * weight[i];
    }
}

double* LBFGS::getGradient() {
    grad = (double*) calloc(W_SIZE, sizeof(double));
    int count = 0;
    for(Example* example : examples) {
        loss.updateGradient(example->prediction, example->features, example->label, grad);
        //printf("%f\t%f\t%f\n", example->prediction, example->label, grad[18]);
    }
    updateGradientWithLambda2(grad);
    return grad;
}

bool LBFGS::learn() {
    // xi+1 = xi + dire
    stepForward();
    predict();
//    //cout << "lossSum and preLossSum: \t" << lossSum << endl;
    END_TIME = clock(); 
    printf("#%d#\tlossSum: %f\t preLossSum: %f\tstepSize:%f\tused time: %f", ROUND, lossSum/SAMPLE_SIZE, preLossSum/SAMPLE_SIZE, stepSize, (END_TIME - START_TIME)/(double)CLOCKS_PER_SEC);
    START_TIME = END_TIME;
    if(!evalWolfe()) {
        stepBackward();
        printf("\t step back\n");
        stepSize /= 2;
        return true;
    }
    printf("\n");
//    if(wolfe1 == 0 || isnan(wolfe1)) {
//        cout << "wolfe1 is nan  " << wolfe1 << endl;
//        return false;
//    }
//    cout << "wolfe1    " << wolfe1 << endl;
    //if(lossSum > preLossSum || wolfe1 < wolfe1Bound) {
    //if(lossSum > preLossSum) {
    //    stepBackward();
    //    hasBack = true;
    //    stepSize /= 2;
    //    return true;
    //}
    if(isnan(lossSum) || lossSum/SAMPLE_SIZE < lossBound) {
        printf("Reach the bound %f so stop.\n", lossBound);
        return false;
    }
    //if(isnan(lossSum) || lossSum/preLossSum > 0.999) {
    ////if(isnan(lossSum) || lossSum < 0.07) {
    //    printf("Decrease in loss in 0.01%% so stop.\n");
    //    return false;
    //}
    // if(hasBack) {
    //     hasBack = false;
    //     stepSize *= 2;
    // }
    // stepSize =((stepSize + 0.1) > 1) ? stepSize : stepSize + 0.1;
    getGradient();
//    float grad_norm = norm(grad);
//    cout << "norm   " << grad_norm << endl;
//  if(grad_norm < stopGrad && isnan(grad_norm)) {
    //if(lossSum == 0 || grad_norm == 0 || isnan(grad_norm)) {
    //if(lossSum == 0 || isnan(grad_norm)) {
    //    cout << "Reach the gap  " << grad_norm << endl;
    //    return false;
    //}
    cal_and_save_ST();
    getDirection(lastGrad);
    stepSize = STEP_SIZE;
    return true;
}

void LBFGS::init() {
    cout << "lbfgs init" << endl;
    predict();
    lastGrad = getGradient();
    getDirection(lastGrad);
}

void LBFGS::stepForward() {
    for(int i = 0; i < W_SIZE; ++i) {
        weight[i] -= direction[i] * stepSize;
    }
}

void LBFGS::stepBackward() {
    for(int i = 0; i < W_SIZE; ++i) {
        weight[i] += direction[i] * stepSize;
    }
    lossSum = preLossSum;
}

void LBFGS::cal_and_save_ST() {
	//  weight = weight - direction * stepSize;
	// s = thisW - lastW = -direction * step
    scal(direction, 0 - stepSize);
    //direction->scal(stepSize);
	double* ss = direction;
	// grad = grad - lastGrad, lastGrad = lastGrad + grad
    //grad->pax(-1, *lastGrad);
    ypax(grad, -1, lastGrad);
    //lastGrad->pax(1, *grad);
    ypax(lastGrad, 1, grad);
	// grad = grad - lastGrad grad will be y
	double* y = grad;
	updateST(ss, y);
    H = dot(ss, y) / dot(y, y);
}

double* LBFGS::getDirection(double* qq) {
	// two loop
	double* q = (double*) malloc(W_SIZE * sizeof(double));
    memcpy(q, qq, W_SIZE * sizeof(double));
	int k = min(m, s->size());
    if(k > 0) rho[k-1] = 1.0 / dot((*s)[k-1], (*t)[k-1]);
	for (int i = k-1; i >= 0; --i) {
        alpha[i] = dot(q, (*s)[i]) * rho[i];
        ypax(q, 0 - alpha[i], (*t)[i]);
	}
    scal(q, H);
	for (int i = 0; i < k; ++i) {
        double beta = rho[i] * dot(q, (*t)[i]);
        ypax(q, alpha[i] - beta, (*s)[i]);
	}
    // shift rho
    if(k == m) {
        for(int i = 0; i < k-1; ++i) {
            rho[i] = rho[i+1];
        }
    }
    direction = q;
    //outputModel(direction, W_SIZE);
	return q;
}

void LBFGS::updateST(double* _s, double* _t) {
	s->appendAndRemoveFirstIfFull(_s);
	t->appendAndRemoveFirstIfFull(_t);
}

void splitString(string s, vector<string>& output, const char delimiter) {
    size_t start=0;
    size_t end=s.find_first_of(delimiter);
    while (end != string::npos) {
        if(end != start)
            output.push_back(s.substr(start, end-start));
        start = s.find_first_of(delimiter, end) + 1;
        end = s.find_first_of(delimiter, start);
    }
}

int getIndex(string item) {
    //map<string, int>::iterator it = table.find(item);
    return str_hash(item) % W_SIZE;
}

bool getNextXY(vector<int>& x, double& y, ifstream& fo, vector<string>& v) {
    if(fo.eof()) {
        cout << "reach the file end.1" << endl;
        return false;
    }
    string str;
    getline(fo, str);
    if(str.length() < 2) {
        cout << "reach the file end.2" << endl;
        return false;
    }
    v.clear();
    splitString(str, v, ' ');
    x.clear();
    y = (atoi(v[0].c_str()) == 1) ? 1.0 : 0;
    for(int i = 1; i < v.size(); ++i) {
        if(v[i].c_str()[0] != '|') {
            x.push_back(getIndex(v[i]));
        }
    }
    return true;
}

bool getOnlyNextXY(vector<int>& x, double& y, ifstream& fo, vector<string>& v) {
    if(fo.eof()) {
        cout << "reach the file end.1" << endl;
        return false;
    }
    string str;
    getline(fo, str);
    if(str.length() < 2) {
        cout << "reach the file end.2" << endl;
        return false;
    }
    v.clear();
    splitString(str, v, ' ');
    x.clear();
    y = (atoi(v[0].c_str()) == 1) ? 1.0 : 0;
    for(int i = 2; i < v.size(); ++i) {
        auto it = table.find(v[i]);
        //map<string, int>::iterator it = table.find(v[i]);
        if (it != table.end()) {
            x.push_back(it->second);
        }
    }
    return true;
}

void outputAcu(double* weight, Loss& ll, char* testfile) {
    cout << "begin acu" << endl;
    vector<int> xx;
    double yy;
    ifstream fo;
    ofstream fw;
    fo.open(testfile);
    fw.open("my_res");
    double allLoss = 0;
    int count = 0;
    int sampleSize = 0;
    vector<string> v;
    int count1 = 0;
    int count1_shot = 0;
    while (getOnlyNextXY(xx, yy, fo, v)) {
        double loss = ll.getLoss(weight, xx, yy);
        //cout << "prediction  " << ll.getVal(lbfgs.weight, xx) <<endl;
        double prediction1 = ll.getVal(weight, xx);
        double prediction = prediction1 > 0.5 ? 1 : 0;
        allLoss += loss;
        //cout << loss << endl;
        if (prediction == yy) {
            ++count;
        }
        if(yy == 1) {
            ++count1;
            if(prediction == yy) {
                ++count1_shot;
            }
        }
        fw << prediction1 << "," << (yy == 1 ? 1 : -1) << endl;
     //   if (sampleSize % 1000 == 0) cout << sampleSize << endl;
        ++sampleSize;
    }
    fo.close();
    fw.close();
    cout << "avg sum: " << allLoss << endl;
    cout << "avg loss: " << allLoss / sampleSize << endl;
    cout << "count percent: " << ((double) count) / sampleSize << endl;
    cout << "count: " << count << endl;
    cout << "count1: " << count1 << endl;
    cout << "count1_shot: " << count1_shot << endl;
    cout << "sampleSize: " << sampleSize << endl;
   // LBFGS.weight.saveToFile();
}

void loadExamples(vector<Example*>& examples, char* filename) {
    ifstream fo;
    fo.open(filename);
    vector<int> xx;
    double yy;
    vector<string> v;
    while(getNextXY(xx, yy, fo, v)) {
        Example* example = new Example(xx);
        example->label = yy;
        examples.push_back(example);
    }
    fo.close();
}

void outputPredictions(vector<Example*>& examples) {
    cout << "predictions" << endl;
    for(Example* example : examples) {
        printf("%f\t", example->prediction);
    }
    printf("\n");
}

void saveModel(double* weight) {
    ofstream fo;
    fo.open("my.model");
    fo << W_SIZE << endl;
    for(int i = 0; i < W_SIZE; ++i) {
        if(weight[i] != 0)
            fo << weight[i] << endl;
    }
    fo.close();
}

void outputTable() {
    for(auto it = table.begin(); it != table.end(); ++it) {
        cout << it->first << endl;
    }
}

void test(char* filename, double lambda2, double lossBound) {
    cout << "load examples." << endl;
    vector<Example*> examples;
    int start = clock();
    loadExamples(examples, filename);
    cout << "load examples cost " << (clock() - start)/(double)CLOCKS_PER_SEC << endl;
    printf("example size is : %ld\n", examples.size());
    SAMPLE_SIZE = examples.size();
    //printf("table size is : %ld\n", W_SIZE);
//    for(auto it = table.begin(); it != table.end(); ++it) cout << "key\t" << it->first << "\tvalue\t" << it->second << endl;
    double* weight = (double*) calloc(W_SIZE, sizeof(double));
    LogLoss2 ll;
	LBFGS lbfgs(ll, examples, weight, lambda2, lossBound);
    cout << "begin learn" << endl;
    start = clock();
    lbfgs.init();
    int end = clock();
    printf("init used %f\n", (end - start)/(double)CLOCKS_PER_SEC);
    start = end;
    //outputModel(weight, W_SIZE);
    //outputPredictions(examples);
    bool flag = false;
    int LEADN_START = clock();
    for(int i = 0; i < 100000; ++i) {
        START_TIME = clock();
    	if(!lbfgs.learn()) {
            break;
        }
        ++ROUND;
        //outputModel(weight, W_SIZE);
        //outputPredictions(examples);
    }
    //cout << "One pass used time: " << (clock() - start)/double(CLOCKS_PER_SEC) << endl; 
    //outputModel(weight, W_SIZE);
    printf("Learned finished, used %f.\n", (clock() - LEARN_START) / (double) CLOCKS_PER_SEC);
    saveModel(weight);
//    outputAcu(weight, ll, filename);
}

double* loadModel() {
    cout << "begin load model" << endl;
    ifstream fr;
    fr.open("my.model");
    string str;
    getline(fr, str);
    int size = atoi(str.c_str());
    double* weight = (double*) calloc(size, sizeof(double));
    int i = 0;
    vector<string> v;
    while(i++ < size) {
        getline(fr, str);
        v.clear();
        splitString(str, v, ',');
        int index = atoi(v[1].c_str());
        table[v[0]] = index;
        weight[index] = atof(v[2].c_str());
    }
    return weight;
}

int main(int argc, char* argv[]) {
    int start_time = clock();
    char* filename = argv[1];
    string mode(filename);
    if(mode == "load") {
        LogLoss2 ll;
        double* weight = loadModel();
        outputAcu(weight, ll, argv[2]);
    } else {
        double lambda2 = atof(argv[2]);
        double lossBound = atof(argv[3]);
        test(filename, lambda2, lossBound);
        cout << "finish program." << endl;
        cout << "Used time: " << (clock() - start_time)/double(CLOCKS_PER_SEC) << endl; 
    }
	return 0;
}
