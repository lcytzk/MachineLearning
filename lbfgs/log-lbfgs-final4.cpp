#include <stdlib.h>
#include <memory.h>
#include <vector>
#include <iostream>
#include <time.h>
#include <math.h>
#include <fstream>
#include <string>
#include <map>
#include <unordered_map>

using namespace std;

int GLO = 0;
int DIRE = 0;
double ROUND = 0;
double DIRE_PH1 = 0;
double  DIRE_PH2 = 0;
double DIRE_PH3 = 0;
double STEP_SIZE = 1;

unordered_map<string,int> table;

void outputModel(double* w, int size) {
    cout << "model" << endl;
    for(int i = 0; i < size; ++i) {
        printf("%f\t", w[i]);
    }
    printf("\n");
}

double dot(double* x, double* y) {
    double res = 0;
    for(int i = 0; i < table.size(); ++i) {
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
    for(int i = 0; i < table.size(); ++i) {
        direction[i] *= stepSize;
    }
}

void ypax(double* y, double alpha, double* x) {
    for(int i = 0; i < table.size(); ++i) {
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
	double* t = (double*) calloc(table.size(), sizeof(double));
	for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] = getFirstDeri(w, _x, _y);
	}
	return t;
}

double* LogLoss::getGradient(double prediction, vector<int>& _x, double _y) {
    double* t = (double*) calloc(table.size(), sizeof(double));
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] = getFirstDeri(prediction, _y);
    }
    return t;
}

void LogLoss::updateGradient(double prediction, vector<int>& _x, double _y, double* t) {
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] += getFirstDeri(prediction, _y) / table.size();
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
    double* t = (double*) calloc(table.size(), sizeof(double));
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] = getFirstDeri(w, _x, _y);
    }
    return t;
}

double* LogLoss2::getGradient(double prediction, vector<int>& _x, double _y) {
    double* t = (double*) calloc(table.size(), sizeof(double));
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] = getFirstDeri(prediction, _y);
    }
    return t;
}

void LogLoss2::updateGradient(double prediction, vector<int>& _x, double _y, double* t) {
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] += (prediction - _y) / table.size();
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
        double evalWolfe();
        void updateGradientWithLambda2(double* grad);
        bool hasBack = false;
        double lambda2 = 0;
	public:
		double* weight;
        double* lastGrad;
        double* grad;
        double* direction;
		double stepSize;
        double lossSum;
        double preLossSum;
		LBFGS(Loss& _loss, vector<Example*>& _examples, double* _weight): loss(_loss),  examples(_examples), weight(_weight) {
            m = 15;
			alpha = (double*) malloc(sizeof(double) * m);
			rho = (double*) malloc(sizeof(double) * m);
			s = new LoopArray(m);
			t = new LoopArray(m);
			stepSize = STEP_SIZE;
			stopGrad = 0;
		};
		bool learn();
        void init();
};

double LBFGS::evalWolfe() {
    //wolfe1 = (loss_sum - previous_loss_sum) / (step_size * g0_d);
    double wolfe1 = (lossSum - preLossSum) / (stepSize * dot(lastGrad, direction));
    stepSize = (lossSum - preLossSum) / (wolfe1Bound * dot(lastGrad, direction));
    return wolfe1 > 0 ? wolfe1 : -wolfe1;
}

void LBFGS::predict() {
    preLossSum = lossSum;
    lossSum = 0;
    double reg = 0.5 * lambda2 * norm(weight) / table.size();
    for(Example* example : examples) {
        // example->prediction = dot(example->features, weight);
        example->prediction = loss.getVal(weight, example->features);
        double l = loss.getLoss(example->prediction, example->label);
        if(l > 1) {
            printf("ERROR: %f\t %f\n", example->prediction, example->label);
            printf("\tERROR: %f\n", dot(example->features, weight));
            printf("\t\ttERROR: %f\n", l);
        }
        lossSum += l;
    }
    lossSum += reg;
}

void LBFGS::updateGradientWithLambda2(double* grad) {
    for(int i = 0; i < table.size(); ++i) {
        grad[i] += lambda2 * weight[i] / table.size();
    }
}

double* LBFGS::getGradient() {
    grad = (double*) calloc(table.size(), sizeof(double));
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
//    double wolfe1 = evalWolfe();
//    //cout << "lossSum and preLossSum: \t" << lossSum << endl;
    printf("lossSum: %f\t preLossSum: %f  \tdirMag: %f\n", lossSum/table.size(), preLossSum/table.size(), norm(direction));
//    if(wolfe1 == 0 || isnan(wolfe1)) {
//        cout << "wolfe1 is nan  " << wolfe1 << endl;
//        return false;
//    }
//    cout << "wolfe1    " << wolfe1 << endl;
    //if(lossSum > preLossSum || wolfe1 < wolfe1Bound) {
    if(lossSum > preLossSum) {
        stepBackward();
        hasBack = true;
        stepSize /= 2;
        return true;
    }
    if(lossSum/preLossSum > 0.999) {
        printf("Decrease in loss in 0.1%% so stop.\n");
        return false;
    }
    // if(hasBack) {
    //     hasBack = false;
    //     stepSize *= 2;
    // }
    // stepSize =((stepSize + 0.1) > 1) ? stepSize : stepSize + 0.1;
    getGradient();
    float grad_norm = norm(grad);
    cout << "norm   " << grad_norm << endl;
//  if(grad_norm < stopGrad && isnan(grad_norm)) {
    if(lossSum == 0 || grad_norm == 0 || isnan(grad_norm)) {
        cout << "Reach the gap  " << grad_norm << endl;
        return false;
    }
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
    for(int i = 0; i < table.size(); ++i) {
        weight[i] -= direction[i] * stepSize;
    }
}

void LBFGS::stepBackward() {
    cout << "step back" << endl;
    for(int i = 0; i < table.size(); ++i) {
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
	double* q = (double*) malloc(table.size() * sizeof(double));
    memcpy(q, qq, table.size() * sizeof(double));
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
    //outputModel(direction, table.size());
	return q;
}

void LBFGS::updateST(double* _s, double* _t) {
	s->appendAndRemoveFirstIfFull(_s);
	t->appendAndRemoveFirstIfFull(_t);
}

void splitString(string s, vector<string>& output, const char delimiter)
{
    size_t start=0;
    size_t end=s.find_first_of(delimiter);
    
    while (end <= string::npos)
    {
        output.push_back(s.substr(start, end-start));

        if (end == string::npos)
            break;

        start=end+1;
        end = s.find_first_of(delimiter, start);
    }
}

int getIndex(string item) {
    //map<string, int>::iterator it = table.find(item);
    auto it = table.find(item);
    if (it == table.end()) {
        table.insert(pair<string, int>(item, table.size()));
    } else {
        return it->second;
    }
    return table.size() - 1;
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
    for(int i = 2; i < v.size(); ++i) {
        x.push_back(getIndex(v[i]));
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
    fo << table.size() << endl;
    for(auto it = table.begin(); it != table.end(); ++it) {
        fo << it->first << ","  << it->second << "," << weight[it->second] << endl;
    }
    fo.close();
}

void test(char* filename) {
    cout << "load examples." << endl;
    vector<Example*> examples;
    int start = clock();
    loadExamples(examples, filename);
    cout << "load examples cost " << (clock() - start)/(double)CLOCKS_PER_SEC << endl;
    printf("example size is : %ld\n", examples.size());
    printf("table size is : %ld\n", table.size());
    double* weight = (double*) calloc(table.size(), sizeof(double));
    LogLoss2 ll;
	LBFGS lbfgs(ll, examples, weight);
    cout << "begin learn" << endl;
    start = clock();
    lbfgs.init();
    int end = clock();
    printf("init used %f\n", (end - start)/(double)CLOCKS_PER_SEC);
    start = end;
    //outputModel(weight, table.size());
    //outputPredictions(examples);
    bool flag = false;
    for(int i = 0; i < 100000; ++i) {
    	if(!lbfgs.learn()) {
            flag = true;
        }
        if (flag) {
            printf("Round %d. Stop here.\n", i+1);
            break;
        }
        //outputModel(weight, table.size());
        //outputPredictions(examples);
    }
    //cout << "One pass used time: " << (clock() - start)/double(CLOCKS_PER_SEC) << endl; 
    //outputModel(weight, table.size());
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
        test(filename);
        cout << "finish program." << endl;
        cout << "Used time: " << (clock() - start_time)/double(CLOCKS_PER_SEC) << endl; 
    }
	return 0;
}
