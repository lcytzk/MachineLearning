#include <malloc.h>
#include <memory.h>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fstream>
#include <string>
#include <map>

using namespace std;

int GLO = 0;
int DIRE = 0;
double ROUND = 0;
double DIRE_PH1 = 0;
double  DIRE_PH2 = 0;
double DIRE_PH3 = 0;

map<string,int> table;

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

double norm(double* x) {
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
		virtual double getLoss(double* w, vector<int>& _x, double _y) = 0;
        virtual double getFirstDeri(double* w, vector<int>& _x, double _y) = 0;
        virtual double getFirstDeri(double prediction, double _y) = 0;
		virtual double* getGradient(double* w, vector<int>& _x, double _y) = 0;
        virtual double* getGradient(double prediction, vector<int>& _x, double _y) = 0;
        virtual void updateGradient(double prediction, vector<int>& _x, double _y, double* t) = 0;
};

class LogLoss : public Loss {
    public:
        double getLoss(double* w, vector<int>& _x, double _y);
        double getFirstDeri(double* w, vector<int>& _x, double _y);
        double getFirstDeri(double prediction, double _y);
        double* getGradient(double* w, vector<int>& _x, double _y);
        double* getGradient(double prediction, vector<int>& _x, double _y);
        void updateGradient(double prediction, vector<int>& _x, double _y, double* t);
};

double LogLoss::getLoss(double* w, vector<int>& _x, double _y) {
	return log(1.0 + exp((1 - 2 *_y) * dot(_x, w)));
}

double LogLoss::getFirstDeri(double* w, vector<int>& _x, double _y) {
    return (1 - 2 *_y) / (1.0 + exp((1 - 2 *_y) * dot(_x, w)));
}

double LogLoss::getFirstDeri(double prediction, double _y) {
    return (1 - 2 *_y) / (1.0 + exp((1 - 2 *_y) * prediction));
}

double* LogLoss::getGradient(double* w, vector<int>& _x, double _y) {
	double* t = (double*) malloc(sizeof(double) * table.size());
	for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] = getFirstDeri(w, _x, _y);
	}
	return t;
}

double* LogLoss::getGradient(double prediction, vector<int>& _x, double _y) {
    double* t = (double*) malloc(sizeof(double) * table.size());
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
        double getLoss(double* w, vector<int>& _x, double _y);
        double getLoss(double prediction, double _y);
        double getFirstDeri(double* w, vector<int>& _x, double _y);
        double getFirstDeri(double prediction, double _y);
        double* getGradient(double* w, vector<int>& _x, double _y);
        double* getGradient(double prediction, vector<int>& _x, double _y);
        void updateGradient(double prediction, vector<int>& _x, double _y, double* t);
};

double LogLoss2::getLoss(double* w, vector<int>& _x, double _y) {
    return _y - 1.0 / (1.0 + exp(0 - dot(_x, w)));
}

double LogLoss2::getLoss(double prediction, double _y) {
    return _y - 1.0 / (1.0 + exp(0 - prediction));
}

double LogLoss2::getFirstDeri(double* w, vector<int>& _x, double _y) {
    return (1 - 2 *_y) / (1.0 + exp((1 - 2 *_y) * dot(_x, w)));
}

double LogLoss2::getFirstDeri(double prediction, double _y) {
    return (1 - 2 *_y) / (1.0 + exp((1 - 2 *_y) * prediction));
}

double* LogLoss2::getGradient(double* w, vector<int>& _x, double _y) {
    double* t = (double*) malloc(sizeof(double) * table.size());
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] = getFirstDeri(w, _x, _y);
    }
    return t;
}

double* LogLoss2::getGradient(double prediction, vector<int>& _x, double _y) {
    double* t = (double*) malloc(sizeof(double) * table.size());
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] = getFirstDeri(prediction, _y);
    }
    return t;
}

void LogLoss2::updateGradient(double prediction, vector<int>& _x, double _y, double* t) {
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] += getLoss(prediction, _y) / table.size();
    }
}

class LBFGS {
	private:
		double stopGrad;
		Loss& loss;
		double stepSize;
        vector<Example*>& examples;
		int m;
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
	public:
		double* weight;
        double* lastGrad;
        double* grad;
        double* direction;
		LBFGS(Loss& _loss, vector<Example*>& _examples, double* _weight): loss(_loss),  examples(_examples), weight(_weight) {
            m = 15;
			alpha = (double*) malloc(sizeof(double) * m);
			rho = (double*) malloc(sizeof(double) * m);
			s = new LoopArray(m);
			t = new LoopArray(m);
			stepSize = 0.05;
			stopGrad = 0.0001;
		};
		bool learn();
        void init();
};

void LBFGS::predict() {
    for(Example* example : examples) {
        example->prediction = dot(example->features, weight);
    }
}

double* LBFGS::getGradient() {
    grad = (double*) calloc(table.size(), sizeof(double));
    int count = 0;
    for(Example* example : examples) {
        loss.updateGradient(example->prediction, example->features, example->label, grad);
        //printf("%f\t%f\t%f\n", example->prediction, example->label, grad[18]);
    }
    return grad;
}

bool LBFGS::learn() {
    // xi+1 = xi + dire
    stepForward();
    predict();
    cal_and_save_ST();
    getDirection(lastGrad);
    double grad_norm = norm(lastGrad);
    if(grad_norm < stopGrad) {
        cout << "Reach the gap  " << norm << endl;
        return false;
    }
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

void LBFGS::cal_and_save_ST() {
	//  weight = weight - direction * stepSize;
	// s = thisW - lastW = -direction * step
    scal(direction, 0 - stepSize);
    //direction->scal(stepSize);
	double* ss = direction;
	grad = getGradient();
	// grad = grad - lastGrad, lastGrad = lastGrad + grad
    //grad->pax(-1, *lastGrad);
    ypax(grad, -1, lastGrad);
    //lastGrad->pax(1, *grad);
    ypax(lastGrad, 1, grad);
	// grad = grad - lastGrad grad will be y
	double* y = grad;
	updateST(ss, y);
    //H = s->dot(*y) / y->dot(*y);
    //outputModel(y, table.size());
    //cout << "y dot y   " << dot(y,y) << endl;
    //cout << "ss dot y   " << dot(ss,y) << endl;
    H = dot(ss, y) / dot(y, y);
    //cout << "HHHHHHHHHHH   " << H << endl;
}

double* LBFGS::getDirection(double* qq) {
	// two loop
	double* q = (double*) malloc(table.size() * sizeof(double));
    memcpy(q, qq, table.size() * sizeof(double));
    //outputModel(q, table.size());
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

void splitString(const std::string& s, std::vector<std::string>& v, const std::string& c) {
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while(std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2-pos1));
 
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if(pos1 != s.length())
    v.push_back(s.substr(pos1));
}

int getIndex(string item) {
    map<string, int>::iterator it = table.find(item);
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
    splitString(str, v, " ");
    x.clear();
    y = (atoi(v[0].c_str()) == 1) ? 1.0 : 0;
    for(int i = 2; i < v.size(); ++i) {
        x.push_back(getIndex(v[i]));
    }
    return true;
}

void outputAcu(LBFGS& LBFGS, Loss& ll) {
    vector<int> xx;
    double yy;
    ifstream fo;
    fo.open("in2-2.txt");
    double allLoss = 0;
    int count = 0;
    int sampleSize = 0;
    vector<string> v;
    int count1 = 0;
    int count1_shot = 0;
    while (getNextXY(xx, yy, fo, v)) {
        allLoss += ll.getLoss(LBFGS.weight, xx, yy);
        if(yy == 1) {
            ++count1;
        }
        ++sampleSize;
    }
    fo.close();
    cout << "avg loss: " << allLoss / sampleSize << endl;
    cout << "count percent: " << ((double) count) / sampleSize << endl;
    cout << "count: " << count << endl;
    cout << "count1: " << count1 << endl;
    cout << "count1_shot: " << count1_shot << endl;
    cout << "sampleSize: " << sampleSize << endl;
   // LBFGS.weight.saveToFile();
}

void loadExamples(vector<Example*>& examples) {
    ifstream fo;
    fo.open("in2-2.txt");
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

void test() {
    cout << "load examples." << endl;
    vector<Example*> examples;
    int start = clock();
    loadExamples(examples);
    cout << "load examples cost " << (clock() - start)/(double)CLOCKS_PER_SEC << endl;
    printf("example size is : %ld\n", examples.size());
    printf("table size is : %ld\n", table.size());
    double* weight = (double*) calloc(table.size(), sizeof(double));
    LogLoss2 ll;
	LBFGS lbfgs(ll, examples, weight);
    cout << "begin learn" << endl;
    int start_time = clock();
    lbfgs.init();
    //outputModel(weight, table.size());
    //outputPredictions(examples);
    for(int i = 0; i < 100000; ++i) {
    	if(!lbfgs.learn()) {
            break;
        }
        //outputModel(weight, table.size());
        //outputPredictions(examples);
    }
    cout << "NORM: " << norm(lbfgs.lastGrad) << endl; 
    cout << "One pass used time: " << (clock() - start_time)/double(CLOCKS_PER_SEC) << endl; 
    outputModel(weight, table.size());
    outputAcu(lbfgs, ll);
}

int main(int argc, char* argv[]) {
    int start_time = clock();
    test();
    cout << "finish program." << endl;
    cout << "Used time: " << (clock() - start_time)/double(CLOCKS_PER_SEC) << endl; 
	return 0;
}
