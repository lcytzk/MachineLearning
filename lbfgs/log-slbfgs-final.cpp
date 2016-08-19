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

extern "C" {
    #include <cblas.h>
}

using namespace std;

void outputModel(double* model, int size) {
    for(int i = 0; i < size; ++i) {
        printf("%f\t", model[i]);
    }
    printf("\n");
}

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
        virtual double getVal(double* w, double* _x, int featureSize) = 0;
		virtual double getLoss(double* w, double* _x, double _y, int featureSize) = 0;
		virtual double* getGradient(double* w, double** _x, double* _y, int featureSize, int sampleSize) = 0;
};

class LogLoss : public Loss {
    public:
        double getVal(double* w, double* _x, int featureSize);
        double getLoss(double* w, double* _x, double _y, int featureSize);
        double* getGradient(double* w, double** _x, double* _y, int featureSize, int sampleSize);
};

double LogLoss::getVal(double* w, double* _x, int featureSize) {
	return 1.0 / (1.0 + exp(0 - cblas_ddot(featureSize, w, 1, _x, 1))) > 0.5 ? 1 : 0;
}

double LogLoss::getLoss(double* w, double* _x, double _y, int featureSize) {
	return _y - 1.0 / (1.0 + exp(0 - cblas_ddot(featureSize, w, 1, _x, 1)));
}

double* LogLoss::getGradient(double* w, double** _x, double* _y, int featureSize, int sampleSize) {
	double* t = (double*) malloc(sizeof(double) * featureSize);
	for (int i = 0; i < featureSize; ++i) {
		double tmp = 0;
		for (int j = 0; j < sampleSize; ++j) {
			tmp += getLoss(w, _x[j], _y[j], featureSize) * _x[j][i];
		}
		t[i] = tmp / sampleSize;
	}
	return t;
}

class SLBFGS {
	private:
		int stopRound;
		float stopGrad;
		Loss& loss;
		int featureSize;
		int sampleSize;
		double* lastGrad;
		double stepSize;
		int m;
		double* alpha;
		double* rho;
		double H = 1.0;
		LoopArray* s;
		LoopArray* t;

		// return gradient.
		void runARound();
		void updateST(double* _s, double* _t);
		double* getDirection(double* q);
	public:
		double* weight;
		double** x;
		double* y;
		SLBFGS(Loss& _loss, double** _x, double* _y, int _featureSize, int _sampleSize): loss(_loss), x(_x), y(_y), featureSize(_featureSize), sampleSize(_sampleSize) {
            m = 10;
			weight = (double*) malloc(sizeof(double) * featureSize);
			memset(weight, 0, sizeof(double) * featureSize);
			alpha = (double*) malloc(sizeof(double) * m);
			rho = (double*) malloc(sizeof(double) * m);
            cout << "get first grad" << endl;
	        lastGrad = loss.getGradient(weight, x, y, featureSize, sampleSize);
            cout << "get first grad done" << endl;
			//lastGrad = (double*) malloc(sizeof(double) * featureSize);
			s = new LoopArray(m);
			t = new LoopArray(m);
			stepSize = 0.1;
			stopGrad = 0.000001;
            stopRound = sampleSize;
		};
		bool learn();
};

bool SLBFGS::learn() {
    free(lastGrad);
    lastGrad = loss.getGradient(weight, x, y, featureSize, sampleSize);
    runARound();
    if(cblas_dnrm2(featureSize, lastGrad, 1) < stopGrad) {
        cout << "Reach the gap" << endl;
        return false;
    }
    return true;
}

void SLBFGS::runARound() {
    double* direction = getDirection(lastGrad);
	//  weight = weight - direction * stepSize;
	cblas_daxpy(featureSize, 0 - stepSize, direction, 1, weight, 1); // weight will be updated here.
	// s = thisW - lastW = -direction * step
	cblas_dscal(featureSize, 0 - stepSize, direction, 1); // direction will be s.
	double* s = direction;
	double* grad = loss.getGradient(weight, x, y, featureSize, sampleSize);
	// grad = grad - lastGrad, lastGrad = lastGrad + grad
	cblas_daxpy(featureSize, -1, lastGrad, 1, grad, 1);
	cblas_daxpy(featureSize, 1, grad, 1, lastGrad, 1);
	// grad = grad - lastGrad grad will be y
	double* y = grad;
	updateST(s, y);
	H = cblas_ddot(featureSize, y, 1, s, 1) / cblas_ddot(featureSize, y, 1, y, 1);
}

double* SLBFGS::getDirection(double* qq) {
	// two loop
	double* q = (double*) malloc(sizeof(double) * featureSize);
	memcpy(q, qq, sizeof(double) * featureSize);
	int k = min(m, s->size());
	for (int i = k-1; i >= 0; --i) {
		rho[i] = 1.0 / cblas_ddot(featureSize, (*s)[i], 1, (*t)[i], 1);
		alpha[i] = cblas_ddot(featureSize, (*s)[i], 1, q, 1) * rho[i];
		// q = q - alpha[i] * t[i];
		cblas_daxpy(featureSize, (0 - alpha[i]), (*t)[i], 1, q, 1);
	}
	// q = H * q;
	cblas_dscal(featureSize, H, q, 1);
	for (int i = 0; i < k; ++i) {
		// double beta = rho[i] * t[i] * q;
		double beta = rho[i] * cblas_ddot(featureSize, q, 1, (*t)[i], 1);
		// q = q + (alpha[i] - beta) * s[i];
		cblas_daxpy(featureSize, alpha[i] - beta, (*s)[i], 1, q, 1);
	}
	return q;
}

void SLBFGS::updateST(double* _s, double* _t) {
	s->appendAndRemoveFirstIfFull(_s);
	t->appendAndRemoveFirstIfFull(_t);
}


void generateData(double** &x, double* &y, double* &weight, int featureSize, int sampleSize) {
    y = (double*) malloc(sizeof(double) * sampleSize);
    x = (double**) malloc(sizeof(double*) * sampleSize);
    weight = (double*) malloc(sizeof(double) * featureSize);
    srand(time(NULL));
    for(int i = 0; i < featureSize; ++i) {
        weight[i] = (double) rand() / RAND_MAX;
    }
    for(int i = 0; i < sampleSize; ++i) {
        double* _x = (double*) malloc(sizeof(double) * featureSize);
        for(int j = 0; j < featureSize; ++j) {
            _x[j] = (double) rand() / RAND_MAX;
        }
        x[i] = _x;
        y[i] = cblas_ddot(featureSize, _x, 1, weight, 1);
    }
}

void generateData2(double** &x, double* &y, double* &weight, int featureSize, int sampleSize) {
    y = (double*) malloc(sizeof(double) * sampleSize);
    x = (double**) malloc(sizeof(double*) * sampleSize);
    weight = (double*) malloc(sizeof(double) * featureSize);
    srand(time(NULL));
    for(int i = 0; i < featureSize; ++i) {
        weight[i] = 1.0;
    }
    for(int i = 0; i < sampleSize; ++i) {
        double* _x = (double*) malloc(sizeof(double) * featureSize);
        for(int j = 0; j < featureSize; ++j) {
            _x[j] = (double) rand() / RAND_MAX > 0.5 ? 1.0 : 0;
        }
        x[i] = _x;
        y[i] = cblas_ddot(featureSize, _x, 1, weight, 1) > 0.5 ? 1.0 : 0;
    }
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

map<string,int> table;

int getIndex(string item) {
    map<string, int>::iterator it = table.find(item);
    if (it == table.end()) {
        table.insert(pair<string, int>(item, table.size() + 1));
    } else {
        return it->second;
    }
    return table.size();
}

bool getNextXY(double* x, double* y, ifstream& fo, vector<string>& v) {
    if(fo.eof()) {
        cout << "reach the file end." << endl;
        return false;
    }
    string str;
    getline(fo, str);
    v.clear();
    splitString(str, v, "\t");
    y[0] = atoi(v[0].c_str()) == 1 ? 1 : 0;
    for(int i = 2; i < v.size(); ++i) {
        x[getIndex(v[i])] = 1.0;
    }
    return true;
}

void outputAcu(SLBFGS& slbfgs, int featureSize, double** xx, double* yy, LogLoss& ll, ifstream& fo) {
    fo.open("/root/liangchenye/mine/MachineLearning/lbfgs/in2-2.txt");
    string str;
    getline(fo, str);
    double allLoss = 0;
    int count = 0;
    int sampleSize = 0;
    vector<string> v;
    while (getNextXY(xx[0], yy, fo, v)) {
        allLoss += ll.getLoss(slbfgs.weight, xx[0], yy[0], featureSize);
        double val = ll.getVal(slbfgs.weight, xx[0], featureSize);
        if(val == yy[0]) {
            ++count;
        }
        ++sampleSize;
    }
    fo.close();
    cout << "avg loss: " << allLoss / sampleSize << endl;
    cout << "count percent: " << ((double) count) / sampleSize << endl;
    cout << "count: " << count << endl;
    cout << "sampleSize: " << sampleSize << endl;
}

void test(int featureSize, ifstream& fo) {
    cout << "feature size is: " << featureSize << endl;
    double** xx = (double**) malloc(sizeof(double*));
    xx[0] = (double*) malloc(sizeof(double) * featureSize);
    memset(xx[0], 0, featureSize * sizeof(double));
    cout << "init xx done." << endl;
    double* yy = (double*) malloc(sizeof(double));
    cout << "init yy done." << endl;
    vector<string> v;
    getNextXY(xx[0], yy, fo, v);
    cout << "get first xy done." << endl;
    LogLoss ll;
	SLBFGS slbfgs(ll, xx, yy, featureSize, 1);
    bool flag = false;
    cout << "begin learn" << endl;
    int start_time = clock();
    int loop = 0;
    while (getNextXY(xx[0], yy, fo, v)) {
        ++loop;
	    if(!slbfgs.learn()) {
            break;
        }
        memset(xx[0], 0, featureSize * sizeof(double));
    }
    printf("Learn finished run %d rounds.", loop);
    cout << "Used time: " << (clock() - start_time)/double(CLOCKS_PER_SEC)*1000 << endl; 
    cout << "learn finish." << endl;
    fo.close();
    outputAcu(slbfgs, featureSize, xx, yy, ll, fo);
//    free(xx[0]);
//    free(xx);
//    free(yy);
}

void loadData(ifstream& fo) {
    fo.open("/root/liangchenye/mine/MachineLearning/lbfgs/in2-2.txt");
}

int getFeatureSize() {
    ifstream fo;
    fo.open("/root/liangchenye/mine/MachineLearning/lbfgs/count.txt");
    string str;
    getline(fo, str);
    return atoi(str.c_str());
}

int main(int argc, char* argv[]) {
    ifstream fo;
    int featureSize = getFeatureSize();
    loadData(fo);
    test(featureSize, fo);
    fo.close();
    cout << "finish program." << endl;
	return 0;
}
