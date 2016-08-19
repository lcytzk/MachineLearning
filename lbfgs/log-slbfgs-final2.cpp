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

void outputModel(double* model, int size) {
    for(int i = 0; i < size; ++i) {
        printf("%f\t", model[i]);
    }
    printf("\n");
}

class SparseVector {
    private:
        map<int, double> index2value;
        vector<int> indexs;
    public:
        SparseVector(): indexs(), index2value() {};
        SparseVector(SparseVector& _sv): indexs(_sv.indexs), index2value(_sv.index2value) {};
        double dot(SparseVector& x);
        void pax(double alpha, SparseVector& x);
        double getVal(int index);
        void addItem(int index, double value);
        void clear() { indexs.clear(); index2value.clear(); };
        int size() { return indexs.size(); }
        void scal(double alpha);
};

double SparseVector::dot(SparseVector& x) {
    double res = 0;
    for(int i = 0; i < indexs.size(); ++i) {
        res += index2value[indexs[i]] * x.getVal(indexs[i]);
    }
    return res;
}

void SparseVector::pax(double alpha, SparseVector& x) {
    if (x.size() > size()) {
        for(int i = 0; i < indexs.size(); ++i) {
            index2value[indexs[i]] += alpha * x.getVal(indexs[i]);
        }
    } else {
        for(int i = 0; i < x.indexs.size(); ++i) {
            index2value[x.indexs[i]] += alpha * x.getVal(x.indexs[i]);
        }
    }
}

double SparseVector::getVal(int index) {
    map<int, double>::iterator it = index2value.find(index);
    return it == index2value.end() ? 0 : it->second;
}

void SparseVector::addItem(int index, double value) {
    indexs.push_back(index);
    index2value[index] = value;
}

void SparseVector::scal(double alpha) {
    for(map<int, double>::iterator it = index2value.begin(); it != index2value.end(); ++it) {
        it->second *= alpha;
    }
}

class LoopArray {
	private:
		int start, end;
		int realSize;
		int maxSize;
		SparseVector** array;
	public:
		LoopArray(int _size) : maxSize(_size) {
			realSize = 0;
			start = end = 0;
			array = (SparseVector**) malloc(sizeof(SparseVector*) * _size);
		}
		SparseVector& operator[](const int _index) const {
    		return *array[(_index + start) % maxSize];
		}
		void appendAndRemoveFirstIfFull(SparseVector* element) {
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
        virtual double getVal(SparseVector& w, SparseVector& _x) = 0;
		virtual double getLoss(SparseVector& w, SparseVector& _x, double _y) = 0;
		virtual SparseVector* getGradient(SparseVector& w, SparseVector& _x, double _y) = 0;
};

class LogLoss : public Loss {
    public:
        double getVal(SparseVector& w, SparseVector& _x);
        double getLoss(SparseVector& w, SparseVector& _x, double _y);
        SparseVector* getGradient(SparseVector& w, SparseVector& _x, double _y);
};

double LogLoss::getVal(SparseVector& w, SparseVector& _x) {
	return 1.0 / (1.0 + exp(0 - w.dot(_x))) > 0.5 ? 1 : 0;
}

double LogLoss::getLoss(SparseVector& w, SparseVector& _x, double _y) {
	return _y - getVal(w, _x);
}

SparseVector* LogLoss::getGradient(SparseVector& w, SparseVector& _x, double _y) {
	SparseVector* t = new SparseVector();
	for (int i = 0; i < _x.size(); ++i) {
        if (_x[i] != 0) {
            t->addItem(_x[i], getLoss(w, _x, _y) * _x.getVal(_x[i]));
        }
	}
	return t;
}

class SLBFGS {
	private:
		int stopRound;
		float stopGrad;
		Loss& loss;
		SparseVector* lastGrad;
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
		SparseVector weight;
		SparseVector* x;
		double& y;
		SLBFGS(Loss& _loss, SparseVector& _x, double& _y): loss(_loss), x(_x), y(_y) {
            m = 10;
			weight = SparseVector();
			alpha = (double*) malloc(sizeof(double) * m);
			rho = (double*) malloc(sizeof(double) * m);
			s = new LoopArray(m);
			t = new LoopArray(m);
			stepSize = 0.1;
			stopGrad = -1;
            stopRound = sampleSize;
		};
		bool learn();
};

bool SLBFGS::learn() {
    lastGrad = loss.getGradient(weight, x, y, featureSize, sampleSize);
    runARound();
    //if(cblas_dnrm2(featureSize, lastGrad, 1) < stopGrad) {
    //    cout << "Reach the gap" << endl;
    //    return false;
    //}
    return true;
}

void SLBFGS::runARound() {
    SparseVector* direction = getDirection(lastGrad);
	//  weight = weight - direction * stepSize;
	//cblas_daxpy(featureSize, 0 - stepSize, direction, 1, weight, 1); // weight will be updated here.
    weight.pax(0 - stepSize, direction);
	// s = thisW - lastW = -direction * step
	//cblas_dscal(featureSize, 0 - stepSize, direction, 1); // direction will be s.
    direction->scal(0 - stepSize);
	SparseVector* s = direction;
	SparseVector* grad = loss.getGradient(weight, x, y, featureSize, sampleSize);
	// grad = grad - lastGrad, lastGrad = lastGrad + grad
	//cblas_daxpy(featureSize, -1, lastGrad, 1, grad, 1);
    grad->pax(-1, lastGrad);
	//cblas_daxpy(featureSize, 1, grad, 1, lastGrad, 1);
    lastGrad->pax(1, grad);
	// grad = grad - lastGrad grad will be y
	SparseVector* y = grad;
	updateST(s, y);
	//H = cblas_ddot(featureSize, y, 1, s, 1) / cblas_ddot(featureSize, y, 1, y, 1);
    H = s.dot(y) / y.dot(y);
}

SparseVector* SLBFGS::getDirection(SparseVector& qq) {
	// two loop
	SparseVector* q = new SparseVector(qq);
	int k = min(m, s->size());
	for (int i = k-1; i >= 0; --i) {
		//rho[i] = 1.0 / cblas_ddot(featureSize, (*s)[i], 1, (*t)[i], 1);
        rho[i] = 1.0 / (*s)[i]->dot(*(*t)[i]));
		//alpha[i] = cblas_ddot(featureSize, (*s)[i], 1, q, 1) * rho[i];
        alpha[i] = q->dot(*(*s)[i]) * rho[i];
		// q = q - alpha[i] * t[i];
        //cblas_daxpy(featureSize, (0 - alpha[i]), (*t)[i], 1, q, 1);
        q->pax(0 - alpha[i], *(*t)[i]);
	}
	// q = H * q;
	//cblas_dscal(featureSize, H, q, 1);
    q->scal(H);
	for (int i = 0; i < k; ++i) {
		// double beta = rho[i] * t[i] * q;
		//double beta = rho[i] * cblas_ddot(featureSize, q, 1, (*t)[i], 1);
        double beta = rho[i] * q->dot(*(*t)[i]);
		// q = q + (alpha[i] - beta) * s[i];
		//cblas_daxpy(featureSize, alpha[i] - beta, (*s)[i], 1, q, 1);
        q->pax(alpha[i] - beta, *(*s)[i]);
	}
	return q;
}

void SLBFGS::updateST(SparseVector* _s, SparseVector* _t) {
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

map<string,int> table;

int getIndex(string item) {
    map<string, int>::iterator it = table.find(item);
    if (it == table.end()) {
        table.insert(pair<string, int>(item, table.size()));
    } else {
        return it->second;
    }
    return table.size() - 1;
}

bool getNextXY(SparseVector& x, double& y, ifstream& fo, vector<string>& v) {
    if(fo.eof()) {
        cout << "reach the file end." << endl;
        return false;
    }
    string str;
    getline(fo, str);
    v.clear();
    splitString(str, v, "\t");
    x.clear();
    y = atoi(v[0].c_str()) == 1 ? 1 : 0;
    for(int i = 2; i < v.size(); ++i) {
        x.addItem(getIndex(v[i]), 1.0);
    }
    return true;
}

void outputAcu(SLBFGS& slbfgs, double** xx, double* yy, LogLoss& ll, ifstream& fo) {
    fo.open("/root/liangchenye/mine/MachineLearning/lbfgs/in2-2.txt");
    string str;
    getline(fo, str);
    double allLoss = 0;
    int count = 0;
    int sampleSize = 0;
    vector<string> v;
    while (getNextXY(xx[0], yy, fo, v)) {
        allLoss += ll.getLoss(slbfgs.weight, xx[0], yy[0]);
        double val = ll.getVal(slbfgs.weight, xx[0]);
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

void test(ifstream& fo) {
    SparseVector xx;
    double yy;
    vector<string> v;
    LogLoss ll;
	SLBFGS slbfgs(ll, xx, yy);
    bool flag = false;
    cout << "begin learn" << endl;
    int start_time = clock();
    int loop = 0;
    while (getNextXY(xx, yy, fo, v)) {
        ++loop;
	    if(!slbfgs.learn()) {
            break;
        }
    }
    printf("Learn finished run %d rounds.", loop);
    cout << "Used time: " << (clock() - start_time)/double(CLOCKS_PER_SEC)*1000 << endl; 
    cout << "learn finish." << endl;
    fo.close();
//    outputAcu(slbfgs, xx, yy, ll, fo);
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
    loadData(fo);
    test(fo);
    fo.close();
    cout << "finish program." << endl;
	return 0;
}
