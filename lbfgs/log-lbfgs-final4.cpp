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

double dot(double* x, double* y) {
    double res = 0;
    for(int i = 0; i < table.size(); ++i) {
        res += x[i] * y[i];
    }
    return res;
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

class SparseVector {
    public:
        map<int, double> index2value;
        vector<int> indexs;
        SparseVector(): indexs(), index2value() {};
        SparseVector(SparseVector& _sv): indexs(_sv.indexs), index2value(_sv.index2value) {};
        double dot(SparseVector& x);
        void pax(double alpha, SparseVector& x);
        double getVal(int index);
        void addItem(int index, double value);
        void addOrUpdate(int index, double value);
        void clear() { indexs.clear(); index2value.clear(); };
        int size() { return indexs.size(); }
        void scal(double alpha);
        void output() {
            for(int i = 0; i < indexs.size(); ++i) {
                printf("index1:%d\tindex2:%d\tvalue:%f\n", i, indexs[i], index2value[indexs[i]]);
            }
        }
        void saveToFile() {
            ofstream fw;
            fw.open("/root/liangchenye/res.txt");
            for(int i = 0; i < indexs.size(); ++i) {
                fw << indexs[i] << '\t' << index2value[indexs[i]] << endl;
            }
            fw.close();
        }
        int operator[](const int i) const {
            return indexs[i];
        }
        double norm();
};

class Example {
    public:
        double prediction;
        double label;
        SparseVector features;
        Example(SparseVector& _features): features(_features), prediction(0) {}
};

double SparseVector::norm() {
    double res = 0;
    for(int i = 0; i < indexs.size(); ++i) {
        res += pow(index2value[indexs[i]],2);
    }
    return sqrt(res);
}

double SparseVector::dot(SparseVector& x) {
    double res = 0;
    if (size() > x.size()) {
        return x.dot(*this);
    }
    for(int i = 0; i < indexs.size(); ++i) {
        res += index2value[indexs[i]] * x.getVal(indexs[i]);
    }
//    cout << "dot  " << res << endl;
    return res;
}

void SparseVector::pax(double alpha, SparseVector& x) {
    //cout << alpha << endl;
//    x.output();
    for(int i = 0; i < x.indexs.size(); ++i) {
        if(index2value.find(x.indexs[i]) != index2value.end()) {
            index2value[x.indexs[i]] += alpha * x.getVal(x.indexs[i]);
        } else {
            addItem(x.indexs[i], alpha * x.getVal(x.indexs[i]));
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

void SparseVector::addOrUpdate(int index, double value) {
    if(index2value.find(index) != index2value.end()) {
        index2value[index] += value;
    } else {
        addItem(index, value);
    }
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
		virtual double getLoss(SparseVector& w, SparseVector& _x, double _y) = 0;
        virtual double getFirstDeri(SparseVector& w, SparseVector& _x, double _y) = 0;
        virtual double getFirstDeri(double prediction, double _y) = 0;
		virtual SparseVector* getGradient(SparseVector& w, SparseVector& _x, double _y) = 0;
        virtual SparseVector* getGradient(double prediction, SparseVector& _x, double _y) = 0;
        virtual void updateGradient(double prediction, SparseVector& _x, double _y, double* t) = 0;
};

class LogLoss : public Loss {
public:
    double getLoss(SparseVector& w, SparseVector& _x, double _y);
    double getFirstDeri(SparseVector& w, SparseVector& _x, double _y);
    double getFirstDeri(double prediction, double _y);
    SparseVector* getGradient(SparseVector& w, SparseVector& _x, double _y);
    SparseVector* getGradient(double prediction, SparseVector& _x, double _y);
    void updateGradient(double prediction, SparseVector& _x, double _y, double* t);
};

double LogLoss::getLoss(SparseVector& w, SparseVector& _x, double _y) {
	return log(1.0 + exp((1 - 2 *_y) * _x.dot(w)));
}

double LogLoss::getFirstDeri(SparseVector& w, SparseVector& _x, double _y) {
    return (1 - 2 *_y) / (1.0 + exp((1 - 2 *_y) * _x.dot(w)));
}

double LogLoss::getFirstDeri(double prediction, double _y) {
    return (1 - 2 *_y) / (1.0 + exp((1 - 2 *_y) * prediction));
}

SparseVector* LogLoss::getGradient(SparseVector& w, SparseVector& _x, double _y) {
	SparseVector* t = new SparseVector();
	for (int i = 0; i < _x.size(); ++i) {
        t->addItem(_x[i], getFirstDeri(w, _x, _y));
	}
	return t;
}

SparseVector* LogLoss::getGradient(double prediction, SparseVector& _x, double _y) {
    SparseVector* t = new SparseVector();
    for (int i = 0; i < _x.size(); ++i) {
        t->addItem(_x[i], getFirstDeri(prediction, _y));
    }
    return t;
}

void LogLoss::updateGradient(double prediction, SparseVector& _x, double _y, double* t) {
    for (int i = 0; i < _x.size(); ++i) {
        t[_x[i]] += getFirstDeri(prediction, _y);
    }
}

class LBFGS {
	private:
		float stopGrad;
		Loss& loss;
		double* lastGrad;
        double* grad;
        double* direction;
		double stepSize;
        vector<Example*>& examples;
		int m;
		double* alpha;
		double* rho;
		double H = 1.0;
		LoopArray* s;
		LoopArray* t;
		void runARound();
		void updateST(double* _s, double* _t);
		double* getDirection(double* q);
        void predict();
        double* getGradient();
        void stepForward();
	public:
		double* weight;
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
        for(int i = 0; i < example->features.size(); ++i) {
            example->prediction += weight[example->features.indexs[i]];
        }
    }
}

double* LBFGS::getGradient() {
    grad = (double*) malloc(sizeof(double) * table.size());
    for(Example* example : examples) {
        loss.updateGradient(example->prediction, example->features, example->label, grad);
    }
    return grad;
}

bool LBFGS::learn() {
    //int start_time = clock();
    //lastGrad = loss.getGradient(example->prediction, x, y);
    int start = clock();
        stepForward();
    int end = clock();
    printf("step forward used: %f\n", (end - start)/(double) CLOCKS_PER_SEC);
    start = end;
        predict();
    end = clock();
    printf("predict used: %f\n", (end - start)/(double) CLOCKS_PER_SEC);
    start = end;
        getDirection(lastGrad);
    end = clock();
    printf("Get direction used: %f\n", (end - start)/(double) CLOCKS_PER_SEC);
    start = end;
        runARound();
    end = clock();
    printf("run a round used: %f\n", (end - start)/(double) CLOCKS_PER_SEC);
    //}
    //free(lastGrad);
//    double norm = lastGrad->norm();
  //  if(norm < stopGrad) {
    //    cout << "Reach the gap  " << norm << endl;
      //  return false;
    //}
    return true;
}

void LBFGS::init() {
    cout << "lbfgs init" << endl;
    predict();
    lastGrad = getGradient();
    getDirection(lastGrad);
}

void LBFGS::stepForward() {
//    printf("weight: %d\tdirec:%d\n", weight.size(), direction->size());
    for(int i =0; i < table.size(); ++i) {
        weight[i] += direction[i] * stepSize;
    }
}

void LBFGS::runARound() {
    //direction = getDirection(*lastGrad);
	//  weight = weight - direction * stepSize;
    //weight.pax(stepSize, *direction);
	// s = thisW - lastW = -direction * step
    scal(direction, stepSize);
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
    H = dot(ss, y) / dot(y, y);
}

double* LBFGS::getDirection(double* qq) {
	// two loop
	double* q = (double*) malloc(sizeof(double) * table.size());
    memcpy(q, qq, sizeof(double) * table.size());
	int k = min(m, s->size());
    //if(k > 0) rho[k-1] = 1.0 / (*s)[k-1].dot((*t)[k-1]);
    if(k > 0) rho[k-1] = 1.0 / dot((*s)[k-1], (*t)[k-1]);
	for (int i = k-1; i >= 0; --i) {
        //rho[i] = 1.0 / (*s)[i].dot((*t)[i]);
        //alpha[i] = q->dot((*s)[i]) * rho[i];
        alpha[i] = dot(q, (*s)[i]) * rho[i];
		// q = q - alpha[i] * t[i];
        //q->pax(0 - alpha[i], (*t)[i]);
        ypax(q, alpha[i], (*t)[i]);
	}
	// q = H * q;
    //q->scal(H);
    scal(q, H);
	for (int i = 0; i < k; ++i) {
		// double beta = rho[i] * t[i] * q;
        //double beta = rho[i] * q->dot((*t)[i]);
        double beta = rho[i] * dot(q, (*t)[i]);
		// q = q + (alpha[i] - beta) * s[i];
        //q->pax(alpha[i] - beta, (*s)[i]);
        ypax(q, alpha[i] - beta, (*s)[i]);
	}
    // shift rho
    for(int i = 0; i < k-1; ++i) {
        rho[i] = rho[i+1];
    }
    direction = q;
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

bool getNextXY(SparseVector& x, double& y, ifstream& fo, vector<string>& v) {
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
        x.addItem(getIndex(v[i]), 1.0);
    }
    return true;
}

// void outputAcu(LBFGS& LBFGS, LogLoss& ll) {
//     SparseVector xx;
//     double yy;
//     ifstream fo;
//     fo.open("/root/liangchenye/mine/MachineLearning/lbfgs/in2-2.txt");
//     double allLoss = 0;
//     int count = 0;
//     int sampleSize = 0;
//     vector<string> v;
//     int count1 = 0;
//     int count1_shot = 0;
//     while (getNextXY(xx, yy, fo, v)) {
//         allLoss += ll.getLoss(LBFGS.weight, xx, yy);
//         double val = 0;// ll.getVal(LBFGS.weight, xx);
//         if(yy == 1) {
//             ++count1;
//         }
//         if((val > 0.5 ? 1 : 0) == yy) {
//             if (yy == 1) {
//                 ++count1_shot;
//             }
//             ++count;
//         }
//         ++sampleSize;
//     }
//     fo.close();
//     cout << "avg loss: " << allLoss / sampleSize << endl;
//     cout << "count percent: " << ((double) count) / sampleSize << endl;
//     cout << "count: " << count << endl;
//     cout << "count1: " << count1 << endl;
//     cout << "count1_shot: " << count1_shot << endl;
//     cout << "sampleSize: " << sampleSize << endl;
//    // LBFGS.weight.saveToFile();
// }

void loadExamples(vector<Example*>& examples) {
    ifstream fo;
    fo.open("in2-2.txt");
    SparseVector xx;
    double yy;
    vector<string> v;
    while(getNextXY(xx, yy, fo, v)) {
        Example* example = new Example(xx);
        example->label = yy;
        examples.push_back(example);
    }
    fo.close();
}

void test() {
    cout << "load examples." << endl;
    vector<Example*> examples;
    int start = clock();
    loadExamples(examples);
    cout << "load examples cost " << (clock() - start)/(double)CLOCKS_PER_SEC << endl;
    printf("example size is : %ld\n", examples.size());
    printf("table size is : %ld\n", table.size());
    double* weight = (double*) malloc(sizeof(double) * table.size());
    LogLoss ll;
	LBFGS lbfgs(ll, examples, weight);
    cout << "begin learn" << endl;
    int start_time = clock();
    lbfgs.init();
    for(int i = 0; i < 20; ++i) {
    	lbfgs.learn();
        cout << endl;
    }
    cout << "One pass used time: " << (clock() - start_time)/double(CLOCKS_PER_SEC) << endl; 
    //outputAcu(LBFGS, ll);
}

void test2() {
    SparseVector sv;
    SparseVector sv2;
    for(int i = 0; i < 10; ++i) {
        sv.addItem(i*5, 2.0);
    }
    for(int i = 0; i < 5; ++i) {
        sv2.addItem(i*4, 3.0);
    }
    sv.output();
    sv2.output();
    cout << sv.dot(sv2) << endl;
    sv.pax(1, sv2);
    sv.output();
}

int main(int argc, char* argv[]) {
    int start_time = clock();
    test();
    cout << "finish program." << endl;
    cout << "Used time: " << (clock() - start_time)/double(CLOCKS_PER_SEC) << endl; 
	return 0;
}
