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

class Example {
    public:
        double prediction;
        double label;
        SparseVector features;
        Example(SparseVector& _features): features(_features) {}
};

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

double SparseVector::norm() {
    double res = 0;
    for(int i = 0; i < indexs.size(); ++i) {
        res += pow(index2value[indexs[i]],2);
    }
    return sqrt(res);
}

double SparseVector::dot(SparseVector& x) {
    double res = 0;
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
        virtual double getFirstDeri(SparseVector& w, SparseVector& _x, double _y) = 0;
        virtual double getFirstDeri(double prediction, double _y) = 0;
		virtual SparseVector* getGradient(SparseVector& w, SparseVector& _x, double _y) = 0;
        virtual SparseVector* getGradient(double prediction, SparseVector& _x, double _y) = 0;
        virtual SparseVector* updateGradient(double prediction, SparseVector& _x, double _y, SparseVector& t) = 0;
};

class LogLoss : public Loss {};

double LogLoss::getLoss(SparseVector& w, SparseVector& _x, double _y) {
	return log(1.0 + exp(0 - _y * _x.dot(w)));
}

double LogLoss::getFirstDeri(SparseVector& w, SparseVector& _x, double _y) {
    return -_y / (1.0 + exp(_y * _x.dot(w)))
}

double LogLoss::getFirstDeri(double prediction, double _y) {
    return -_y / (1.0 + exp(_y * prediction))
}

SparseVector* LogLoss::getGradient(SparseVector& w, SparseVector& _x, double _y) {
	SparseVector* t = new SparseVector();
	for (int i = 0; i < _x.size(); ++i) {
        t->addItem(_x[i], getFirstDeri(w, _x, _y) * _x.getVal(_x[i]));
	}
	return t;
}

SparseVector* LogLoss::getGradient(double prediction, SparseVector& _x double _y) {
    SparseVector* t = new SparseVector();
    for (int i = 0; i < _x.size(); ++i) {
        t->addItem(_x[i], getFirstDeri(prediction, _y) * _x.getVal(_x[i]));
    }
    return t;
}

void LogLoss::updateGradient(double prediction, SparseVector& _x double _y, SparseVector& t) {
    for (int i = 0; i < _x.size(); ++i) {
        t->addOrUpdate(_x[i], getFirstDeri(prediction, _y) * _x.getVal(_x[i]));
    }
}

class LBFGS {
	private:
		float stopGrad;
		Loss& loss;
		SparseVector* lastGrad;
		double stepSize;
        vector<Example*>& examples;
		int m;
		double* alpha;
		double* rho;
		double H = 1.0;
		LoopArray* s;
		LoopArray* t;
		void runARound();
		void updateST(SparseVector* _s, SparseVector* _t);
		SparseVector* getDirection(SparseVector& q);
        void predict();
        void getGradient();
	public:
		SparseVector weight;
		LBFGS(Loss& _loss, vector<Example*>& _examples): loss(_loss),  examples(_examples) {
            m = 15;
			weight = SparseVector();
			alpha = (double*) malloc(sizeof(double) * m);
			rho = (double*) malloc(sizeof(double) * m);
			s = new LoopArray(m);
			t = new LoopArray(m);
			stepSize = 0.05;
			stopGrad = 0.000001;
            predict();
		};
		bool learn();
};

void LBFGS::predict() {
    for(Example* example : example) {
        example->prediction = example->features.dot(weight);
    }
}

void LBFGS::getGradient() {
    for(Example* example : example) {
        loss.updateGradient(example->prediction, example->features, example.label, lastGrad);
    }
}

bool LBFGS::learn(Example* example) {
    int start_time = clock();
    predict();
    lastGrad = loss.getGradient(example->prediction, x, y);
    //for(Example* example : examples) {
        runARound(example);
    //}
    free(lastGrad);
    ROUND += (clock() - start_time)/(double)CLOCKS_PER_SEC;
//    if(norm < stopGrad) {
//        cout << "Reach the gap  " << norm << endl;
//        return false;
//    }
    return true;
}

void LBFGS::runARound(Example* example) {
    SparseVector* direction = getDirection(*lastGrad);
	//  weight = weight - direction * stepSize;
	//cblas_daxpy(featureSize, 0 - stepSize, direction, 1, weight, 1); // weight will be updated here.
    weight.pax(stepSize, *direction);
	// s = thisW - lastW = -direction * step
	//cblas_dscal(featureSize, 0 - stepSize, direction, 1); // direction will be s.
    direction->scal(stepSize);
	SparseVector* s = direction;
	SparseVector* grad = loss.getGradient(example->prediction, x, y);
	// grad = grad - lastGrad, lastGrad = lastGrad + grad
	//cblas_daxpy(featureSize, -1, lastGrad, 1, grad, 1);
    grad->pax(-1, *lastGrad);
	//cblas_daxpy(featureSize, 1, grad, 1, lastGrad, 1);
    lastGrad->pax(1, *grad);
	// grad = grad - lastGrad grad will be y
	SparseVector* y = grad;
    //if(y->dot(*s) > 0.0000001) {
	    updateST(s, y);
	    //H = cblas_ddot(featureSize, y, 1, s, 1) / cblas_ddot(featureSize, y, 1, y, 1);
        H = s->dot(*y) / y->dot(*y);
    //}
}

SparseVector* LBFGS::getDirection(SparseVector& qq) {
	// two loop
    int start_time = clock();
	SparseVector* q = new SparseVector(qq);
    int end = clock();
    DIRE_PH1 += (end - start_time)/double(CLOCKS_PER_SEC);
    start_time = end;
	int k = min(m, s->size());
    if(k > 0) rho[0] = 1.0 / (*s)[0].dot((*t)[0]);
	for (int i = k-1; i >= 0; --i) {
		//rho[i] = 1.0 / cblas_ddot(featureSize, (*s)[i], 1, (*t)[i], 1);
        //rho[i] = 1.0 / (*s)[i].dot((*t)[i]);
		//alpha[i] = cblas_ddot(featureSize, (*s)[i], 1, q, 1) * rho[i];
        alpha[i] = q->dot((*s)[i]) * rho[i];
		// q = q - alpha[i] * t[i];
        //cblas_daxpy(featureSize, (0 - alpha[i]), (*t)[i], 1, q, 1);
        q->pax(0 - alpha[i], (*t)[i]);
	}
    end = clock();
    DIRE_PH2 += (end - start_time)/(double)CLOCKS_PER_SEC;
    start_time = end;
	// q = H * q;
	//cblas_dscal(featureSize, H, q, 1);
    q->scal(H);
	for (int i = 0; i < k; ++i) {
		// double beta = rho[i] * t[i] * q;
		//double beta = rho[i] * cblas_ddot(featureSize, q, 1, (*t)[i], 1);
        double beta = rho[i] * q->dot((*t)[i]);
		// q = q + (alpha[i] - beta) * s[i];
		//cblas_daxpy(featureSize, alpha[i] - beta, (*s)[i], 1, q, 1);
        q->pax(alpha[i] - beta, (*s)[i]);
	}
    // shift rho
    for(int i = k-1; i > 0; --i) {
        rho[i] = rho[i-1];
    }
    end = clock();
    DIRE_PH3 += (end - start_time)/(double)CLOCKS_PER_SEC;
    //cout << q->size() << endl;
	return q;
}

void LBFGS::updateST(SparseVector* _s, SparseVector* _t) {
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
    if(str.length() < 2) {
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

void outputAcu(LBFGS& LBFGS, LogLoss& ll) {
    SparseVector xx;
    double yy;
    ifstream fo;
    fo.open("/root/liangchenye/mine/MachineLearning/lbfgs/in2-2.txt");
    double allLoss = 0;
    int count = 0;
    int sampleSize = 0;
    vector<string> v;
    int count1 = 0;
    int count1_shot = 0;
    while (getNextXY(xx, yy, fo, v)) {
        allLoss += ll.getLoss(LBFGS.weight, xx, yy);
        double val = ll.getVal(LBFGS.weight, xx);
        if(yy == 1) {
            ++count1;
        }
        if((val > 0.5 ? 1 : 0) == yy) {
            if (yy == 1) {
                ++count1_shot;
            }
            ++count;
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
    LBFGS.weight.saveToFile();
}

void loadExamples(vector<Example*>& examples, SparseVector& weight) {
    ifstream fo;
    fo.open("/root/liangchenye/mine/MachineLearning/lbfgs/in2-2.txt");
    SparseVector xx;
    double yy;
    vector<string> v;
    while(getNextXY(xx, yy, fo, v)) {
        Example* example = new Example(xx);
        example->prediction = xx.dot(weight);
        example->label = yy;
        examples.push_back(example);
    }
    fo.close();
}

void test() {
    SparseVector weight;
    Example* examples = loadExamples(weight);
    LogLoss ll;
	LBFGS LBFGS(ll, examples);
    bool flag = false;
    cout << "begin learn" << endl;
    int start_time = clock();
    int loop = 0;
//    for(int i = 0; i < 5; ++i) {
        while (getNextXY(xx, yy, fo, v)) {
            ++loop;
            ++GLO;
            if(loop % 1000 == 0) {
                cout << loop << endl;
                printf("DIRE_PH1: %f\t DIRE2: %f\t DIRE3: %f\t ROUND: %f\n", DIRE_PH1, DIRE_PH2, DIRE_PH3, ROUND);
                DIRE = 0;
                ROUND = 0;
                DIRE_PH1 = 0;
                DIRE_PH2 = 0;
                DIRE_PH3 = 0;
            }
	        if(!LBFGS.learn()) {
                flag = true;
                break;
            }
        }
    //    if(flag) break;
  //  }
    printf("Learn finished run %d rounds.", --loop);
    cout << "Used time: " << (clock() - start_time)/double(CLOCKS_PER_SEC)*1000 << endl; 
    outputAcu(LBFGS, ll);
//    free(xx[0]);
//    free(xx);
//    free(yy);
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
    test();
    cout << "finish program." << endl;
	return 0;
}
