#include <malloc.h>
#include <memory.h>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <time.h>

extern "C" {
    #include <cblas.h>
}

using namespace std;

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
};

class Loss {
	public:
		virtual double getLoss(double* w, double* _x, double _y, int featureSize) = 0;
		virtual double* getGradient(double* w, double** _x, double* _y, int featureSize, int sampleSize) = 0;
};

class LinearLoss : public Loss {
    public:
        double getLoss(double* w, double* _x, double _y, int featureSize);
        double* getGradient(double* w, double** _x, double* _y, int featureSize, int sampleSize);
};

double LinearLoss::getLoss(double* w, double* _x, double _y, int featureSize) {
	return _y - cblas_ddot(featureSize, w, 1, _x, 1);
}

double* LinearLoss::getGradient(double* w, double** _x, double* _y, int featureSize, int sampleSize) {
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
	        lastGrad = loss.getGradient(weight, x, y, featureSize, sampleSize);
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

void outputModel(double* model, int size) {
    for(int i = 0; i < size; ++i) {
        printf("%f\t", model[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    double** x;
    double* y;
    double* weight;
//    int featureSize = 100;
//    int sampleSize = 100000;
    int featureSize = atoi(argv[1]);
    int sampleSize = atoi(argv[2]);
    generateData(x, y, weight, featureSize, sampleSize);
//    cout << "init data done." << endl;
    double** xx = (double**) malloc(sizeof(double**));
    double* yy = (double*) malloc(sizeof(double*));
    xx[0] = x[0];
    yy[0] = y[0];
    LinearLoss ll;
	SLBFGS slbfgs(ll, xx, yy, featureSize, 1);
//    cout << "begin learn" << endl;
    bool flag = false;
    while(1){
    for(int i = 0; i < sampleSize; ++i) {
        xx[0] = x[i];
        yy[0] = y[i];
	    if(!slbfgs.learn()) {
            flag = true;
            break;
        }
    }
    if(flag) break;
    }
//    cout << "true model is: " << endl;
//    outputModel(weight, featureSize);
//    cout << "my model is: " << endl;
//    outputModel(slbfgs.weight, featureSize);
    double allLoss = 0;
    for(int i = 0; i < sampleSize; ++i) {
        allLoss += ll.getLoss(slbfgs.weight, x[i], y[i], featureSize);
    }
    cout << "avg loss: " << allLoss / sampleSize << endl;
	return 0;
}
