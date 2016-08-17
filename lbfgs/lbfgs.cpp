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
		virtual double getLoss(double* w, double* _x, double _y) = 0;
		virtual double* getGradient(double* w, double** _x, double* _y) = 0;
};

class LinearLoss : public Loss {
    public:
        double getLoss(double* w, double* _x, double _y);
        double* getGradient(double* w, double** _x, double* _y);
};


double LinearLoss::getLoss(double* w, double* _x, double _y) {
	return 0.1;
}

double* LinearLoss::getGradient(double* w, double** _x, double* _y) {
    double* t = (double*) malloc(100);
    return t;
}

class LBFGS {
	private:
		double** x;
		double* y;
		double stop = 0.0001;
		int stopRound;
		float stopGrad;
		Loss& loss;
		double* weight;
		int featureSize;
		double* lastGrad;
		double* thisGrad;
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
		LBFGS(Loss& _loss, double** _x, double* _y, int _featureSize): loss(_loss), x(_x), y(_y), featureSize(_featureSize) {
            m = 10;
			weight = (double*) malloc(sizeof(double) * featureSize);
			memset(weight, 0, sizeof(double) * featureSize);
			alpha = (double*) malloc(sizeof(double) * m);
			rho = (double*) malloc(sizeof(double) * m);
			s = new LoopArray(m);
			t = new LoopArray(m);
			stepSize = 0.1;
			stopGrad = 0.001;
		};
		void learn();
};

void LBFGS::learn() {
	int realRound = 0;
	lastGrad = loss.getGradient(weight, x, y);
	do {
		runARound();
		++realRound;
	} while (realRound < stopRound && cblas_dnrm2(featureSize, thisGrad, 1) < stopGrad);
	if (realRound == stopRound) {
		cout << "Reach max round." << endl;
	} else {
		printf("Reach the gap: %lf", cblas_dnrm2(featureSize, thisGrad, 1));
	}
}

void LBFGS::runARound() {
    double* direction = getDirection(lastGrad);
	//  thisWeight = weight - direction * stepSize;
	cblas_daxpy(featureSize, 0 - stepSize, direction, 1, weight, 1);
	// s = thisW - lastW = -direction * step
	cblas_dscal(featureSize, 0 - stepSize, direction, 1);
	double* grad = loss.getGradient(weight, x, y);
	cblas_daxpy(featureSize, -1, lastGrad, 1, grad, 1);
	cblas_dswap(featureSize, lastGrad, 1, grad, 1);
	updateST(direction, lastGrad);
	H = cblas_ddot(featureSize, lastGrad, 1, direction, 1) / cblas_ddot(featureSize, lastGrad, 1, lastGrad, 1);
}

double* LBFGS::getDirection(double* qq) {
	// two loop
	double* q = (double*) malloc(sizeof(double) * featureSize);
	memcpy(q, qq, sizeof(double) * featureSize);
	int k = min(m, s->size());
	for (int i = k-1; i >= 0; --i) {
		rho[i] = 1.0 / cblas_ddot(featureSize, (*s)[i], 1, (*t)[i], 1);
		alpha[i] = cblas_ddot(featureSize, (*s)[i], 1, q, 1) * rho[i];
		// q = q - alpha[i] * t[i];
		cblas_daxpy(featureSize, alpha[i], (*t)[i], 1, q, 1);
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

void LBFGS::updateST(double* _s, double* _t) {
	s->appendAndRemoveFirstIfFull(_s);
	t->appendAndRemoveFirstIfFull(_t);
}


void generateData(double** x, double* y, double* weight, int featureSize, int sampleSize) {
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

int main() {
    double** x;
    double* y;
    double* weight;
    int featureSize = 10;
    int sampleSize = 100;
    generateData(x, y, weight, featureSize, sampleSize);
    LinearLoss ll;
	LBFGS lbfgs(ll, x, y, featureSize);
	lbfgs.learn();
	return 0;
}
