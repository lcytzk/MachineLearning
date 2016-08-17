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
	//memcpy(t, _y, sizeof(double) * featureSize);
	for (int i = 0; i < featureSize; ++i) {
		double tmp = 0;
		for (int j = 0; j < sampleSize; ++j) {
			tmp += getLoss(w, _x[j], _y[j], featureSize) * _x[j][i];
		}
		t[i] = tmp / sampleSize;
	}
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
		int featureSize;
		int sampleSize;
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
		LBFGS(Loss& _loss, double** _x, double* _y, int _featureSize, int _sampleSize): loss(_loss), x(_x), y(_y), featureSize(_featureSize), sampleSize(_sampleSize) {
            m = 10;
			weight = (double*) malloc(sizeof(double) * featureSize);
			memset(weight, 0, sizeof(double) * featureSize);
			alpha = (double*) malloc(sizeof(double) * m);
			rho = (double*) malloc(sizeof(double) * m);
			thisGrad = (double*) malloc(sizeof(double) * featureSize);
	        lastGrad = loss.getGradient(weight, x, y, featureSize, sampleSize);
			//lastGrad = (double*) malloc(sizeof(double) * featureSize);
			s = new LoopArray(m);
			t = new LoopArray(m);
			stepSize = 0.1;
			stopGrad = 0.001;
            stopRound = sampleSize;
		};
		void learn();
		double* weight;
};

void LBFGS::learn() {
	int realRound = 0;
	do {
		runARound();
		++realRound;
        printf("Round: %d, gradNorm: %f\n", realRound, cblas_dnrm2(featureSize, thisGrad, 1));
	} while (realRound < stopRound && cblas_dnrm2(featureSize, thisGrad, 1) > stopGrad);
	if (realRound == stopRound) {
		cout << "Reach max round." << endl;
	} else {
		printf("Reach the gap: %f\n", cblas_dnrm2(featureSize, thisGrad, 1));
		printf("Run %d rounds.\n", realRound);
	}
}

void LBFGS::runARound() {
    double* direction = getDirection(lastGrad);
	//  thisWeight = weight - direction * stepSize;
	cblas_daxpy(featureSize, 0 - stepSize, direction, 1, weight, 1); // weight will be updated here.
	// s = thisW - lastW = -direction * step
	cblas_dscal(featureSize, 0 - stepSize, direction, 1); // direction will be s.
	double* s = direction;
	double* grad = loss.getGradient(weight, x, y, featureSize, sampleSize);
	// lastGrad = thisGrad, thisGrad = grad
	memcpy(lastGrad, thisGrad, sizeof(double) * featureSize);
	memcpy(thisGrad, grad, sizeof(double) * featureSize);
	// grad = grad - lastGrad grad will be y
	cblas_daxpy(featureSize, -1, thisGrad, 1, grad, 1);
	double* y = grad;
	updateST(s, y);
	H = cblas_ddot(featureSize, y, 1, s, 1) / cblas_ddot(featureSize, y, 1, y, 1);
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

int main() {
    double** x;
    double* y;
    double* weight;
    int featureSize = 10;
    int sampleSize = 100;
    generateData(x, y, weight, featureSize, sampleSize);
    cout << "init data done." << endl;
    LinearLoss ll;
	LBFGS lbfgs(ll, x, y, featureSize, sampleSize);
    cout << "begin learn" << endl;
	lbfgs.learn();
    cout << "true model is: " << endl;
    outputModel(weight, featureSize);
    cout << "my model is: " << endl;
    outputModel(lbfgs.weight, featureSize);
    double allLoss = 0;
    for(int i = 0; i < sampleSize; ++i) {
        allLoss = ll.getLoss(lbfgs.weight, x[i], y[i], featureSize);
    }
    cout << "avg loss: " << allLoss / sampleSize << endl;
	return 0;
}
