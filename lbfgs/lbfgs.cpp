#include <malloc.h>
#include <memory.h>
#include <math.h>
#include <vector>
#include <iostream>

using namespace std;

template<typename T>
class LoopArray {
	private:
		int start, end;
		int realSize;
		int maxSize;
		T* array;
	public:
		LoopArray(int _size) : maxSize(_size) {
			realSize = 0;
			start = end = 0;
			array = (T*) malloc(sizeof(T) * _size);
		}
		T operator[](const int _index) const {
			return array[(_index + start) % size];
		}
		void appendAndRemoveFirstIfFull(T element) {
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

template<typename DataType>
class Loss {
	public:
		virtual double getLoss(double* w, DataType* _x, DataType* _y) = 0;
		virtual double* getGradient(double* w, DataType* _x, DataType* _y) = 0;
};

template<typename DataType>
class LinearLoss : public Loss<DataType> {
    public:
        double getLoss(double* w, DataType* _x, DataType* _y);
        double* getGradient(double* w, DataType* _x, DataType* _y);
};


template<typename DataType>
double LinearLoss<DataType>::getLoss(double* w, DataType* _x, DataType* _y) {
	return 0.1;
}

template<typename DataType>
double* LinearLoss<DataType>::getGradient(double* w, DataType* _x,DataType* _y) {
    double* t = (double*) malloc(100);
    return t;
}

template<typename DataType>
class LBFGS {
	private:
		double* x;
		double* y;
		double stop = 0.0001;
		int stopRound;
		double stopGrad;
		Loss<DataType>& loss;
		double* weight;
		int featureSize;
		double* lastGrad;
		double* thisGrad;
		double stepSize;
		int m;
		double* alpha;
		double* rho;
		double H = 1.0;
		LoopArray<double*>* s;
		LoopArray<double*>* t;

		// return gradient.
		void runARound();
		void updateST(double* _s, double* _t);
		double* getDirection(double* q);
	public:
		LBFGS(Loss<DataType>& _loss, DataType* _x, DataType* _y, int _featureSize): loss(_loss), x(_x), y(_y), featureSize(_featureSize), m(10) {
			weight = (double*) malloc(sizeof(double) * featureSize);
			memset(weight, 0, sizeof(double) * featureSize);
			alpha = (double*) malloc(sizeof(double) * m);
			rho = (double*) malloc(sizeof(double) * m);
			s = new LoopArray<double*>(10);
			t = new LoopArray<double*>(10);
			stepSize = 0.1;
		};
		void learn();
};

template<typename DataType>
void LBFGS<DataType>::learn() {
	int realRound = 0;
	lastGrad = loss.getGradient(weight, x, y);
	do {
		runARound();
		++realRound;
	} while (realRound < stopRound && abs(thisGrad) < stopGrad);
	if (realRound == stopRound) {
		cout << "Reach max round." << endl;
	} else {
		printf("Reach the gap: %lf", abs(thisGrad));
	}
}

template<typename DataType>
void LBFGS<DataType>::runARound() {
    double* direction = getDirection(lastGrad);
    double* thisWeight = weight - direction * stepSize;
}

template<typename DataType>
double* LBFGS<DataType>::getDirection(double* q) {
	// two loop
	int k = math.min(m, s.size());
	for (int i = k-1; i >= 0; --i) {
		rho[i] = 1.0 / s[i] * t[i];
		alpha[i] = s[i] * q * rho[i];
		q = q - alpha[i] * t[i];
	}
	double* r = H * q;
	for (int i = 0; i < k; ++i) {
		double beta = rho[i] * t[i] * r;
		r = r + (alpha[i] - beta) * s[i];
	}
	return r;
}

template<typename DataType>
void LBFGS<DataType>::updateST(double* _s, double* _t) {
	s.appendAndRemoveFirstIfFull(_s);
	t.appendAndRemoveFirstIfFull(_t);
}


void generateData(double* x, double* y) {

}

int main() {
    double* x;
    double* y;
    int featureSize;
    LinearLoss<double> ll;
	LBFGS<double> lbfgs(ll, x, y, featureSize);
	lbfgs.learn();
	return 0;
}
