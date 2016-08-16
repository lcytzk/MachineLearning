#include <malloc.h>
#include <memory.h>

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
		// return gradient.
		void runARound();
		void updateST();
		double* getDirection();
		Loss<DataType>& loss;
		double* weight;
		int featureSize;
		double* lastGrad;
		double stepSize;
		double** s;
		double** t;
	public:
		LBFGS(Loss<DataType>& _loss, DataType* _x, DataType* _y, int _featureSize): loss(_loss), x(_x), y(_y), featureSize(_featureSize) {};
		void learn();
};

template<typename DataType>
void LBFGS<DataType>::learn() {
    weight = (double*) malloc(sizeof(double) * featureSize);
    memset(weight, 0, sizeof(double) * featureSize);
    stepSize = 0.1;
}

template<typename DataType>
void LBFGS<DataType>::runARound() {
    double* direction = getDirection(lastGrad);
    double* thisWeight = weight - direction * stepSize;

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
