#ifndef __LBFGS_UTIL__
#define __LBFGS_UTIL__

#include <math.h>
#include <vector>
#include "ThreadPool.h"

using namespace std;

double dot(double* x, double* y, int size) {
    double res = 0;
    for(int i = 0; i < size; ++i) {
        res += x[i] * y[i];
    }
    return res;
}

double dot(double* x, double* y, int size, ThreadPool &pool, int batchSize) {
    vector<future<double>> results;
    int gap = size / batchSize + 1;
    for(int i = 0; i < batchSize; ++i) {
        int start = gap * i;
        int end = gap * (i + 1);
        if(start >= size) break;
        end = end > size ? size : end;
        results.emplace_back( 
            pool.enqueue( [x, y, start, end] {
                double res = 0;
                for(int i = start; i < end; ++i) {
                    res += x[i] * y[i];
                }
                return res;
            })
        );
    }
    double ret = 0;
    for(auto && result : results) {
        ret += result.get();
    }
    return ret;
}

float norm(double* x, int size) {
    return sqrt(dot(x, x, size));
}

void scal(double* direction, double stepSize, int size) {
    for(int i = 0; i < size; ++i) {
        direction[i] *= stepSize;
    }
}

void scal(double* direction, double stepSize, int size, ThreadPool &pool, int batchSize) {
    vector<future<void>> results;
    int gap = size / batchSize + 1;
    for(int i = 0; i < batchSize; ++i) {
        int start = gap * i;
        int end = gap * (i + 1);
        if(start >= size) break;
        end = end > size ? size : end;
        results.emplace_back( 
            pool.enqueue( [direction, stepSize, start, end] {
                for(int i = start; i < end; ++i) {
                    direction[i] *= stepSize;
                }
            })
        );
    }
    for(auto && result : results) {
        result.wait();
    }
}

void scalWithCondition(double* direction, double* condition, double stepSize, int size) {
    for(int i = 0; i < size; ++i) {
        direction[i] *= stepSize * condition[i];
    }
}

void ypax(double* y, double alpha, double* x, int size, ThreadPool &pool, int batchSize) {
    vector<future<void>> results;
    int gap = size / batchSize + 1;
    for(int i = 0 ; i < batchSize; ++i) {
        int start = gap * i;
        int end = gap * (i + 1);
        if(start >= size) break;
        end = end > size ? size : end;
        results.emplace_back( 
            pool.enqueue( [y, alpha, x, start, end] {
                for(int i = start; i < end; ++i) {
                    y[i] += alpha * x[i];
                }
            })
        );
    }
    for(auto && result : results) {
        result.wait();
    }
}

void ypax(double* y, double alpha, double* x, int size) {
    for(int i = 0; i < size; ++i) {
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

double dot(int* x, double* w, int size) {
    double res = 0;
    for(int i = 0 ; i < size; ++i) {
        res += w[x[i]];
    }
    return res;
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
        virtual double getLoss(double prediction, double _y) = 0;
		virtual double getLoss(double* w, vector<int>& _x, double _y) = 0;
        virtual double getVal(double* w, vector<int>& _x) = 0;
        virtual double getVal(double* w, int* _x, int size) = 0;
        virtual double getFirstDeri(double prediction, double _y) = 0;
        virtual void updateGradient(double prediction, int* _x, double _y, double* t, int size, int sampleSize) = 0;
};

class LogLoss2 : public Loss {
    public:
        double getLoss(double prediction, double _y);
        double getLoss(double* w, vector<int>& _x, double _y);
        double getVal(double* w, vector<int>& _x);
        double getVal(double* w, int* _x, int size);
        double getFirstDeri(double prediction, double _y);
        void updateGradient(double prediction, int* _x, double _y, double* t, int size, int sampleSize);
};

double LogLoss2::getVal(double* w, vector<int>& _x) {
    return dot(_x, w);
}

double LogLoss2::getVal(double* w, int* _x, int size) {
    return 0;
}

double LogLoss2::getLoss(double* w, vector<int>& _x, double _y) {
	return log(1 + exp((1 - 2 *_y) * dot(_x, w)));
}


double LogLoss2::getLoss(double prediction, double _y) {
    return log(1 + exp((1 - 2 *_y) * prediction));
}

double LogLoss2::getFirstDeri(double prediction, double _y) {
    return (1 - 2 *_y) / (1.0 + exp((2 *_y - 1) * prediction));
}

void LogLoss2::updateGradient(double prediction, int* _x, double _y, double* t, int size, int sampleSize) {
    for (int i = 0; i < size; ++i) {
        t[_x[i]] += getFirstDeri(prediction, _y) / sampleSize;
    }
}

class LogLoss : public Loss {
    public:
        double getLoss(double prediction, double _y);
        double getLoss(double* w, vector<int>& _x, double _y);
        double getVal(double* w, vector<int>& _x);
        double getVal(double* w, int* _x, int size);
        double getFirstDeri(double prediction, double _y);
        void updateGradient(double prediction, int* _x, double _y, double* t, int size, int sampleSize);
};

double LogLoss::getVal(double* w, vector<int>& _x) {
    return 1 / (1 + exp(-dot(_x, w)));
}

double LogLoss::getVal(double* w, int* _x, int size) {
    return 1 / (1 + exp(-dot(_x, w, size)));
}

double LogLoss::getLoss(double* w, vector<int>& _x, double _y) {
    return getLoss(getVal(w, _x), _y);
}

double LogLoss::getLoss(double prediction, double _y) {
    return 0 - (_y == 1 ? log(prediction) : log(1 - prediction));
}

double LogLoss::getFirstDeri(double prediction, double _y) {
    return (1 - 2 *_y) / (1.0 + exp((2 *_y - 1) * prediction));
}

void LogLoss::updateGradient(double prediction, int* _x, double _y, double* t, int size, int sampleSize) {
    for (int i = 0; i < size; ++i) {
        t[_x[i]] += (prediction - _y);// / sampleSize;
    }
}

#endif // __LBFGS_UTIL__
