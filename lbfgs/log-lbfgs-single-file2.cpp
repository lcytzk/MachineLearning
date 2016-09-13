#include <stdlib.h>
#include <memory.h>
#include <vector>
#include <iostream>
#include <time.h>
#include <fstream>
#include <string>
#include <unordered_map>
#include <functional>

#include "lbfgs_util.h"
#include "ThreadPool.h"
#include "ls.h"

using namespace std;

int LEARN_START = 0;
double STEP_SIZE = 1.3;
int SAMPLE_SIZE = 0;    
int ROUND = 1;
int START_TIME, END_TIME;

int W_SIZE;
int INDEX_BIT = 18;
int INDEX_SIZE;
int* WEIGHT_INDEX;
double* WEIGHT;
double* CONDITION;
int THREAD_MAX = 10;
int BATCH_SIZE = 5;
ThreadPool pool(THREAD_MAX);

void outputModel(double* w, int size) {
    cout << "model" << endl;
    for(int i = 0; i < size; ++i) {
        printf("%f\t", w[i]);
    }
    printf("\n");
}

void saveModel(double* weight) {
    ofstream fo;
    fo.open("my.model");
    fo << INDEX_BIT << endl;
    for(int i = 0; i < INDEX_SIZE; ++i) {
        if(WEIGHT_INDEX[i] != -1)
            fo << i  << ':' << weight[WEIGHT_INDEX[i]] << endl;
    }
    fo.close();
}

int getOnlyIndex(string s) {
    int hash = 17;
    const char* c = s.c_str();
    while(*c) { hash = hash * 31 + *c; ++c; }
    return hash & (INDEX_SIZE - 1);
}

class Example {
    public:
        double prediction;
        double label;
        int featureSize;
        int* features;
        Example(vector<int>& _features): prediction(0) {
            features = (int*) malloc(_features.size() * sizeof(int));
            int index = 0;
            featureSize = _features.size();
            for(int f : _features) {
                features[index] = f;
                ++index;
            }
        }
        ~Example() { free(features); }
};

class LBFGS {
	private:
		double stopGrad;
		Loss& loss;
        vector<Example*>& examples;
		int m;
        double wolfe1Bound =  0.5;
		double* alpha;
		double* rho;
		double H = 1.0;
		LoopArray* s;
		LoopArray* t;
        double lambda2;
        double lossBound;
		double stepSize;
		double* weight;
        int weightSize;
        double* lastGrad;
        double* grad;
        double* direction = NULL;
        double lossSum;
        double preLossSum;

        void stepForward();
        void stepBackward();
        bool evalWolfe();
        void updateGradientWithLambda2(double* grad);
		void cal_and_save_ST();
		void updateST(double* _s, double* _t);
		double* getDirection(double* q);
        void predict();
        double* getGradient();
	public:
		LBFGS(Loss& _loss, vector<Example*>& _examples, double* _weight, int wSize, double _lambda2, double _lossBound): loss(_loss),  examples(_examples), weight(_weight), weightSize(wSize), lambda2(_lambda2), lossBound(_lossBound) {
            m = 15;
			alpha = (double*) malloc(sizeof(double) * m);
			rho = (double*) malloc(sizeof(double) * m);
			s = new LoopArray(m);
			t = new LoopArray(m);
			stepSize = 0.01;
			stopGrad = 0;
		};
		bool learn();
        void init();
};

bool LBFGS::evalWolfe() {
    //wolfe1 = (loss_sum - previous_loss_sum) / (step_size * g0_d);
    return lossSum - preLossSum  <= lambda2 * stepSize * dot(lastGrad, direction, W_SIZE);
}

void updateCondition(int* fs, double p, int size) {
    for(int i = 0; i < size; ++i) {
        CONDITION[fs[i]] += p*(1-p) / SAMPLE_SIZE;
    }
}

void LBFGS::predict() {
    preLossSum = lossSum;
    lossSum = 0;
    //double reg = 0.5 * lambda2 * dot(weight, weight, W_SIZE);
    vector<future<double>> results;
    int gap = SAMPLE_SIZE / BATCH_SIZE + 1;
    for(int i = 0 ; i < BATCH_SIZE; ++i) {
        int start = gap * i;
        int end = gap * (i + 1);
        if(start >= SAMPLE_SIZE) break;
        end = end > SAMPLE_SIZE ? SAMPLE_SIZE : end;
        results.emplace_back(
            pool.enqueue( [this, start, end] {
                double sum = 0;
                for(int index = start; index < end; ++index) {
                Example* example = this->examples[index];
                example->prediction = this->loss.getVal(weight, example->features, example->featureSize);
                //updateCondition(example->features, example->prediction, example->featureSize);
                sum += loss.getLoss(example->prediction, example->label);
                }
                return sum;
            })
        );
    }
    for(auto && result : results) {
        lossSum += result.get();
    }
}

void LBFGS::updateGradientWithLambda2(double* grad) {
    for(int i = 0; i < W_SIZE; ++i) {
        grad[i] += lambda2 * weight[i];
    }
}

double* LBFGS::getGradient() {
    grad = (double*) calloc(W_SIZE, sizeof(double));
    vector<future<double*>> results;
    int gap = SAMPLE_SIZE / BATCH_SIZE + 1;
    for(int i = 0 ; i < BATCH_SIZE; ++i) {
        int start = gap * i;
        int end = gap * (i + 1);
        if(start >= SAMPLE_SIZE) break;
        end = end > SAMPLE_SIZE ? SAMPLE_SIZE : end;
        results.emplace_back(
            pool.enqueue( [this, start, end] {
                double* thisgrad = (double*) calloc(W_SIZE, sizeof(double));
                for(int index = start; index < end; ++index) {
                Example* example = this->examples[index];
                this->loss.updateGradient(example->prediction, example->features, example->label, thisgrad, example->featureSize, SAMPLE_SIZE);
                //updateCondition(example->features, example->prediction, example->featureSize);
                }
                return thisgrad;
            })
        );
    }
    for(auto && result : results) {
        double* g = result.get();
        for(int i = 0; i < W_SIZE; ++i) {
            grad[i] += g[i];
        }
        free(g);
    }
    //updateGradientWithLambda2(grad);
    return grad;
}

bool LBFGS::learn() {
    stepForward();
    predict();
    END_TIME = clock(); 
    printf("#%d#\tlossSum: %f\t preLossSum: %f\tstepSize:%f\tused time: %f", ROUND, lossSum/SAMPLE_SIZE, preLossSum/SAMPLE_SIZE, stepSize, (END_TIME - START_TIME)/(double)CLOCKS_PER_SEC);
    START_TIME = END_TIME;
    if(isnan(lossSum)) {
        printf("Loss sum nan so stop.\n");
        return false;
    }
    if(!evalWolfe()) {
        stepBackward();
        printf("\t step back\n");
        stepSize /= 2;
        return true;
    }
    printf("\n");
//    if(wolfe1 == 0 || isnan(wolfe1)) {
//        cout << "wolfe1 is nan  " << wolfe1 << endl;
//        return false;
//    }
//    cout << "wolfe1    " << wolfe1 << endl;
    if(lossBound == 0) {
        if(isnan(lossSum) || lossSum/preLossSum > 0.999) {
            printf("Decrease in loss in 0.01%% so stop.\n");
            return false;
        }
    } else {
        if(isnan(lossSum) || lossSum/SAMPLE_SIZE < lossBound) {
            printf("Reach the bound %f so stop.\n", lossBound);
            return false;
        }
    }
    getGradient();
//    float grad_norm = norm(grad);
//    cout << "norm   " << grad_norm << endl;
//  if(grad_norm < stopGrad && isnan(grad_norm)) {
    //if(lossSum == 0 || grad_norm == 0 || isnan(grad_norm)) {
    //if(lossSum == 0 || isnan(grad_norm)) {
    //    cout << "Reach the gap  " << grad_norm << endl;
    //    return false;
    //}
    cal_and_save_ST();
    getDirection(lastGrad);
    stepSize = STEP_SIZE;
    return true;
}

void LBFGS::init() {
    cout << "lbfgs init" << endl;
    predict();
    lastGrad = getGradient();
    getDirection(lastGrad);
}

void LBFGS::stepForward() {
    for(int i = 0; i < W_SIZE; ++i) {
        weight[i] -= direction[i] * stepSize;
    }
}

void LBFGS::stepBackward() {
    for(int i = 0; i < W_SIZE; ++i) {
        weight[i] += direction[i] * stepSize;
    }
    lossSum = preLossSum;
}

void LBFGS::cal_and_save_ST() {
	//  weight = weight - direction * stepSize;
	// s = thisW - lastW = -direction * step
    scal(direction, 0 - stepSize, W_SIZE);
    //direction->scal(stepSize);
	double* ss = direction;
	// grad = grad - lastGrad, lastGrad = lastGrad + grad
    //grad->pax(-1, *lastGrad);
    ypax(grad, -1, lastGrad, W_SIZE);
    //lastGrad->pax(1, *grad);
    ypax(lastGrad, 1, grad, W_SIZE);
	// grad = grad - lastGrad grad will be y
	double* y = grad;
	updateST(ss, y);
    H = dot(ss, y, W_SIZE) / dot(y, y, W_SIZE);
}

double* LBFGS::getDirection(double* qq) {
	// two loop
	double* q = (double*) malloc(W_SIZE * sizeof(double));
    memcpy(q, qq, W_SIZE * sizeof(double));
	int k = min(m, s->size());
    if(k > 0) rho[k-1] = 1.0 / dot((*s)[k-1], (*t)[k-1], W_SIZE);
	for (int i = k-1; i >= 0; --i) {
        alpha[i] = dot(q, (*s)[i], W_SIZE) * rho[i];
        ypax(q, 0 - alpha[i], (*t)[i], W_SIZE);
	}
    scal(q, H, W_SIZE);
    //scalWithCondition(q, H, W_SIZE);
	for (int i = 0; i < k; ++i) {
        double beta = rho[i] * dot(q, (*t)[i], W_SIZE);
        ypax(q, alpha[i] - beta, (*s)[i], W_SIZE);
	}
    // shift rho
    if(k == m) {
        for(int i = 0; i < k-1; ++i) {
            rho[i] = rho[i+1];
        }
    }
    //if(direction != NULL) free(direction);
    direction = q;
	return q;
}

void LBFGS::updateST(double* _s, double* _t) {
	s->appendAndRemoveFirstIfFull(_s);
	t->appendAndRemoveFirstIfFull(_t);
}

void splitString(string s, vector<string>& output, const char delimiter) {
    size_t start;
    size_t index = 0;

    while(index != string::npos) {
        index = s.find_first_not_of(delimiter, index);
        if(index == string::npos) break;
        start = index;
        index = s.find_first_of(delimiter, index);
        if(index == string::npos) {
            output.push_back(s.substr(start, s.size() - start));
            break;
        }
        output.push_back(s.substr(start, index - start));
    }
}

void splitStringAndHash(string s, const char delimiter, vector<int>& x, double& y) {
    const char* c = s.c_str();
    uint64_t hash = 17;
    uint64_t hash2 = 17;
    bool first = true;
    bool group = false;
    
    y = (atoi(s.substr(0, 2).c_str()) == 1) ? 1 : 0;

    while(*c) {
        while(*c && *c == delimiter) { ++c; }
        if(!*c) break;
        if(*c == '|') {
            group = true;
            hash = 17;
        }
        hash2 = hash;
        while(*c && *c != delimiter) { 
            hash2 = hash2 * 31 + *c;
            ++c;
        }
        if(group) { 
            hash = hash2; 
            group = false; 
        } else {
            if(first) {
                first = false;
                continue;
            }
            //WEIGHT_INDEX[hash2 & (INDEX_SIZE - 1)] = 1;
            x.push_back(hash2 & (INDEX_SIZE - 1));
        }
    }
} 

bool getNextXY(vector<int>& x, double& y, istream& fo) {
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
    x.clear();
    splitStringAndHash(str, ' ', x, y);
    return true;
}

bool getOnlyNextXY(vector<int>& x, double& y, istream& fo, vector<string>& v) {
    string str;
    if(!getline(fo, str)) {
        cout << "reach the file end.1" << endl;
        return false;
    }
    if(str.length() < 2) {
        cout << "reach the file end.2" << endl;
        return false;
    }
    v.clear();
    splitString(str, v, ' ');
    x.clear();
    y = (atoi(v[0].c_str()) == 1) ? 1.0 : 0;
    string prefix;
    for(int i = 1; i < v.size(); ++i) {
        if(v[i].c_str()[0] != '|') {
            x.push_back(getOnlyIndex(prefix + v[i]));
        } else {
            prefix = v[i];
        }
    }
    return true;
}

void outputAcu(double* weight, Loss& ll, char* testfile) {
    cout << "begin acu" << endl;
    vector<int> xx;
    double yy;
    ifstream fo;
    ofstream fw;
    fo.open(testfile);
    fw.open("my_res");
    double allLoss = 0;
    int count = 0;
    int sampleSize = 0;
    vector<string> v;
    int count1 = 0;
    int count1_shot = 0;
    while (getOnlyNextXY(xx, yy, fo, v)) {
        double loss = ll.getLoss(weight, xx, yy);
        double prediction1 = ll.getVal(weight, xx);
        double prediction = prediction1 > 0.5 ? 1 : 0;
        allLoss += loss;
        if (prediction == yy) {
            ++count;
        }
        if(yy == 1) {
            ++count1;
            if(prediction == yy) {
                ++count1_shot;
            }
        }
        fw << prediction1 << "," << (yy == 1 ? 1 : -1) << endl;
     //   if (sampleSize % 1000 == 0) cout << sampleSize << endl;
        ++sampleSize;
    }
    fo.close();
    fw.close();
    cout << "avg loss: " << allLoss / sampleSize << endl;
    cout << "count percent: " << ((double) count) / sampleSize << endl;
    cout << "count: " << count << endl;
    cout << "count1: " << count1 << endl;
    cout << "count1_shot: " << count1_shot << endl;
    cout << "sampleSize: " << sampleSize << endl;
}

void freeExamples(vector<Example*>* examples) {
    delete examples;
}

void loadExamples(vector<Example*>& examples, istream& f) {
    vector<future<vector<Example*>*>> results;
    string** strs = new string*[1001];
    string* str;
    int count = 0;
    while(!f.eof()) {
        str = new string();
        getline(f, *str);
        if(str->size() < 3) break;
        strs[count] = str;
        count = (count + 1) % 1000;
        if(!count) {
            results.emplace_back(
                pool.enqueue( [strs] {
                    vector<Example*>* exs = new vector<Example*>();
                    vector<int> x;
                    double y;
                    for(int i = 0; i < 1000; ++i) {
                        x.clear();
                        splitStringAndHash(*strs[i], ' ', x, y);
                        delete strs[i];
                        Example* example = new Example(x);
                        example->label = y;
                        exs->push_back(example);
                    }
                    delete strs;
                    return exs;
                })
            );
            strs = new string*[1001];
        }
    }
    if(count) {
        results.emplace_back(
            pool.enqueue( [strs, count] {
                vector<Example*>* exs = new vector<Example*>();
                vector<int> x;
                double y;
                for(int i = 0; i < count; ++i) {
                    x.clear();
                    splitStringAndHash(*strs[i], ' ', x, y);
                    delete strs[i];
                    Example* example = new Example(x);
                    example->label = y;
                    exs->push_back(example);
                }
                delete strs;
                return exs;
            })
        );
    }
    for(auto && job : results) {
        vector<Example*>* res = job.get();
        examples.insert(examples.begin(), res->begin(), res->end());
        freeExamples(res);
    }
}

void loadExamples(vector<Example*>& examples, vector<string>& files) {
    ifstream fo;
    for(auto file : files) {
        fo.open(file);
        loadExamples(examples, fo);
        fo.close();
    }
}

void loadExamples(vector<Example*>& examples, string& file) {
    ifstream fo;
    fo.open(file);
    loadExamples(examples, fo);
    fo.close();
}

void initgap(vector<Example*>& examples) {
    cout << "initgap begin" << endl;
    int count = 0;
    for(int i = 0; i < INDEX_SIZE; ++i) {
        if(WEIGHT_INDEX[i] == 1) {
            WEIGHT_INDEX[i] = count++;
        } else {
            WEIGHT_INDEX[i] = -1;
        }
    }
    WEIGHT = (double*) calloc(count, sizeof(double));
    CONDITION = (double*) calloc(count, sizeof(double));
    W_SIZE = count;
    cout << "Initgap finish. After hash feature size is: " << count << endl;
    for(Example* example : examples) {
        for(int i = 0; i < example->featureSize; ++i) {
            example->features[i] = WEIGHT_INDEX[example->features[i]];
        }
    }
}

void do_main(vector<Example*>& examples, double lambda2, double lossBound) {
    int start, end;
    SAMPLE_SIZE = examples.size();
    LogLoss ll;
	LBFGS lbfgs(ll, examples, WEIGHT, W_SIZE, lambda2, lossBound);
    cout << "begin learn" << endl;
    lbfgs.init();
    start = end;
    bool flag = false;
    int LEADN_START = clock();
    for(int i = 0; i < 1000; ++i) {
        START_TIME = clock();
    	if(!lbfgs.learn()) {
            break;
        }
        ++ROUND;
    }
    printf("Learned finished, used %f.\n", (clock() - LEARN_START) / (double) CLOCKS_PER_SEC);
    saveModel(WEIGHT);
}

void lbfgs_main(double lambda2, double lossBound) {
    cout << "load examples." << endl;
    int start = clock();
    vector<Example*> examples;
    loadExamples(examples, cin);
    initgap(examples);
    cout << "load examples cost " << (clock() - start)/(double)CLOCKS_PER_SEC << endl;
    printf("example size is : %ld\n", examples.size());
    do_main(examples, lambda2, lossBound);
}


void lbfgs_main(vector<string>& files, double lambda2, double lossBound) {
    cout << "lbfgs_main" << endl;
    int passes = 1;
    vector<Example*> examples;
    LogLoss ll;
	LBFGS lbfgs(ll, examples, WEIGHT, W_SIZE, lambda2, lossBound);
    bool needInit = true;
    while(passes--) {
        for(string& file : files) {
            loadExamples(examples, file);
            SAMPLE_SIZE = examples.size();
            if(needInit) {
                needInit = false;
                lbfgs.init();
            }
            for(int i = 0; i < 3; ++i) {
                if(!lbfgs.learn()) return;
            }
            for(int i = 0; i < examples.size(); ++i) {
                delete examples[i];
            }
            examples.clear();
        }
        ROUND++;
    }
}

double* loadModel() {
    cout << "begin load model" << endl;
    ifstream fr;
    fr.open("my.model");
    string str;
    getline(fr, str);
    INDEX_BIT = atoi(str.c_str());
    INDEX_SIZE = 1 << INDEX_BIT;
    WEIGHT = (double*) calloc(INDEX_SIZE, sizeof(double));
    int i = 0;
    vector<string> v;
    while(!fr.eof()) {
        getline(fr, str);
        v.clear();
        splitString(str, v, ':');
        if(v.size() < 2) break;
        int index = atoi(v[0].c_str());
        WEIGHT[index] = atof(v[1].c_str());
    }
    return WEIGHT;
}

int main(int argc, char* argv[]) {
    int start_time = clock();
    string mode(argv[1]);

    char* filename;
    double lambda2;
    double lossBound;
    vector<string> files;

    if(mode == "load") {
        LogLoss ll;
        double* weight = loadModel();
        outputAcu(weight, ll, argv[2]);
    } else { 
        if(mode == "file") {
            filename = argv[2];
            files.push_back(string(filename));
            lambda2 = atof(argv[3]);
            lossBound = atof(argv[4]);
            INDEX_BIT = atoi(argv[5]);
        } else if(mode == "cat") {
            lambda2 = atof(argv[2]);
            lossBound = atof(argv[3]);
            INDEX_BIT = atoi(argv[4]);
        } else if(mode == "dir") {
            char* dir = argv[2];
            lambda2 = atof(argv[3]);
            lossBound = atof(argv[4]);
            INDEX_BIT = atoi(argv[5]);
            list_directory(dir, files);
        }
        for(auto file : files) cout << file << endl;
        INDEX_SIZE = 1 << INDEX_BIT;
        WEIGHT_INDEX = (int*) calloc(INDEX_SIZE, sizeof(int));
        WEIGHT = (double*) calloc(INDEX_SIZE, sizeof(double));
        W_SIZE = INDEX_SIZE;
        if(mode == "cat") {
            lbfgs_main(lambda2, lossBound);
        } else {
            lbfgs_main(files, lambda2, lossBound);
        }
        cout << "finish program." << endl;
        cout << "Used time: " << (clock() - start_time)/double(CLOCKS_PER_SEC) << endl; 
    }
	return 0;
}
