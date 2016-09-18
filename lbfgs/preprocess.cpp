#include <iostream>
#include <istream>
#include <fstream>
#include <vector>

using namespace std;

class Example {
    public:
        double prediction;
        short label;
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

void splitStringAndHash(string s, const char delimiter, vector<int>& x, short& y) {
    const char* c = s.c_str();
    int hash = 17;
    int hash2 = 17;
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
            x.push_back(hash2);
        }
    }
} 

bool getNextXY(vector<int>& x, short& y, istream& fo, vector<string>& v) {
    if(fo.eof()) {
        cout << "reach the file end.1" << endl;
        return false;
    }
    string str;
    getline(fo, str);
    if(str.length() < 2) {
        return false;
    }
    x.clear();
    splitStringAndHash(str, ' ', x, y);
    return true;
}

void loadExamples(vector<Example*>& examples, istream& f) {
    vector<int> xx;
    short yy;
    vector<string> v;
    while(getNextXY(xx, yy, f, v)) {
        Example* example = new Example(xx);
        example->label = yy;
        examples.push_back(example);
    }
}

void serialization(vector<Example*>& examples) {
    FILE* fo = fopen("se.txt", "wb");
    for(Example* example : examples) {
        fwrite(&example->label, sizeof(short), 1, fo);
        fwrite(&example->featureSize, sizeof(int), 1, fo);
        fwrite(example->features, sizeof(int), example->featureSize, fo);
        //printf("%hd\t%d\n", example->label, example->featureSize);
    }
    fclose(fo);
}

void derialization() {
    short label = 2;
    int featureSize = -1;
    int* features = new int[200];
    FILE* fi = fopen("se.txt", "rb");
    while(fread(&label, sizeof(short), 1, fi)) {
        fread(&featureSize, sizeof(int), 1, fi);
        fread(features, sizeof(int), featureSize, fi);
        //printf("%hd\t%d\n", label, featureSize);
    }
    fclose(fi);
}

int derializationAndCmp(vector<Example*>& es1) {
    short label;
    //int featureSize;
    int* features = new int[200];
    FILE* fi = fopen("se.txt", "rb");
    int i = 0;
    while(fread(&label, sizeof(short), 1, fi)) {
        fread(&featureSize, sizeof(int), 1, fi);
        fread(features, sizeof(int), featureSize, fi);
        //printf("%hd\t%d\n", es1[i]->label, es1[i]->featureSize);
        //printf("%hd\t%d\n", label, featureSize);
        if(es1[i]->label != label || es1[i]->featureSize != featureSize) return -1;
        for(int j = 0; j < featureSize; ++j) {
            if(es1[i]->features[j] != features[j]) return -1;
        }
        ++i;
    }
    fclose(fi);
}

void readAndParse(vector<Example*>& examples) {
    ifstream fi;
    //fi.open("1.txt");
    fi.open("/mnt/storage01/yoyo_dq/vwq_20160917/000000_0");
    loadExamples(examples, fi);
    fi.close();
    //serialization(examples);
    //derialization();
}

int cmpExps(vector<Example*>& es1, vector<Example*>& es2) {
    if(es1.size() != es2.size()) return -1;
    int s = es1.size();
    for(int i = 0; i < s; ++i) {
        if(es1[i]->label != es2[i]->label || es1[i]->featureSize != es2[i]->featureSize) return -1;
        for(int j = 0; j < es1[i]->featureSize; ++j) {
            if(es1[i]->features[j] != es2[i]->features[j]) return -1;
        }
    }
    return 0;
}

void test() {
    vector<Example*> examples;
    int start, end;
    start = clock();
    readAndParse(examples);
    end = clock();
    printf("Read and parse use: %f\n", (end - start)/(double) CLOCKS_PER_SEC);
    start = clock();
    serialization(examples);
    end = clock();
    printf("seri use: %f\n", (end - start)/(double) CLOCKS_PER_SEC);
    start = clock();
    if(derializationAndCmp(examples) == -1) cout << "Fail" << endl;
    end = clock();
    printf("deri and cmp use: %f\n", (end - start)/(double) CLOCKS_PER_SEC);
    start = clock();
}

int main() {
    //readAndParse();
    test();
}

