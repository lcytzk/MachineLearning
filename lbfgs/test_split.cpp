#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <functional>

using namespace std;

hash<string> str_hash;

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

void splitStringAndHash(string s, vector<string>& output, const char delimiter, vector<int>& x, double& y) {
    const char* c = s.c_str();

    uint64_t hash = 17;
    uint64_t hash2 = 17;

    bool first = true;
    bool group = false;

    y = atof(s.substr(0, 2).c_str());

    while(*c) {
        while(*c && *c == delimiter) { ++c; }
        if(!*c) break;
        if(*c == '|') {
            group = true;
            hash = 17;
        }
        hash2 = hash;
        while(*c && *c != delimiter) { 
            hash2 = hash2 * 31 + (unsigned int) *c;
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

int getIndex(string s) {
    int hash = 17;
    const char* c = s.c_str();
    while(*c) { hash = hash * 31 + (unsigned int) *c; ++c; }
    return hash;
}

int getIndex2(string s) {
    return str_hash(s);
}

bool getNextXY(vector<int>& x, double& y, ifstream& fo, vector<string>& v) {
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
    splitString(str, v, ' ');
    x.clear();
    y = (atoi(v[0].c_str()) == 1) ? 1.0 : 0;
    string prefix;
    for(int i = 1; i < v.size(); ++i) {
        if(v[i].c_str()[0] != '|') {
            x.push_back(getIndex(prefix + v[i]));
        } else {
            prefix = v[i];
        }
    }
    return true;
}

bool getNextXY2(vector<int>& x, double& y, ifstream& fo, vector<string>& v) {
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
    x.clear();
    splitStringAndHash(str, v, ' ', x, y);
    return true;
}

void outputVec(vector<int>& x, ofstream& f) {
    for(auto i : x) f << i << endl;
}

void loadExamples(const char* filename) {
    ifstream fo;
    fo.open(filename);
    vector<int> xx;
    double yy;
    vector<string> v;
    ofstream savef;
    savef.open("1_split_file");
    while(getNextXY(xx, yy, fo, v)) {
        outputVec(xx, savef);
    }
    fo.close();
    savef.close();
}

void loadExamples2(const char* filename) {
    ifstream fo;
    fo.open(filename);
    vector<int> xx;
    double yy;
    vector<string> v;
    ofstream savef;
    savef.open("2_split_file");
    while(getNextXY2(xx, yy, fo, v)) {
        //cout << yy << endl;
        outputVec(xx, savef);
    }
    fo.close();
    savef.close();
}

string filename = "test2.txt";

void test() {
    loadExamples(filename.c_str());
}

void test2() {
    loadExamples2(filename.c_str());
}

int main() {
    int start = clock();
    test();
    int end = clock();
    printf("Used: %f\n", (end - start) / (double) CLOCKS_PER_SEC);
    start = end;
    test2();
    end = clock();
    printf("Used: %f\n", (end - start) / (double) CLOCKS_PER_SEC);
    return 0;
}
