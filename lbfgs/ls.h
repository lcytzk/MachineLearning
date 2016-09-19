#ifndef LS_H
#define LS_H

#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>

using namespace std;

void list_directory(const char *dirname, vector<string>& filenames) {
    DIR *dir;
    struct dirent *ent;
    string dname(dirname);
    if(dname[dname.size() - 1] != '/') dname = dname + '/';
                
    /* Open directory stream */
    dir = opendir (dirname);
    if (dir != NULL) {
        /* Print all files and directories within the directory */
        while ((ent = readdir (dir)) != NULL) {
            string s(ent->d_name);
            if(s.find(".") == string::npos) filenames.push_back(dname + s);
        }
        closedir (dir);
    } else {
        /* Could not open directory */
        printf ("Cannot open directory %s\n", dirname);
    }
}

#endif // LS_H
