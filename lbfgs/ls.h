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
            if(ent->d_name[0] != '.') filenames.push_back(dname + string(ent->d_name));
        }
        closedir (dir);
    } else {
        /* Could not open directory */
        printf ("Cannot open directory %s\n", dirname);
    }
}

#endif // LS_H
