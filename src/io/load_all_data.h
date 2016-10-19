#pragma once
#include "io.h"
#include <vector>

namespace DML{
class LOAD_ALL_DATA : public IO{
    public:
        LOAD_ALL_DATA(const char *file_path, int rank, int nproc) : IO(file_path), rank(rank), nproc(nproc){
        }
        ~LOAD_ALL_DATA(){}

        void load(){
            fea_matrix.clear();
            while(!fin_.eof()){
                std::getline(fin_, line);
                sample.clear();
                const char *pline = line.c_str();
                if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
                    pline += nchar;
                    label.push_back(y);
                    while(sscanf(pline, "%d:%ld:%f%n", &fgid, &fid, &val, &nchar) >= 3){
                        pline += nchar;
                        keyval.fgid = fgid;
                        keyval.fid = fid;
                        keyval.val = val;
                        sample.push_back(keyval);
                    }
                }
                fea_matrix.push_back(sample);
            }    
        }

    public:
        key_val keyval;
        std::vector<key_val> sample;
        std::vector<std::vector<key_val>> fea_matrix;
        std::vector<int> label;
        int rank;
        int nproc;
};
}
