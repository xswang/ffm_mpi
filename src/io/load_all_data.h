#pragma once
#include "io.h"
#include <vector>

namespace dml{
class LoadAllData : public IO{
    public:
        LoadAllData(const char *file_path, int rank, int nproc) : IO(file_path), rank(rank), nproc(nproc){
        }
        ~LoadAllData(){}

        void load();

    public:
        key_val keyval;
        std::vector<key_val> sample;
        std::vector<std::vector<key_val>> fea_matrix;
        std::vector<int> label;
        int rank;
        int nproc;
};
}
