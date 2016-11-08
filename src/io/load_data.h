#pragma once
#include "io.h"
#include <vector>

namespace dml{
class LoadData : public IO{
    public:
        LoadData(const char *file_path) : IO(file_path){
        }
        ~LoadData(){}

        void load_all_data();
        void load_batch_data(int num);

    public:
        key_val keyval;
        std::vector<key_val> sample;
        std::vector<std::vector<key_val>> fea_matrix;
        std::vector<int> label;
};
}
