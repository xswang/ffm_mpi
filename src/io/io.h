#pragma once
#include <fstream>
#include <iostream>

namespace DML{

struct kv{
    int fgid;
    long int fid;
    float val;
};

class IO{
    public:
        IO(const char *file_path) : file_path(file_path){
            Init();
        };
        ~IO(){};

        void Init(){
            fin_.open(file_path, std::ios::in);
            if(!fin_.is_open()){
                std::cout<<"open file"<<file_path<<" error!"<<std::endl;
                exit(1);
            }else{
                std::cout<<"open file"<<file_path<<" sucess!"<<std::endl;
            }
        }

        virtual void load() = 0;

    public:
        const char *file_path;
        std::ifstream fin_;
        typedef kv key_val;
        std::string line;
        int fgid;
        int fid;
        float val;
        int nchar;
        int y;
};

}
