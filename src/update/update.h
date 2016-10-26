#pragma once
#include "param.h"
namespace dml{
class Update(){
    public:
        Update(Param *param) : param(param){}
        ~Update(){}
    public:
        Param param;


};
}
