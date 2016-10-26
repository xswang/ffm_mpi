#include "load_all_data.h"

namespace dml{
void LoadAllData::load(){
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
}//end load

}
