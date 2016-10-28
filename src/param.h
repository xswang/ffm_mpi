#pragma once
#include <gflags/gflags.h>

DEFINE_int64(isbatch, 0, "");
DEFINE_int64(isonline, 0, "");
DEFINE_int64(epoch, 0, "epoch");
DEFINE_int64(batch_size, 0, "batchsize");
DEFINE_int64(fea_dim, 0, "");
DEFINE_int64(factor, 0, "");
DEFINE_int64(group, 0, "");
DEFINE_int64(isffm, 0, "");
DEFINE_int64(isfm, 0, "");
DEFINE_int64(islr, 0, "");
DEFINE_int64(issgd, 0, "");
DEFINE_int64(isftrl, 0, "");
DEFINE_int64(isowlqn, 0, "");
DEFINE_int64(issinglethread, 0, "");
DEFINE_int64(ismultithread, 0, "");

DEFINE_double(bias, 0.0, "bias");
DEFINE_double(alpha, 0.0, "alpha");
DEFINE_double(beta, 0.0, "");
DEFINE_double(lambda1, 0.0, "");
DEFINE_double(lambda2, 0.0, "");

DEFINE_string(train_data_path, "", "");
DEFINE_string(test_data_path, "", "");

namespace dml{
class Param{
    public:
        Param(int &argc, char *argv[]) : argc(argc), argv(argv){
            ::google::ParseCommandLineFlags(&argc, &argv, true);
            Init();
        }
        ~Param(){}

        void Init(){
            isbatch = FLAGS_isbatch;
            isonline = FLAGS_isonline;
            epoch = FLAGS_epoch;
            batch_size = FLAGS_batch_size;
            bias = FLAGS_fea_dim;
            alpha = FLAGS_alpha;
            beta = FLAGS_beta;
            lambda1 = FLAGS_lambda1;
            lambda2 = FLAGS_lambda2;
            fea_dim = FLAGS_fea_dim;
            factor = FLAGS_factor;
            group = FLAGS_group;
            isffm = FLAGS_isffm;
            isfm = FLAGS_isfm;
            islr = FLAGS_islr;
            issgd = FLAGS_issgd;
            isftrl = FLAGS_isftrl;
            isowlqn = FLAGS_isowlqn;
            issinglethread = FLAGS_issinglethread;
            ismultithread = FLAGS_ismultithread;
            train_data_path = FLAGS_train_data_path;
            test_data_path = FLAGS_test_data_path;
        }
    public:
        int argc;
        char **argv;
        int isbatch;
        int isonline;
        int epoch;
        int batch_size;
        double bias;
        double alpha;
        double beta;
        double lambda1;
        double lambda2;
        long int fea_dim;
        int factor;
        int group;
        int isffm;
        int isfm;
        int islr;
        int issgd;
        int isftrl;
        int isowlqn;
        int issinglethread;
        int ismultithread;
        std::string train_data_path;
        std::string test_data_path;
};
}
