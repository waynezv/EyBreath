#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/algorithm/string.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

unsigned NUM_FRAMES        = 55;
unsigned FEAT_DIM          = 40;
unsigned NUM_FILTERS_CONV1 = 10;
unsigned FILTER_SIZE_CONV1 = 9;
unsigned ROWS1             = 40;
unsigned NUM_FILTERS_CONV2 = 5;
unsigned FILTER_SIZE_CONV2 = 5;
unsigned ROWS2             = 20;
unsigned EMBEDDING_SIZE    = 1024;
unsigned CONV_OUTPUT_DIM   = 400;
unsigned NUM_SPEAKERS      = 623;
unsigned K_1               = 36;

Dict speaker_dict;

class Instance {
    public:
        string        filename; // file storing instance
        int           speaker_id;
        int           num_frames;
        vector<float> feat_vec;
        bool          filled; // filled or not

        vector<float> read_vec();
};

vector<float> Instance::read_vec() {
    ifstream f(filename, ifstream::in);
    {
        string spkid;
        string numfr;
        getline(f, spkid);
        getline(f, numfr);
        speaker_id = stoi(spkid);
        num_frames = stoi(numfr);
        string line;
        vector<string> splitted;
        while (getline(f, line)) {
            boost::trim_if(line, boost::is_any_of("[ "));
            boost::trim_if(line, boost::is_any_of(" ]"));
            boost::split(splitted, line, boost::is_any_of("\t "), boost::token_compress_on);
            for(vector<string>::const_iterator it = splitted.begin();
                    it != splitted.end(); it++) {
                feat_vec.push_back(stof(*it));
            }
        }
        // for (auto i : feat_vec)
              // cout << i << ' ';
        // cout << feat_vec.size() << endl;
    }
    f.close();
    return feat_vec;
}

vector<Instance> read_instances(string path, string filelist) {
    vector<Instance> instances;
    ifstream f(filelist);
    {
        string line;
        while(getline(f, line)) {
            cerr << "Reading instance from " << path <<  line << "...\n";
            Instance instance;
            instance.filename = path + line;
            instance.read_vec();
            instance.filled = true;
            instances.push_back(instance);
        }
    }
    f.close();
    return instances;
}

struct NetBuilder {
    vector<Parameter> p_filts1;
    vector<Parameter> p_filts2;
    Parameter p_conv2embd;
    Parameter p_embd2spid;

    explicit NetBuilder(Model* model):
        p_conv2embd(model->add_parameters({EMBEDDING_SIZE, CONV_OUTPUT_DIM})),
        p_embd2spid(model->add_parameters({NUM_SPEAKERS, EMBEDDING_SIZE})) {
            for (int i=0; i<NUM_FILTERS_CONV1; ++i) {
                p_filts1.push_back(model->add_parameters({ROWS1, FILTER_SIZE_CONV1}));
            }
            for (int i=0; i<NUM_FILTERS_CONV2; ++i) {
                p_filts2.push_back(model->add_parameters({ROWS2, FILTER_SIZE_CONV2}));
            }
        }

    Expression get_loss(ComputationGraph& cg, Instance instance) {
        Expression raw_input = input(cg, {NUM_FRAMES, 40}, instance.feat_vec);
        Expression inp = transpose(raw_input);

        vector<Expression> conv1_out;
        for (int i=0; i<NUM_FILTERS_CONV1; ++i) {
            conv1_out.push_back(conv1d_wide(inp, parameter(cg, p_filts1[i])));
        }

        Expression pooled = rectify(kmax_pooling(fold_rows(sum(conv1_out), 2), K_1));

        vector<Expression> conv2_out;
        for (int i=0; i<NUM_FILTERS_CONV2; ++i) {
            conv2_out.push_back(conv1d_wide(pooled, parameter(cg, p_filts2[i])));
        }

        Expression flattened = reshape(rectify(fold_rows(sum(conv2_out), 2)), {400});

        Expression embedding = parameter(cg, p_conv2embd) * flattened;

        Expression output = parameter(cg, p_embd2spid) * embedding;

        Expression loss = pickneglogsoftmax(output, instance.speaker_id);

        return loss;
    }
};


int main(int argc, char** argv) {
    dynet::initialize(argc, argv, 3640753077);
    vector<Instance> instances = read_instances("../featvec_ey/", "../featvec_ey.list");

    Model model;
    NetBuilder network(&model);
    SimpleSGDTrainer sgd(&model);

    vector<int> torder;
    for (int i = 0; i < instances.size(); ++i) {
        torder.push_back(i);
    }
    random_shuffle(torder.begin(), torder.end());

    int dev_size = 2;
    vector<int> dev(&torder[torder.size()-dev_size], &torder[torder.size()]);
    vector<int> order(&torder[0], &torder[torder.size()-dev_size]);
    int train_update_every_n           = 300;
    int dev_update_every_n             = 600;
    float total_loss                   = 0;
    float total_loss_since_last_update = 0;
    float best_loss                    = 1000000;
    int iter                           = -1;
    int i                              = -1;

    cout << "Training...\n";

    while(true) {
      ++iter;
      ++i;
      if (i == order.size()) {
        i = 0;
        cerr << "**SHUFFLE\n";
        random_shuffle(order.begin(), order.end());
      }

      // Dev update
      if ( (i % dev_update_every_n) == (dev_update_every_n - 1) ) {
        //test each task on each instance in the dev set
        float dev_loss = 0;
        float num_tests = 0;
        for (int j = 0; j < dev.size(); ++j) {
            ComputationGraph cg;
            Instance instance = instances[dev[j]];

            Expression loss = network.get_loss(cg, instance);
            float lp = as_scalar(cg.incremental_forward(loss));
            dev_loss += lp;
            num_tests += 1;
        }
        cerr << "Iter: " << iter << " dev update: avg loss per instance : "
            << dev_loss/num_tests << endl;

        // Save model
        // if (dev_loss < best_loss) {
          // cerr << "saving model" << endl;
          // best_loss = dev_loss;
          // ofstream out("best_model");
          // boost::archive::text_oarchive oa(out);
          // oa << model;
        // }
      }

      // Training
      ComputationGraph cg;
      Instance instance = instances[order[i]];

      Expression loss = network.get_loss(cg, instance);

      float lp = as_scalar(cg.incremental_forward(loss));
      total_loss += lp;
      total_loss_since_last_update += lp;
      cg.backward(loss);
      sgd.update(0.001);

      if ( (i % train_update_every_n) == (train_update_every_n - 1) ) {
        cerr << "Iter: " << iter << " through " << i << " instances out of "  << instances.size() <<
            " total, avg loss since last update: " << total_loss_since_last_update/train_update_every_n << endl;
        total_loss_since_last_update = 0;
      }
    }
}
