#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

bool TO_TRAIN = true;
string model_name = "model.model";
unsigned EMBEDDING_SIZE = 1024;
unsigned CONV_OUTPUT_DIM = 400;
unsigned GLOVE_DIM = 50;
unsigned CONV1_FILTERS = 10;
unsigned CONV1_SIZE = 9;
unsigned CONV2_FILTERS = 5;
unsigned CONV2_SIZE = 5;
unsigned ROWS1 = 40;
unsigned ROWS2 = 20;
unsigned K_1 = 36;
float WORD_TASK_PROB = .8;
//unsigned K_2 = 36;


unsigned INPUT_SIZE = 100;
unsigned FBANK_DIM = 40;

Dict speaker_d;
Dict education_d;
Dict dialect_d;
Dict word_d;

string strip_string(string s, string to_strip) {
  size_t str_begin = s.find_first_not_of(to_strip);
  size_t str_end = s.find_last_not_of(to_strip);
  size_t str_range = str_end - str_begin + 1;
  if (str_begin == string::npos || str_end == string::npos || str_range <= 0) {
    return "";
  }
  return s.substr(str_begin, str_range);
}


class Speaker {
  public:
    string speaker_id;
    bool gender; // false = male, true = female
    int age;
    string education;
    string dialect; 
};

class Instance {
  public:
    string input_filename; // name of file containing data for the instance
    string word;
    vector<float> gloVe_sem_vector;
    Speaker speaker;
    bool filled;

    void fill_instance(unordered_map<string, Speaker> speakers_info,
        unordered_map<string, vector<float>> word_to_gloVe);

    vector<float> read_vec();
};

enum Task { WORD=0, SEM_SIMILARITY=1, SPEAKER_ID=2, GENDER=3, AGE=4, EDUCATION=5, DIALECT=6 };

void Instance::fill_instance(unordered_map<string, Speaker> speakers_info,
    unordered_map<string, vector<float>> word_to_gloVe) {
  ifstream in(input_filename);
  {
    string w;
    string line;
    getline(in, w);
    transform(w.begin(), w.end(), w.begin(), ::tolower);
    word = w;
    gloVe_sem_vector = word_to_gloVe[word];

    string speaker_str;
    string speaker_id;
    getline(in, speaker_str);
    speaker_id = speaker_str;
    speaker = speakers_info[speaker_id];

    filled = true;
  }
  in.close();
}

vector<float> Instance::read_vec() {
  vector<float> input_vector;
  ifstream in(input_filename);
  {
    string w;
    string line;
    string substr;
    string speaker_str;
    getline(in, w);
    getline(in, speaker_str);
    vector<float> instance_vector;
    int count = 0;
    while(getline(in, line)) {
      // Cutting
      if (count >= INPUT_SIZE) {
        break;
      }

      istringstream iss(line);
      line = strip_string(line, " ");
      while(getline(iss, substr, ' ')) {
        instance_vector.push_back(stof(substr));
      }
      count += 1;
    }

    vector<float> padded_vector;

    // Padding
    int to_pad = INPUT_SIZE - count;
    if (to_pad >= 1) {
      for (int i = 0; i < to_pad/2; ++i) {
        for (int j = 0; j < FBANK_DIM; ++j) {
          padded_vector.push_back(0.0);
        }
        count += 1;
      }

      for (int i = 0; i < instance_vector.size(); ++i) {
        padded_vector.push_back(instance_vector[i]);
      }

      while (count < INPUT_SIZE) {
        for (int j = 0; j < FBANK_DIM; ++j) {
          padded_vector.push_back(0.0);
        }
        count += 1;
      }
    }
    else {
      padded_vector = instance_vector;
    }

    input_vector = padded_vector;
  }
  in.close();
  return input_vector;
}

unordered_map<string, Speaker> load_speakers(string speaker_filename) {
  unordered_map<string, Speaker> speakers_info;
  ifstream in(speaker_filename);
  {
    cerr << "Reading speaker data from " << speaker_filename << " ...\n";
    string line;
    string substr;
    string speaker_id;
    bool gender;
    int age;
    string dialect;
    string education;

    while(getline(in, line)) {
      istringstream iss(line);
      vector<string> vect;
      while(getline(iss, substr, ',')) {
        vect.push_back(substr);
      }

      speaker_id = strip_string(vect[0], "\" ");

      string gender_str = strip_string(vect[3], "\" ");
      if (gender_str.compare("MALE") == 0) {
        gender = false;
      }
      else {
        gender = true;
      }

      string age_str = strip_string(vect[4], "\" ");
      age = 1997 - stoi(age_str);

      dialect = strip_string(vect[5], "\" ");
      if (dialect.compare("") == 0) {
        dialect = "UNK";
      }

      education = strip_string(vect[6], "\" ");

      Speaker si;
      si.speaker_id = speaker_id;
      si.gender = gender;
      si.age = age;
      si.dialect = dialect;
      si.education = education;

      speaker_d.convert(speaker_id);
      dialect_d.convert(dialect);
      education_d.convert(education);

      speakers_info[speaker_id] = si;
    }
  }
  in.close();
  return speakers_info;
}

void read_vocab(string vocab_filename) {
  ifstream in(vocab_filename);
  {
    cerr << "Reading vocab from " << vocab_filename << "...\n";
    string word;
    string line;
    string w;
    while(getline(in, line)) {
      word = line.substr(0, line.find_first_of(','));
      transform(w.begin(), w.end(), w.begin(), ::tolower);
      word_d.convert(word);
    }
  }
  in.close();
}

unordered_map<string, vector<float>> load_gloVe_vectors(string gloVe_filename) {
  unordered_map<string, vector<float>> word_to_gloVe;
  ifstream in(gloVe_filename);
  {
    cerr << "Reading gloVe vectors from " << gloVe_filename << "...\n";
    string line;
    string word;
    while(getline(in, line)) {
      istringstream iss(line);
      getline(iss, word, ' ');
      if (word_d.convert(word_d.convert(word)).compare("UNK") == 0) {
        continue;
      }

      vector<float> gloVe;
      string substr;
      while(getline(iss, substr, ' ')) {
        gloVe.push_back(stof(substr));
      }
      word_to_gloVe[word] = gloVe;
    }
  }
  in.close();
  return word_to_gloVe;
}

vector<Instance> read_instances(string instances_filename,
    unordered_map<string, Speaker> speakers_info,
    unordered_map<string, vector<float>> word_to_gloVe) {
  vector<Instance> instances;
  ifstream in(instances_filename);
  {
    cerr << "Reading instances data from " << instances_filename << "...\n";
    string line;
    while(getline(in, line)) {
      Instance instance;
      instance.input_filename = ("../data/"+line);
      instance.filled = false;
      instances.push_back(instance);
    }
  }
  in.close();
  return instances;
}

Expression calc_loss_with_forward(ComputationGraph& cg, Expression embedding, Parameter param, int correct) {
  Expression output = parameter(cg, param)*embedding;
  Expression loss = pickneglogsoftmax(output, correct);
  return loss;
}

struct MTLBuilder {
  vector<Parameter> p_ifilts; //filters for the 1dconv over the input
  vector<Parameter> p_cfilts; //filters for the 1dconv over the (altered) output of the first convolution
  Parameter p_c2we; //the output of the convolution to the word embedding
  Parameter p_we2sr; //the word embedding to speech recognition
  Parameter p_we2ss; //word embedding to the semantic similarity
  Parameter p_we2id; //word embedding to the speaker id
  Parameter p_we2gen; //word embedding to gender
  Parameter p_we2age; //word embedding to age
  Parameter p_we2edu; //word embedding to education
  Parameter p_we2dia;  //word embedding to dialect

  explicit MTLBuilder(Model* model) :
    p_c2we(model->add_parameters({EMBEDDING_SIZE, CONV_OUTPUT_DIM})),
    p_we2sr(model->add_parameters({word_d.size(), EMBEDDING_SIZE})),
    p_we2ss(model->add_parameters({GLOVE_DIM, EMBEDDING_SIZE})),
    p_we2id(model->add_parameters({speaker_d.size(), EMBEDDING_SIZE})),
    p_we2gen(model->add_parameters({2, EMBEDDING_SIZE})),
    p_we2age(model->add_parameters({1, EMBEDDING_SIZE})),
    p_we2edu(model->add_parameters({education_d.size(), EMBEDDING_SIZE})),
    p_we2dia(model->add_parameters({dialect_d.size(), EMBEDDING_SIZE})) {
      for (int i = 0; i < CONV1_FILTERS; ++i) {
        p_ifilts.push_back(model->add_parameters({ROWS1, CONV1_SIZE}));
      }
      for (int i = 0; i < CONV2_FILTERS; ++i) {
        p_cfilts.push_back(model->add_parameters({ROWS2, CONV2_SIZE}));
      }
    }


    Expression loss_against_task(ComputationGraph& cg, Task task, Instance instance) {
      vector<float> fb = instance.read_vec();
      unsigned fb_size = fb.size();
      Expression raw_input = input(cg, {fb_size/40, 40}, fb);
      Expression inp = transpose(raw_input);
      vector<float> blah = as_vector(cg.incremental_forward(inp));

      vector<Expression> conv1_out;
      for (int i = 0; i < CONV1_FILTERS; ++i) {
        conv1_out.push_back(conv1d_wide(inp, parameter(cg, p_ifilts[i])));
      }

      Expression s = rectify(kmax_pooling(fold_rows(sum(conv1_out),2), K_1));
      vector<Expression> conv2_out;
      for (int i = 0; i < CONV2_FILTERS; ++i) {
        conv2_out.push_back(conv1d_wide(s, parameter(cg, p_cfilts[i])));
      }

      Expression t = rectify(fold_rows(sum(conv2_out),2));
      Expression flattened_t = reshape(t, {400});

      Expression embedding = parameter(cg, p_c2we)*flattened_t;


      Expression loss;
      if (task == SEM_SIMILARITY) {
        Expression output =parameter(cg, p_we2ss)*embedding;
        loss = squared_distance(output, input(cg, {GLOVE_DIM}, instance.gloVe_sem_vector));
      }
      else if (task == AGE) {
        Expression output = parameter(cg, p_we2age)*embedding;
        loss = squared_distance(output, input(cg, instance.speaker.age));
      }
      else if (task == GENDER) {
        loss = calc_loss_with_forward(cg, embedding, p_we2gen, instance.speaker.gender);
      }
      else if (task == WORD) {
        loss = calc_loss_with_forward(cg, embedding, p_we2sr, word_d.convert(instance.word));
      }
      else if (task == SPEAKER_ID) {
        loss = calc_loss_with_forward(cg, embedding, p_we2id, speaker_d.convert(instance.speaker.speaker_id));
      }
      else if (task == EDUCATION) {
        loss = calc_loss_with_forward(cg, embedding, p_we2edu, education_d.convert(instance.speaker.education));
      }
      else {
        assert(task == DIALECT);
        loss = calc_loss_with_forward(cg, embedding, p_we2dia, dialect_d.convert(instance.speaker.dialect));
      }
      return loss;
    }
};

int main(int argc, char** argv) {
  dynet::initialize(argc, argv, 3640753077);

  unordered_map<string, Speaker> speakers_info = load_speakers("../caller_tab.csv");
  speaker_d.freeze();
  education_d.freeze();
  dialect_d.freeze();

  speaker_d.set_unk("UNK");
  education_d.set_unk("UNK");
  dialect_d.set_unk("UNK");

  read_vocab("../word_dict.txt");
  word_d.freeze();
  word_d.set_unk("UNK");

  unordered_map<string, vector<float>> word_to_gloVe =
      load_gloVe_vectors("../gloVe.6B.50d.txt");
  vector<Instance> instances = read_instances("../word_feat.filelist",
      speakers_info, word_to_gloVe);

  Model model;
  MTLBuilder mtl(&model);
  SimpleSGDTrainer sgd(&model);


  vector<int> torder;

  for (int i = 0; i < instances.size(); ++i) {
    torder.push_back(i);
  }

  random_shuffle(torder.begin(), torder.end());

  int dev_size = 2;
  vector<int> dev(&torder[torder.size()-dev_size], &torder[torder.size()]);
  vector<int> order(&torder[0], &torder[torder.size()-dev_size]);
  int iter = -1;
  int dev_update_every_n = 600;
  int train_update_every_n = 300;
  float total_loss = 0;
  float total_loss_since_last_update = 0;
  int i = -1;
  float best_loss = 1000000;


  cout << "Training...\n";
  if (TO_TRAIN) {
    while(true) {
      ++iter;
      ++i;
      if (i == order.size()) {
        i = 0;
        cerr << "**SHUFFLE\n";
        random_shuffle(order.begin(), order.end());
      }

      // dev update
      if (i% dev_update_every_n == dev_update_every_n-1) {
        //test each task on each instance in the dev set
        float dev_loss = 0;
        vector<float> task_losses(7,0);

        float num_tests = 0;
        for (int j = 0; j < dev.size(); ++j) {
          for (int k = 0; k < 7; ++k) {
            ComputationGraph cg;  
            Instance instance = instances[dev[j]];
            if (!instance.filled) {
              instance.fill_instance(speakers_info, word_to_gloVe);
            }

            if (k == SEM_SIMILARITY && instance.gloVe_sem_vector.size() != GLOVE_DIM) {
              continue;
            }

            Task task = static_cast<Task>(k);
            Expression loss = mtl.loss_against_task(cg, task, instance);
            float lp = as_scalar(cg.incremental_forward(loss));
            dev_loss += lp;
            task_losses[k] += lp;
            num_tests += 1;
          }

        }
        cerr << "dev update: avg loss per instance : " << dev_loss/num_tests << endl;
        for (int k = 0; k < 7; ++k) {
          cerr << "dev loss on task " << k << ": " << task_losses[k]/dev_size << endl;
        }

        //model saving
        if (dev_loss < best_loss) {
          cerr << "saving model" << endl;
          best_loss = dev_loss;
          ofstream out(model_name);
          boost::archive::text_oarchive oa(out);
          oa << model;
        }
      }

      // Training
      ComputationGraph cg;  
      Instance instance = instances[order[i]];
      if (!instance.filled) {
        instance.fill_instance(speakers_info, word_to_gloVe);
      }

      float r = static_cast<float>(rand())/static_cast<float>(RAND_MAX);
      Task task;
      if (r < WORD_TASK_PROB) {
        task = WORD;
      }
      else {
        if (instance.gloVe_sem_vector.size() != GLOVE_DIM) {
          task = static_cast<Task>(rand()%5+2);
        }
        else {
          task = static_cast<Task>(rand()%6+1);
        }
      }
      Expression loss = mtl.loss_against_task(cg, task, instance);

      float lp = as_scalar(cg.incremental_forward(loss));
      total_loss += lp;
      total_loss_since_last_update += lp;
      cout << instance.word << " " << task << endl;
      cg.backward(loss);
      sgd.update(0.001);

      if (i%train_update_every_n == train_update_every_n-1) {
        cerr << "through " << i << " instances out of "  << instances.size() <<
            " total, avg loss since last update: " << total_loss_since_last_update/train_update_every_n << endl;
        total_loss_since_last_update = 0;
      }
    }
  }
}
