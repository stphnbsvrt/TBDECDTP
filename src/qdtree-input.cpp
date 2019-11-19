#include "qdtree-input.hpp"
#include <iostream>
#include <fstream>
#include <json/json.h>

namespace qdt {

po::variables_map parseCommandline(int argc, char** argv) {
    po::options_description desc("QDT options");
    desc.add_options()
        ("help", "print options info")
        ("json_input", po::value<std::string>(), "json data to input")
        ("dump_data", "Dump the data elements after reading them into the app")
        ("forest_size", po::value<uint32_t>(), "Number of trees to use in the ensemble predictors")
        ("tree_height", po::value<uint32_t>(), "Height of trees to create with random generation")
        ("population_size", po::value<uint32_t>(), "Population size to use for genetic programming algorithms")
        ("num_generations", po::value<uint32_t>(), "Number of generations to execute genetic programming algorithms")
        ("pruning_factor", po::value<float>(), "Percentage of the training sample required for all nodes in final trees")
        ("bc_bins", po::value<uint>(), "Number of equivalence classes to use for thresholds when comparing decisions")
        ("num_folds", po::value<uint32_t>(), "Number of folds to use for cross validation")
    ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
    }
    catch(std::exception& e) {
        std::cout << e.what() << std::endl;
        std::cout << desc << std::endl;
        exit(1);
    }
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        exit(0);
    }
    return vm;
}

std::vector<DataElem> parseJson(std::string json_file) {
    
    std::unordered_map<uint64_t, DataElem> data;
    std::ifstream ifs(json_file);
    Json::CharReaderBuilder reader_builder;
    Json::Value obj;
    std::string errs;
    Json::parseFromStream(reader_builder, ifs, &obj, &errs);

    // Group features by ID key
    auto members = obj.getMemberNames();
    for (auto member : members) {
        auto members2 = obj[member].getMemberNames();
        for (auto member2 : members2) {
            uint64_t id = std::stoi(member2);
            if (data.end() == data.find(id)) {
                data.emplace(id, DataElem(id));
            }
            if (member == "classification") {
                data.at(id).output = obj[member][member2].asFloat();
            }
            else {
                data.at(id).features.emplace(member, obj[member][member2].asFloat());
            }
        }
    }

    // Move to a vector
    std::vector<DataElem> data_vec;
    for (auto it = data.begin(); it != data.end(); it++) {
        data_vec.push_back(it->second);
    }
    return data_vec;
}

std::vector<DataSet> createDataSets(const std::vector<DataElem>& data, po::variables_map args) {

    uint32_t num_folds = DEFAULT_NUM_FOLDS;
    if (0 != args.count("num_folds")) {
        num_folds = args.at("num_folds").as<uint32_t>();
    }
    (void)num_folds;

    // First group by label
    std::unordered_map<float, std::vector<const DataElem*>> groups;
    for (auto& entry : data) {
        groups[entry.output].push_back(&entry);
    }

    // Shuffle them then distribute round robin
    std::vector<std::vector<const DataElem*>> data_folds = std::vector<std::vector<const DataElem*>>(num_folds, std::vector<const DataElem*>());
    for (auto& group : groups) {
        std::random_shuffle(group.second.begin(), group.second.end());
        for (uint32_t i = 0; i < group.second.size(); i++) {
            data_folds[i % data_folds.size()].push_back(group.second[i]);
        }
    }

    // Create data sets
    std::vector<DataSet> data_sets = std::vector<DataSet>(num_folds, DataSet());
    for (uint32_t i = 0; i < data_sets.size(); i++) {
        for (uint32_t j = 0; j < data_folds.size(); j++) {
            if (j == i) {
                data_sets[i].training_data.insert(data_sets[i].training_data.begin(), data_folds[j].begin(), data_folds[j].end());
            }
            else {
                data_sets[i].testing_data.insert(data_sets[i].testing_data.begin(), data_folds[j].begin(), data_folds[j].end());
            }
        }
    }

    return data_sets;
}

}