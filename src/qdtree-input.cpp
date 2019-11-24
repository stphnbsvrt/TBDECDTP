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
        ("min_distance_percentage", po::value<uint32_t>(), "Percentage of training data which must be different between items in archive")
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
    std::unordered_map<std::string, std::unordered_map<std::string, float>> discrete_feature_map;
    auto members = obj.getMemberNames();
    for (auto member : members) {
        auto members2 = obj[member].getMemberNames();
        for (auto member2 : members2) {
            uint64_t id = std::stoi(member2);
            if (data.end() == data.find(id)) {
                data.emplace(id, DataElem(id));
            }
            if (member == "classification") {
                if (obj[member][member2].isDouble()) {
                    data.at(id).output = obj[member][member2].asFloat();
                }
                else {
                    std::string feature_category = obj[member][member2].asString();
                    if (discrete_feature_map[member].find(feature_category) == discrete_feature_map[member].end()) {
                        discrete_feature_map[member].insert({feature_category, discrete_feature_map[member].size()});
                    }         
                    data.at(id).output = discrete_feature_map.at(member).at(feature_category);    
                }

            }
            else {
                if (obj[member][member2].isDouble()) {
                    data.at(id).features.emplace(member, std::pair<float, bool>(obj[member][member2].asFloat(), true));
                }
                else {
                    std::string feature_category = obj[member][member2].asString();
                    if (discrete_feature_map[member].find(feature_category) == discrete_feature_map[member].end()) {
                        discrete_feature_map[member].insert({feature_category, discrete_feature_map[member].size()});
                    }
                    data.at(id).features.emplace(member, std::pair<float, bool>(discrete_feature_map.at(member).at(feature_category), false));
                }
            }
        }
    }

    // Print string->float mappings and copy to return set
    if (discrete_feature_map.size() > 0) {
        std::cout << "Mapped string feature names: " << std::endl;
        for (auto entry : discrete_feature_map) {
            DataSet::discrete_features[entry.first] = std::unordered_set<float>();
            std::cout << "  Feature: " << entry.first << std::endl;
            for (auto entry2 : entry.second) {
                DataSet::discrete_features[entry.first].insert(entry2.second);
                std::cout << "   Category " << entry2.first << " --> " << entry2.second << std::endl;
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