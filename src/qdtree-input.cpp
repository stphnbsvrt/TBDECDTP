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
    ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
    }
    catch(boost::wrapexcept<po::unknown_option>& e) {
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
}