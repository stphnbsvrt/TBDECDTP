#include <json/json.h>
#include <fstream>
#include <iostream>
#include <unordered_map>

struct DataElem {
    DataElem(uint64_t id) : id(id) {}
    const uint64_t id;
    float output;
    std::unordered_map<std::string, float> features;
};

int main(int argc, char** argv) {

    std::unordered_map<uint64_t, DataElem> data;
    std::ifstream ifs("./Data/BreastCancer/data.json");
    Json::CharReaderBuilder reader_builder;
    Json::Value obj;
    std::string errs;
    Json::parseFromStream(reader_builder, ifs, &obj, &errs);

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
    for (auto elem : data) {
        std::cout << "Data id: " << elem.first << "/" << elem.second.id << std::endl;
        std::cout << "    Output: " << elem.second.output << std::endl;
        for (auto feature : elem.second.features) {
            std::cout << "    " << feature.first << ":" << feature.second << std::endl;
        }
    }
    return 0;
}