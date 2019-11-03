#include "qdtree-input.hpp"
#include "qdtree-tests.hpp"
#include <iostream>


int main(int argc, char** argv) {

    // Parse args from commandline
    po::variables_map args = qdt::parseCommandline(argc, argv);

    // Get the input data
    if (0 == args.count("json_input")) {
        std::cout << "No json data input found!" << std::endl;
        return 1;
    }
    std::string json_file = args.at("json_input").as<std::string>();
    std::vector<qdt::DataElem> data = qdt::parseJson(json_file);
    if (0 == data.size()) {
        std::cout << "Didn't find data! Does the file exist?" << std::endl;
        return 1;
    }

    // Optionally dump the data
    if (args.count("dump_data")) {
        for (uint32_t i = 0; i < data.size(); i++) {
            std::cout << "Elem[" << i << "]: " << data[i].toStr() << std::endl;
        }
    }

    // Create the tests
    srand(time(NULL));
    std::vector<qdt::DataSet> data_sets = qdt::createDataSets(data, args);

    // Test greedy single tree
    qdt::testGreedyHeuristic(data_sets, args);

    // Test bagging greedy ensemble 
    qdt::testBaggingEnsemble(data_sets, args);

    // Test complete random tree
    qdt::testCompleteRandomSingle(data_sets, args);

    // Test complete random ensemble
    qdt::testCompleteRandomEnsemble(data_sets, args);

    //qdt::testEP
    //qdt::testQD
}