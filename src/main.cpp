#include "qdtree-input.hpp"
#include "qdtree-tests.hpp"
#include <iostream>
#include <fstream>


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

    // Find test type
    if (0 == args.count("test_type")) {
        std::cout << "No test type found!" << std::endl;
        return 1;
    }
    qdt::TestType test_type = (qdt::TestType)args.at("test_type").as<uint32_t>();

    // Optionally dump the data
    if (args.count("dump_data")) {
        for (uint32_t i = 0; i < data.size(); i++) {
            std::cout << "Elem[" << i << "]: " << data[i].toStr() << std::endl;
        }
    }

    // Create the tests
    srand(time(NULL));

    uint32_t num_iterations = 1;
    if (args.count("iterations")) {
        num_iterations = args.at("iterations").as<uint32_t>();
    }

    // Output stats to csv
    if (0 == args.count("csv_out")) {
        std::cout << "no csv output found!" << std::endl;
        return 1;
    }
    std::ofstream csv_out(args.at("csv_out").as<std::string>());
    csv_out << "accuracy,diversity" << std::endl;
    for (uint32_t i = 0; i < num_iterations; i++) {
    std::vector<qdt::DataSet> data_sets = qdt::createDataSets(data, args);

        switch(test_type) {
            case qdt::TestType::BAGGING: {
                auto stats = qdt::testBaggingEnsemble(data_sets, args);
                csv_out << stats.first << "," << stats.second << std::endl;
                break;
            }
            case qdt::TestType::RANDOM: {
                auto stats = qdt::testCompleteRandomEnsemble(data_sets, args);
                csv_out << stats.first << "," << stats.second << std::endl;
                break;
            }
            case qdt::TestType::QD: {
                auto stats = qdt::testQD(data_sets, args);
                csv_out << stats.first << "," << stats.second << std::endl;
                break;
            }
            default:
                std::cout << "Unknown test type!" << std::endl;
                return 1;
                // Leftovers 
                // Test greedy single tree
                //qdt::testGreedyHeuristic(data_sets, args);

                // Test complete random tree
                //qdt::testCompleteRandomSingle(data_sets, args);

                // Test genetic programming tree
                //qdt::testGeneticSingle(data_sets, args);

                // Test genetic programming ensemble
                //qdt::testGeneticEnsemble(data_sets, args);
        
        }


    }
}