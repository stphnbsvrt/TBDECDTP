#include "qdtree-tests.hpp"
#include "qdtree-input.hpp"
#include <iostream>

namespace qdt {

static float testGreedyHeuristic(const std::vector<const DataElem*>& training_data, const std::vector<const DataElem*>& testing_data, po::variables_map args) {
    
    // Create a tree with greedy heuristic
    (void)args;
    std::shared_ptr<qdt::DecisionTree> tree = qdt::DecisionTree::greedyTrain(training_data);
    return tree->testAccuracy(testing_data, false);
}

void testGreedyHeuristic(const std::vector<DataSet>& data, po::variables_map args) {

    float total = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        total += testGreedyHeuristic(data[i].training_data, data[i].testing_data, args);
    }
    std::cout << "Got greedy accuracy: " << total/data.size() << std::endl;
}

static float testBaggingEnsemble(const std::vector<const DataElem*>& training_data, const std::vector<const DataElem*>& testing_data, po::variables_map args) {

    uint32_t forest_size = DEFAULT_FOREST_SIZE;
    if (0 != args.count("forest_size")) {
        forest_size = args.at("forest_size").as<uint32_t>();
    }

    // Create trees with greedy heuristic and bagged data
    std::vector<std::vector<const DataElem*>> bagging_data;
    std::vector<std::shared_ptr<qdt::DecisionTree>> trees;
    for (uint32_t i = 0; i < forest_size; i++) {
        bagging_data.push_back(std::vector<const DataElem*>());
        for (uint32_t j = 0; j < training_data.size(); j++) {
            bagging_data[i].push_back(training_data[rand() % training_data.size()]);
        }
        trees.push_back(qdt::DecisionTree::greedyTrain(bagging_data[i]));
    }
    return qdt::DecisionTree::testEnsembleAccuracy(trees, testing_data, false);
}

void testBaggingEnsemble(const std::vector<DataSet>& data, po::variables_map args) {

    float total = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        total += testBaggingEnsemble(data[i].training_data, data[i].testing_data, args);
    }
    std::cout << "Got bagging accuracy: " << total/data.size() << std::endl;
}
 
static float testCompleteRandomSingle(const std::vector<const DataElem*>& training_data, const std::vector<const DataElem*>& testing_data, po::variables_map args) {
    
    uint32_t tree_height = DEFAULT_TREE_HEIGHT;
    if (0 != args.count("tree_height")) {
        tree_height = args.at("tree_height").as<uint32_t>();
    }

    std::shared_ptr<qdt::DecisionTree> tree = qdt::DecisionTree::randomTrain(training_data, tree_height);
    return tree->testAccuracy(testing_data, false);
}

void testCompleteRandomSingle(const std::vector<DataSet>& data, po::variables_map args) {

    float total = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        total += testCompleteRandomSingle(data[i].training_data, data[i].testing_data, args);
    }
    std::cout << "Got random single accuracy: " << total/data.size() << std::endl;
}

static float testCompleteRandomEnsemble(const std::vector<const DataElem*>& training_data, const std::vector<const DataElem*>& testing_data, po::variables_map args) {

    // Create random trees
    uint32_t forest_size = DEFAULT_FOREST_SIZE;
    if (0 != args.count("forest_size")) {
        forest_size = args.at("forest_size").as<uint32_t>();
    }

    uint32_t tree_height = DEFAULT_TREE_HEIGHT;
    if (0 != args.count("tree_height")) {
        tree_height = args.at("tree_height").as<uint32_t>();
    }

    std::vector<std::shared_ptr<qdt::DecisionTree>> trees;
    for (uint32_t i = 0; i < forest_size; i++) {
        trees.push_back(qdt::DecisionTree::randomTrain(training_data, tree_height));
    }

    // Test the ensemble
    return qdt::DecisionTree::testEnsembleAccuracy(trees, testing_data, false);
}

void testCompleteRandomEnsemble(const std::vector<DataSet>& data, po::variables_map args) {

    float total = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        total += testCompleteRandomEnsemble(data[i].training_data, data[i].testing_data, args);
    }
    std::cout << "Got random ensemble accuracy: " << total/data.size() << std::endl;
}

}