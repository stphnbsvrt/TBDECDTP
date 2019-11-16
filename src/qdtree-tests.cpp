#include "qdtree-tests.hpp"
#include "qdtree-input.hpp"
#include <iostream>

namespace qdt {

static float testDiversity(std::vector<std::shared_ptr<qdt::DecisionTree>> trees, po::variables_map args) {

    uint32_t bc_bins = DEFAULT_BC_BINS;
    if (0 != args.count("bc_bins")) {
        bc_bins = args.at("bc_bins").as<uint32_t>();
    }

    // Expensive! Justified by being a test rather than part of the algorithm. 
    float total = 0;
    uint32_t divisor = 0;
    for (auto tree1 : trees) {
        for (auto tree2 : trees) {
            if (tree1 != tree2) {
                total += tree1->getBehavioralCharacteristic(bc_bins)->compare(tree2->getBehavioralCharacteristic(bc_bins));
                divisor += 1;
            }
        }
    }
    return total / divisor;
}

static float testGreedyHeuristic(const std::vector<const DataElem*>& training_data, const std::vector<const DataElem*>& testing_data, po::variables_map args) {
    
    // Create a tree with greedy heuristic
    float pruning_factor = DEFAULT_PRUNING_FACTOR;
    if (0 != args.count("pruning_factor")) {
        pruning_factor = args.at("pruning_factor").as<float>();
    }

    uint32_t bc_bins = DEFAULT_BC_BINS;
    if (0 != args.count("bc_bins")) {
        bc_bins = args.at("bc_bins").as<uint32_t>();
    }

    std::shared_ptr<qdt::DecisionTree> tree = qdt::DecisionTree::greedyTrain(training_data, pruning_factor);
    std::cout << "Tree BC: " << std::endl << tree->getBehavioralCharacteristic(bc_bins)->toStr() << std::endl;
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

    float pruning_factor = DEFAULT_PRUNING_FACTOR;
    if (0 != args.count("pruning_factor")) {
        pruning_factor = args.at("pruning_factor").as<float>();
    }

    // Create trees with greedy heuristic and bagged data
    std::vector<std::vector<const DataElem*>> bagging_data;
    std::vector<std::shared_ptr<qdt::DecisionTree>> trees;
    for (uint32_t i = 0; i < forest_size; i++) {
        bagging_data.push_back(std::vector<const DataElem*>());
        for (uint32_t j = 0; j < training_data.size(); j++) {
            bagging_data[i].push_back(training_data[rand() % training_data.size()]);
        }
        trees.push_back(qdt::DecisionTree::greedyTrain(bagging_data[i], pruning_factor));
    }
    std::cout << "ensemble diversity = " << testDiversity(trees, args) << std::endl;
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

    float pruning_factor = DEFAULT_PRUNING_FACTOR;
    if (0 != args.count("pruning_factor")) {
        pruning_factor = args.at("pruning_factor").as<float>();
    }

    uint32_t bc_bins = DEFAULT_BC_BINS;
    if (0 != args.count("bc_bins")) {
        bc_bins = args.at("bc_bins").as<uint32_t>();
    }

    std::shared_ptr<qdt::DecisionTree> tree = qdt::DecisionTree::randomTrain(training_data, tree_height, pruning_factor);
    std::cout << "Tree BC: " << std::endl << tree->getBehavioralCharacteristic(bc_bins)->toStr() << std::endl;
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

    float pruning_factor = DEFAULT_PRUNING_FACTOR;
    if (0 != args.count("pruning_factor")) {
        pruning_factor = args.at("pruning_factor").as<float>();
    }

    std::vector<std::shared_ptr<qdt::DecisionTree>> trees;
    for (uint32_t i = 0; i < forest_size; i++) {
        trees.push_back(qdt::DecisionTree::randomTrain(training_data, tree_height, pruning_factor));
    }

    // Test the ensemble
    std::cout << "ensemble diversity = " << testDiversity(trees, args) << std::endl;
    return qdt::DecisionTree::testEnsembleAccuracy(trees, testing_data, false);
}

void testCompleteRandomEnsemble(const std::vector<DataSet>& data, po::variables_map args) {

    float total = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        total += testCompleteRandomEnsemble(data[i].training_data, data[i].testing_data, args);
    }
    std::cout << "Got random ensemble accuracy: " << total/data.size() << std::endl;
}
 
static float testGeneticSingle(const std::vector<const DataElem*>& training_data, const std::vector<const DataElem*>& testing_data, po::variables_map args) {
    
    uint32_t tree_height = DEFAULT_TREE_HEIGHT;
    if (0 != args.count("tree_height")) {
        tree_height = args.at("tree_height").as<uint32_t>();
    }

    uint32_t population_size = DEFAULT_POPULATION_SIZE;
    if (0 != args.count("population_size")) {
        population_size = args.at("population_size").as<uint32_t>();
    }

    uint32_t num_generations = DEFAULT_NUM_GENERATIONS;
    if (0 != args.count("num_generations")) {
        num_generations = args.at("num_generations").as<uint32_t>();
    }
    
    float pruning_factor = DEFAULT_PRUNING_FACTOR;
    if (0 != args.count("pruning_factor")) {
        pruning_factor = args.at("pruning_factor").as<float>();
    }

    uint32_t bc_bins = DEFAULT_BC_BINS;
    if (0 != args.count("bc_bins")) {
        bc_bins = args.at("bc_bins").as<uint32_t>();
    }

    std::shared_ptr<qdt::DecisionTree> tree = qdt::DecisionTree::geneticProgrammingTrain(training_data, tree_height, population_size, num_generations);
    tree->prune(training_data, pruning_factor);
    std::cout << "Tree BC: " << std::endl << tree->getBehavioralCharacteristic(bc_bins)->toStr() << std::endl;
    return tree->testAccuracy(testing_data, false);
}

void testGeneticSingle(const std::vector<DataSet>& data, po::variables_map args) {

    float total = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        total += testGeneticSingle(data[i].training_data, data[i].testing_data, args);
    }
    std::cout << "Got genetic single accuracy: " << total/data.size() << std::endl;
}

static float testGeneticEnsemble(const std::vector<const DataElem*>& training_data, const std::vector<const DataElem*>& testing_data, po::variables_map args) {

    // Create random trees
    uint32_t forest_size = DEFAULT_FOREST_SIZE;
    if (0 != args.count("forest_size")) {
        forest_size = args.at("forest_size").as<uint32_t>();
    }

    uint32_t tree_height = DEFAULT_TREE_HEIGHT;
    if (0 != args.count("tree_height")) {
        tree_height = args.at("tree_height").as<uint32_t>();
    }

    uint32_t population_size = DEFAULT_POPULATION_SIZE;
    if (0 != args.count("population_size")) {
        population_size = args.at("population_size").as<uint32_t>();
    }

    uint32_t num_generations = DEFAULT_NUM_GENERATIONS;
    if (0 != args.count("num_generations")) {
        num_generations = args.at("num_generations").as<uint32_t>();
    }
    
    float pruning_factor = DEFAULT_PRUNING_FACTOR;
    if (0 != args.count("pruning_factor")) {
        pruning_factor = args.at("pruning_factor").as<float>();
    }

    std::vector<std::shared_ptr<qdt::DecisionTree>> trees;
    for (uint32_t i = 0; i < forest_size; i++) {
        trees.push_back(qdt::DecisionTree::geneticProgrammingTrain(training_data, tree_height, population_size, num_generations));
        trees.back()->prune(training_data, pruning_factor);
    }

    // Test the ensemble
    std::cout << "ensemble diversity = " << testDiversity(trees, args) << std::endl;
    return qdt::DecisionTree::testEnsembleAccuracy(trees, testing_data, false);
}

void testGeneticEnsemble(const std::vector<DataSet>& data, po::variables_map args) {

    float total = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        total += testGeneticEnsemble(data[i].training_data, data[i].testing_data, args);
    }
    std::cout << "Got genetic ensemble accuracy: " << total/data.size() << std::endl;

}
}