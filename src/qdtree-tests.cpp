#include "qdtree-tests.hpp"
#include "qdtree-input.hpp"
#include <iostream>

namespace qdt {

static float testDiversity(std::vector<std::shared_ptr<qdt::DecisionTree>> trees, const std::vector<const DataElem*>& training_data, po::variables_map args) {

    uint32_t bc_bins = DEFAULT_BC_BINS;
    if (0 != args.count("bc_bins")) {
        bc_bins = args.at("bc_bins").as<uint32_t>();
    }
    (void)bc_bins;

    // Expensive! Justified by being a test rather than part of the algorithm. 
    float total = 0;
    uint32_t divisor = 0;
    for (auto tree1 : trees) {
        for (auto tree2 : trees) {
            if (tree1 != tree2) {
                for (auto data : training_data) {
                    auto predict1 = tree1->predict(data->features);
                    auto predict2 = tree2->predict(data->features);
                                        
                    // TODO: handle regression
                    bool equal = predict1 == predict2;
                    if (!equal) {
                       total += 1;
                    }
                }
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

    std::cout << std::endl << "------Greedy heuristic test----------" << std::endl;
    float total = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        total += testGreedyHeuristic(data[i].training_data, data[i].testing_data, args);
    }
    std::cout << "Got greedy accuracy: " << total/data.size() << std::endl;
}

// returns accuracy, diversity pair
static std::pair<float, float> testBaggingEnsemble(const std::vector<const DataElem*>& training_data, const std::vector<const DataElem*>& testing_data, po::variables_map args) {

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

    auto retval = qdt::DecisionTree::testEnsembleAccuracy(trees, testing_data, false);
    return {retval, testDiversity(trees, testing_data, args)};
}

std::pair<float, float> testBaggingEnsemble(const std::vector<DataSet>& data, po::variables_map args) {
    
    std::cout << std::endl << "------Bagging ensemble test----------" << std::endl;
    std::cout << "diversity,accuracy" << std::endl;
    float total_accuracy = 0;
    float total_diversity = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        auto stats = testBaggingEnsemble(data[i].training_data, data[i].testing_data, args);
        total_accuracy += stats.first;
        total_diversity += stats.second;
    }
    std::cout << "Got cross validation bagging accuracy: " << total_accuracy/data.size() << ", diveristy " << total_diversity/data.size() << std::endl;
    return {total_accuracy/data.size(), total_diversity/data.size()};
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

    std::cout << std::endl << "------Complete random tree test----------" << std::endl;
    float total = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        total += testCompleteRandomSingle(data[i].training_data, data[i].testing_data, args);
    }
    std::cout << "Got random single accuracy: " << total/data.size() << std::endl;
}

// returns accuracy, diversity pair
static std::pair<float, float> testCompleteRandomEnsemble(const std::vector<const DataElem*>& training_data, const std::vector<const DataElem*>& testing_data, po::variables_map args) {

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
    auto retval = qdt::DecisionTree::testEnsembleAccuracy(trees, testing_data, false);
    return {retval, testDiversity(trees, testing_data, args)};
}

std::pair<float, float> testCompleteRandomEnsemble(const std::vector<DataSet>& data, po::variables_map args) {

    std::cout << std::endl << "------Complete random ensemble test----------" << std::endl;
    std::cout << "diversity,accuracy" << std::endl;
    float total_accuracy = 0;
    float total_diversity = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        auto stats = testCompleteRandomEnsemble(data[i].training_data, data[i].testing_data, args);
        total_accuracy += stats.first;
        total_diversity += stats.second;
    }
    std::cout << "Got cross validation random ensemble accuracy: " << total_accuracy/data.size() <<  ", diversity = " << total_diversity/data.size() << std::endl;
    return {total_accuracy/data.size(), total_diversity/data.size()};
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

    std::shared_ptr<qdt::DecisionTree> tree = qdt::DecisionTree::geneticProgrammingTrain(training_data, tree_height, population_size, 1, num_generations)[0];
    tree->prune(training_data, pruning_factor);
    std::cout << "Tree BC: " << std::endl << tree->getBehavioralCharacteristic(bc_bins)->toStr() << std::endl;
    return tree->testAccuracy(testing_data, false);
}

void testGeneticSingle(const std::vector<DataSet>& data, po::variables_map args) {

    std::cout << std::endl << "------Genetic single tree test----------" << std::endl;
    float total = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        total += testGeneticSingle(data[i].training_data, data[i].testing_data, args);
    }
    std::cout << "Got genetic single accuracy: " << total/data.size() << std::endl;
}

// returns accuracy, diversity pair
static std::pair<float, float> testGeneticEnsemble(const std::vector<const DataElem*>& training_data, const std::vector<const DataElem*>& testing_data, po::variables_map args) {

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

    std::vector<std::shared_ptr<qdt::DecisionTree>> trees = qdt::DecisionTree::geneticProgrammingTrain(training_data, tree_height, population_size, forest_size, num_generations);
    for (uint32_t i = 0; i < trees.size(); i++) {
        trees.back()->prune(training_data, pruning_factor);
    }

    // Test the ensemble
    auto retval = qdt::DecisionTree::testEnsembleAccuracy(trees, testing_data, false);
    return {retval, testDiversity(trees, testing_data, args)};
}

void testGeneticEnsemble(const std::vector<DataSet>& data, po::variables_map args) {

    std::cout << std::endl << "------Genetic restart ensemble tree test----------" << std::endl;
    float total_accuracy = 0;
    float total_diversity = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        auto stats = testGeneticEnsemble(data[i].training_data, data[i].testing_data, args);
        total_accuracy += stats.first;
        total_diversity += stats.second;
    }
    std::cout << "Got genetic ensemble accuracy: " << total_accuracy/data.size() << ", diversity = " << total_diversity/data.size() << std::endl;
}

// returns accuracy, diversity pair
static std::pair<float, float> testQD(const std::vector<const DataElem*>& training_data, const std::vector<const DataElem*>& testing_data, po::variables_map args) {

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

    uint32_t bc_bins = DEFAULT_BC_BINS;
    if (0 != args.count("bc_bins")) {
        bc_bins = args.at("bc_bins").as<uint32_t>();
    }

    uint32_t min_distance_percentage = DEFAULT_MIN_DISTANCE_PERCENTAGE;
    if (0 != args.count("min_distance_percentage")) {
        min_distance_percentage = args.at("min_distance_percentage").as<uint32_t>();
    }

    SelectionStrategy selection = DEFAULT_SELECTION_STRATEGY;
    if (0 != args.count("selection_strategy")) {
        selection = (SelectionStrategy)args.at("selection_strategy").as<uint32_t>();
    }

    std::vector<std::shared_ptr<qdt::DecisionTree>> trees = qdt::DecisionTree::QDTrain(training_data, tree_height, population_size, forest_size, num_generations, bc_bins, min_distance_percentage, selection);

    // Retry with larger percentage if returned 0
    while (trees.size() == 0) {
        min_distance_percentage += 1;
        std::cout << "retrying with min_distance_percentage " << min_distance_percentage << std::endl;
        trees = qdt::DecisionTree::QDTrain(training_data, tree_height, population_size, forest_size, num_generations, bc_bins, min_distance_percentage, selection);    
    }
    for (auto tree : trees) {
        tree->prune(training_data, pruning_factor);
    }

    // Test the ensemble
    auto retval = qdt::DecisionTree::testEnsembleAccuracy(trees, testing_data, false);
    return {retval, testDiversity(trees, testing_data, args)};
}

std::pair<float, float> testQD(const std::vector<DataSet>& data, po::variables_map args) {
 
    std::cout << std::endl << "------QD algorithm test----------" << std::endl;
    std::cout << "diversity,accuracy" << std::endl;
    float total_accuracy = 0;
    float total_diversity = 0;
    for (uint32_t i = 0; i < data.size(); i++) {
        auto stats = testQD(data[i].training_data, data[i].testing_data, args);
        total_accuracy += stats.first;
        total_diversity += stats.second;
    }
    std::cout << "Got QD cross validation accuracy: " << total_accuracy/data.size() << ", diversity = " << total_diversity/data.size() << std::endl;
    return {total_accuracy/data.size(), total_diversity/data.size()};
}

}