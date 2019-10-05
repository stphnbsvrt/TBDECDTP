#include "qdtree-tests.hpp"
#include <iostream>

namespace qdt {

void testGreedyHeuristic(const std::vector<DataElem>& data, po::variables_map args) {

    // TODO: parse experiment parameters from args
    // For now, create a single training/testing set pair
    (void)args;
    int training_chance = 66;
    std::vector<DataElem> training_data;
    std::vector<DataElem> testing_data;
    srand(time(NULL));
    for (auto elem : data) {
        if ((rand() % 100) < training_chance) {
            training_data.push_back(elem);
        }
        else {
            testing_data.push_back(elem);
        }
    }

    // Create a tree with greedy heuristic
    std::shared_ptr<qdt::DecisionTree> tree = qdt::DecisionTree::greedyTrain(training_data);
    std::cout << "Got greedy accuracy: " << tree->testAccuracy(testing_data, false) << std::endl;
}

}