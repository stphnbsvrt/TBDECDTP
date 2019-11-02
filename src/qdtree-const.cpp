#include "qdtree-const.hpp"
#include <iostream>
#include <limits>

namespace qdt {

///
/// \brief Determines the label which would be assigned to minimize the error on the data
///
/// \param[in] data The data to consider
///
/// \return The label chosen
///

static float calculateLabelForData(const std::vector<const DataElem*>& data) {
    
    // TODO: handle regression by averaging
    std::unordered_map<float, uint64_t> outputs;
    for (auto elem : data) {
        outputs[elem->output] = outputs[elem->output] + 1;
    }

    uint64_t best_count = 0;
    float best_output = -1;
    for (auto output : outputs) {
        if (output.second > best_count) {
            best_count = output.second;
            best_output = output.first;
        }
    }
    return best_output;
}

///
/// \brief Determines the confidence for each label on the node
///
/// \param[in] data The data to consider
///
/// \return Confidence for each label
///

static std::unordered_map<float, float> calculateConfidencesForData(const std::vector<const DataElem*>& data) {
        
    // TODO: handle regression by?
    std::unordered_map<float, float> outputs;
    for (auto elem : data) {
        outputs[elem->output] = outputs[elem->output] + 1;
    }

    for (auto entry : outputs) {
        outputs[entry.first] /= data.size();
    }
    return outputs;
}

///
/// \brief Calculates the classification error of the data using the given label
///
/// \param[in] label The label to use for all data
/// \param[in] data The data to consider
///
/// \return The number of mispredicted elements
///

static float calculateClassificationError(float label, const std::vector<const DataElem*>& data) {
    
    // Using misclassification error here
    // TODO: use 
    float total_error = 0;
    for (auto elem : data) {
        if (label != elem->output) {
            total_error += 1;
        }
    }
    return total_error;
}

///
/// \brief Returns the minimum classification error on the data
///
/// \param[in] data The data to consider
///
/// \return The minimum number of mispredicted elements
///

static float calculateClassificationError(const std::vector<const DataElem*>& data) {
    return calculateClassificationError(calculateLabelForData(data), data);
}

std::shared_ptr<DecisionTree> DecisionTree::greedyTrain(const std::vector<const DataElem*>& training_data) {
    std::shared_ptr<DecisionTree> tree(new DecisionTree);
    tree->root_ = greedyTrain(training_data, nullptr, 0);
    return tree;
}

std::shared_ptr<DecisionTreeNode> DecisionTree::greedyTrain(const std::vector<const DataElem*>& training_data, std::shared_ptr<DecisionTreeNode> parent, uint64_t node_number) {
    
    // Make a new decision tree node
    auto tree_node = std::make_shared<DecisionTreeNode>();
    tree_node->node_number = node_number;
    tree_node->parent = parent;
    tree_node->label = calculateLabelForData(training_data);
    tree_node->confidences = calculateConfidencesForData(training_data);

    // TODO: handle regression error
    float error = calculateClassificationError(tree_node->label, training_data);
    
    // TODO: configure threshold
    float threshold = 0.05 * training_data.size();
    if (error < threshold) {

        // Base case
        return tree_node;
    }
    
    // Find best split
    float best_error = std::numeric_limits<float>::max();
    auto best_split = std::make_shared<Decision>();
    std::shared_ptr<std::vector<const DataElem*>> best_left_group;
    std::shared_ptr<std::vector<const DataElem*>> best_right_group;
    for (auto feature : training_data[0]->features) {
        std::string feature_key = feature.first;
        for (float split = 0.0; split < 1.0; split += 0.10) {
            auto left_group = std::make_shared<std::vector<const DataElem*>>();
            auto right_group = std::make_shared<std::vector<const DataElem*>>();
            for (auto elem : training_data) {
                if (elem->features.at(feature_key) < split) {
                    left_group->push_back(elem);
                }
                else {
                    right_group->push_back(elem);
                }
            }
            float new_error = calculateClassificationError(*left_group) + calculateClassificationError(*right_group);
            if (new_error < best_error) {
                best_error = new_error;
                best_split->feature = feature_key;
                best_split->threshold = split;
                best_left_group = left_group;
                best_right_group = right_group;
            }
        }
    }

    // If a best group is size 0, we're unseparable
    if ((0 == best_left_group->size()) || (0 == best_right_group->size())) {
        return tree_node;
    }

    tree_node->decision = best_split;
    tree_node->l_child = greedyTrain(*best_left_group, tree_node, (2 * tree_node->node_number) + 1);
    tree_node->r_child = greedyTrain(*best_right_group, tree_node, (2 * tree_node->node_number) + 2);
    return tree_node;
}

float DecisionTree::testAccuracy(const std::vector<const DataElem*>& testing_data, bool verbose) {

    uint64_t num_correct = 0;
    for (auto elem : testing_data) {
        float predicted_val = predict(elem->features);
        if (elem->output == predicted_val) {
            num_correct++;
        }
        else {
            if (verbose) {
                std::cout << "mispredicted " << elem->output << " as " << predicted_val << std::endl; 
            }
        }
    }
    return (float)num_correct / testing_data.size();
}

float DecisionTree::testEnsembleAccuracy(const std::vector<std::shared_ptr<DecisionTree>>& ensemble, const std::vector<const DataElem*>& testing_data, bool verbose) {

    uint64_t num_correct = 0;
    for (auto elem : testing_data) {

        // Create votes
        // FIXME: use sum rule
        std::unordered_map<float, float> votes;
        for (auto tree : ensemble) {
            auto confidence = tree->predictConfidence(elem->features);
            for (auto entry : confidence) {
                votes[entry.first] += entry.second;
            }
        }

        // Pick winner
        float prediction = -1;
        float prediction_confidence = std::numeric_limits<float>::min();
        for (auto entry : votes) {
            if (entry.second > prediction_confidence) {
                prediction = entry.first;
                prediction_confidence = entry.second;
            }
        }

        // Check correctness
        if (elem->output == prediction) {
            num_correct++;
        }
        else {
            if (verbose) {
                std::cout << "mispredicted " << elem->output << " as " << prediction << std::endl; 
            }
        }

    }
    return (float)num_correct / testing_data.size();
}

float DecisionTree::predict(const std::unordered_map<std::string, float> features) {

    auto cur_node = getRoot();
    while (cur_node->decision != nullptr) {
        if (cur_node->decision->test(features)) {
            cur_node = cur_node->l_child;
        }
        else {
            cur_node = cur_node->r_child;
        }
    }
    return cur_node->label;
}


std::unordered_map<float, float> DecisionTree::predictConfidence(const std::unordered_map<std::string, float> features) {

    auto cur_node = getRoot();
    while (cur_node->decision != nullptr) {
        if (cur_node->decision->test(features)) {
            cur_node = cur_node->l_child;
        }
        else {
            cur_node = cur_node->r_child;
        }
    }
    return cur_node->confidences;
}
}