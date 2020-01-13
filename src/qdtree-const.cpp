#include "qdtree-const.hpp"
#include <iostream>
#include <random>
#include <unordered_set>
#include <algorithm>

namespace qdt {

std::unordered_map<std::string, std::unordered_set<float>> DataSet::discrete_features;

bool operator==(const qdt::Decision& lhs, const qdt::Decision& rhs) { 
    return ((lhs.feature == rhs.feature) && (lhs.threshold == rhs.threshold));
}

bool operator==(const qdt::DecisionTreeBehavioralCharacteristic& lhs, const qdt::DecisionTreeBehavioralCharacteristic& rhs) {

    if (lhs.decision_frequencies.size() != rhs.decision_frequencies.size()) {
        return false;
    }
    for (auto frequency : lhs.decision_frequencies) {
        if (rhs.decision_frequencies.find(frequency.first) == rhs.decision_frequencies.end()) {
            return false;
        }
        if (rhs.decision_frequencies.at(frequency.first) != frequency.second) {
            return false;
        }
    }
    return true;
}

///
/// \brief Determines the label which would be assigned to minimize the error on the data
///
/// \param[in] data The data to consider
///
/// \return The label chosen
///

static float calculateLabelForData(const std::vector<const DataElem*>& data) {
    
    if (data.size() == 0) {
        return NOT_LABELED;
    }

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
        
    if (data.size() == 0) {
        return {{NOT_LABELED, 1}};
    }
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

uint32_t DecisionTree::getHeight() {
    uint32_t height = 0;
    std::shared_ptr<DecisionTreeNode> node = root_;
    while (nullptr != node->r_child) {
        node = node->r_child;
        height += 1;
    }
    return height;
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

std::shared_ptr<DecisionTree> DecisionTree::greedyTrain(const std::vector<const DataElem*>& training_data, float pruning_factor) {

    // Translate pruning factor into minimum number of samples required to reach each node in the tree
    uint32_t min_samples = (pruning_factor * 0.01) * training_data.size();
    min_samples = (min_samples == 0) ? 1 : min_samples;

    // Create the tree
    std::shared_ptr<DecisionTree> tree(new DecisionTree);
    tree->root_ = greedyTrain(training_data, nullptr, 0, min_samples);
    return tree;
}

std::shared_ptr<DecisionTreeNode> DecisionTree::greedyTrain(const std::vector<const DataElem*>& training_data, std::shared_ptr<DecisionTreeNode> parent, uint64_t node_number, uint32_t min_samples) {
    
    // Make a new decision tree node
    auto tree_node = std::make_shared<DecisionTreeNode>();
    tree_node->node_number = node_number;
    tree_node->parent = parent;
    tree_node->label = calculateLabelForData(training_data);
    tree_node->confidences = calculateConfidencesForData(training_data);

    // Find best split
    float best_error = std::numeric_limits<float>::max();
    auto best_split = std::make_shared<Decision>();
    std::shared_ptr<std::vector<const DataElem*>> best_left_group;
    std::shared_ptr<std::vector<const DataElem*>> best_right_group;
    for (auto feature : training_data[0]->features) {

        std::string feature_key = feature.first; 

        // Check threshold values if feature is real valued
        if (feature.second.second == true) {

            for (float split = 0.0; split < 1.0; split += 0.10) {
                auto left_group = std::make_shared<std::vector<const DataElem*>>();
                auto right_group = std::make_shared<std::vector<const DataElem*>>();
                for (auto elem : training_data) {
                    if (elem->features.at(feature_key).first < split) {
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
        else {

            for (auto possible_value : DataSet::discrete_features.at(feature_key)) {

                auto left_group = std::make_shared<std::vector<const DataElem*>>();
                auto right_group = std::make_shared<std::vector<const DataElem*>>();
                for (auto elem : training_data) {
                    if (elem->features.at(feature_key).first == possible_value) {
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
                    best_split->threshold = possible_value;
                    best_left_group = left_group;
                    best_right_group = right_group;
                }                
            }
        }
    }

    // If a best group is size 0, we're unseparable
    if (((best_left_group->size() < min_samples) || (best_right_group->size() < min_samples))) {
        return tree_node;
    }

    tree_node->decision = best_split;
    tree_node->l_child = greedyTrain(*best_left_group, tree_node, (2 * tree_node->node_number) + 1, min_samples);
    tree_node->r_child = greedyTrain(*best_right_group, tree_node, (2 * tree_node->node_number) + 2, min_samples);
    return tree_node;
}

std::shared_ptr<DecisionTreeNode> DecisionTree::generateRandomTree(uint32_t height, const std::vector<std::string>& features, std::shared_ptr<DecisionTreeNode> parent, uint64_t node_number, std::shared_ptr<const std::vector<const DataElem*>> training_data) {

    // Make a new decision tree node
    auto tree_node = std::make_shared<DecisionTreeNode>();
    tree_node->node_number = node_number;
    tree_node->parent = parent;

    // Create decision and subtree for non-leaves
    if (1 <= height) {
        tree_node->decision = std::make_shared<Decision>();
        tree_node->decision->feature = features[rand() % features.size()];

        if (DataSet::discrete_features.find(tree_node->decision->feature) == DataSet::discrete_features.end()) {
            
            if ((training_data == nullptr) || (training_data->size() == 0)) {
                tree_node->decision->threshold = 0.01 * (rand() % 100);
            }
            else {
                uint32_t idx1 = rand() % training_data->size();
                uint32_t idx2 = rand() % training_data->size();
                std::string feature = tree_node->decision->feature;
                float val1 = training_data->at(idx1)->features.at(feature).first;
                float val2 = training_data->at(idx2)->features.at(feature).first;
                tree_node->decision->threshold = (val1 + val2) / 2;
            }

        }
        else {
            if ((training_data == nullptr) || (training_data->size() == 0)) {
                auto& choices = DataSet::discrete_features[tree_node->decision->feature];
                auto iterator = choices.begin();
                std::advance(iterator, (rand() % choices.size()));
                tree_node->decision->threshold = *iterator;
            }
            else {
                uint32_t idx1 = rand() % training_data->size();
                uint32_t idx2 = rand() % training_data->size();
                std::string feature = tree_node->decision->feature;
                float val1 = training_data->at(idx1)->features.at(feature).first;
                float val2 = training_data->at(idx2)->features.at(feature).first;
                if ((rand() % 2) == 0) {
                    tree_node->decision->threshold = val1;
                }
                else {
                    tree_node->decision->threshold = val2;
                }
            }
        }
        
        // Check threshold values if feature is real valued
        if ((training_data != nullptr) && (training_data->size() != 0)) {
            if (DataSet::discrete_features.find(tree_node->decision->feature) == DataSet::discrete_features.end()) {

                float split = tree_node->decision->threshold;
                auto left_group = std::make_shared<std::vector<const DataElem*>>();
                auto right_group = std::make_shared<std::vector<const DataElem*>>();
                for (auto elem : *training_data) {
                    if (elem->features.at(tree_node->decision->feature).first < split) {
                        left_group->push_back(elem);
                    }
                    else {
                        right_group->push_back(elem);
                    }
                }

                // Fill child nodes
                tree_node->l_child = generateRandomTree(height - 1, features, tree_node, (2 * tree_node->node_number) + 1, left_group);
                tree_node->r_child = generateRandomTree(height - 1, features, tree_node, (2 * tree_node->node_number) + 2, right_group);   
            }
            else {

                float split = tree_node->decision->threshold;
                auto left_group = std::make_shared<std::vector<const DataElem*>>();
                auto right_group = std::make_shared<std::vector<const DataElem*>>();
                for (auto elem : *training_data) {
                    if (elem->features.at(tree_node->decision->feature).first == split) {
                        left_group->push_back(elem);
                    }
                    else {
                        right_group->push_back(elem);
                    }
                }

                // Fill child nodes
                tree_node->l_child = generateRandomTree(height - 1, features, tree_node, (2 * tree_node->node_number) + 1, left_group);
                tree_node->r_child = generateRandomTree(height - 1, features, tree_node, (2 * tree_node->node_number) + 2, right_group);   
            }

        }
        else {

            // Fill child nodes
            tree_node->l_child = generateRandomTree(height - 1, features, tree_node, (2 * tree_node->node_number) + 1);
            tree_node->r_child = generateRandomTree(height - 1, features, tree_node, (2 * tree_node->node_number) + 2);   
        }
    }
    return tree_node;

}
    
void DecisionTree::copy(const std::shared_ptr<DecisionTreeNode>& source, std::shared_ptr<DecisionTreeNode>& dest, std::shared_ptr<DecisionTreeNode> parent) {
    
    // base case
    if (source == nullptr) {
        return;
    }

    // Allocate new structs
    dest = std::make_shared<DecisionTreeNode>();
    dest->parent = parent;

    // Copy info
    if (nullptr != source->decision) {
        dest->decision = std::make_shared<Decision>();
        dest->decision->feature = source->decision->feature;
        dest->decision->threshold = source->decision->threshold;
    }

    dest->confidences = source->confidences;
    dest->label = source->label;
    dest->node_number = source->node_number;

    // Recurse
    copy(source->l_child, dest->l_child, dest);
    copy(source->r_child, dest->r_child, dest);
}

std::shared_ptr<DecisionTree> DecisionTree::copy() {
    
    // Make a new decision tree
    std::shared_ptr<DecisionTree> tree(new DecisionTree);
    copy(root_, tree->root_, nullptr);
    return tree;
}

void DecisionTree::fillLabels(const std::vector<const DataElem*>& training_data) {
    fillLabels(root_, training_data);
}

void DecisionTree::prune(const std::vector<const DataElem*>& training_data, float min_sample_representation) {
    uint32_t min_samples = (min_sample_representation * 0.01) * training_data.size();
    fillLabels(root_, training_data, min_samples);
}

void DecisionTree::fillLabels(std::shared_ptr<DecisionTreeNode> tree_node, const std::vector<const DataElem*>& training_data, uint32_t min_samples) {

    // label
    tree_node->label = calculateLabelForData(training_data);
    tree_node->confidences = calculateConfidencesForData(training_data);

    // Split data for non-leaf nodes
    if (nullptr != tree_node->decision) {
        std::vector<const DataElem*> left_group;
        std::vector<const DataElem*> right_group;
        for (auto elem : training_data) {

            if (elem->features.at(tree_node->decision->feature).second == true) {
                if (elem->features.at(tree_node->decision->feature).first < tree_node->decision->threshold) {
                    left_group.push_back(elem);
                }
                else {
                    right_group.push_back(elem);
                }
            }
            else {
                if (elem->features.at(tree_node->decision->feature).first == tree_node->decision->threshold) {
                    left_group.push_back(elem);
                }
                else {
                    right_group.push_back(elem);
                }                
            }
        }   

        // Prune if node has a small sample size
        if ((right_group.size() + left_group.size()) < min_samples) {
            tree_node->l_child = nullptr;
            tree_node->r_child = nullptr;
            tree_node->decision = nullptr;
        }       
        else {
            // Recurse
            fillLabels(tree_node->l_child, left_group, min_samples);
            fillLabels(tree_node->r_child, right_group, min_samples);
        }
    }
}

std::shared_ptr<DecisionTree> DecisionTree::randomTrain(const std::vector<const DataElem*>& training_data, uint32_t height, float pruning_factor) {
    std::shared_ptr<DecisionTree> tree(new DecisionTree);

    // Find feature labels
    std::vector<std::string> features;
    for (auto feature : training_data[0]->features) {
        features.push_back(feature.first);
    }

    // Randomly generate the tree
    auto training_ptr = std::make_shared<const std::vector<const DataElem*>>(training_data);
    tree->root_ = generateRandomTree(height, features, nullptr, 0, training_ptr);

    // Label the nodes according to data
    tree->prune(training_data, pruning_factor);
    return tree;
}

static void tournamentSelection(const std::vector<float>& fitness, uint32_t tournament_size, uint32_t& first_idx, uint32_t& second_idx) {
        
        // Select parents w/ tournament
        std::unordered_set<uint32_t> tournament_trees;
        while (tournament_trees.size() < tournament_size) {
            tournament_trees.insert(rand() % fitness.size());
        }

        float best_fitness = std::numeric_limits<float>::min();
        float second_fitness = std::numeric_limits<float>::min();
        for (auto idx : tournament_trees) {
            if (fitness[idx] > best_fitness) {
                second_idx = first_idx;
                second_fitness = best_fitness;
                first_idx = idx;
                best_fitness = fitness[idx];
            }
            else if (fitness[idx] > second_fitness) {
                second_idx = idx;
                second_fitness = fitness[idx];
            }
        }
}

std::shared_ptr<DecisionTreeNode> DecisionTree::getRandomNodeAtHeight(uint32_t height) {
    std::shared_ptr<DecisionTreeNode> ret_node = root_;
    for (uint32_t i = 0; i < height; i++) {
        if ((rand() % 2) == 0) {
            ret_node = ret_node->l_child;
        }
        else {
            ret_node = ret_node->r_child;
        }
    }
    return ret_node;
}

void DecisionTree::fixNodeNumbers(std::shared_ptr<DecisionTreeNode> node, std::shared_ptr<DecisionTreeNode> parent, uint32_t node_number) {

    if (nullptr == node) {
        return;
    }
    node->node_number = node_number;
    node->parent = parent;
    fixNodeNumbers(node->l_child, node, (2 * node->node_number) + 1);
    fixNodeNumbers(node->r_child, node, (2 * node->node_number) + 2);
}

void DecisionTree::repair(const std::vector<const DataElem*>& training_data) {

    fixNodeNumbers(root_, nullptr, 0);
    fillLabels(training_data);
}

static void mutateNode(std::shared_ptr<DecisionTreeNode> node, std::vector<std::string>& features, float decision_chance, std::default_random_engine& generator, std::normal_distribution<double> threshold_delta) {

    if ((rand() % 100) < decision_chance) {
        node->decision->feature = features[rand() % features.size()];
        if (DataSet::discrete_features.find(node->decision->feature) == DataSet::discrete_features.end()) {
            node->decision->threshold = 0.01 * (rand() % 100);
        }
        else {
            auto& choices = DataSet::discrete_features[node->decision->feature];
            auto iterator = choices.begin();
            std::advance(iterator, (rand() % choices.size()));
            node->decision->threshold = *iterator;
        }
    }
    else {
        if (DataSet::discrete_features.find(node->decision->feature) == DataSet::discrete_features.end()) {
            node->decision->threshold += threshold_delta(generator);
            /*
            if (node->decision->threshold < 0) {
                node->decision->threshold = 0;
            }
            if (node->decision->threshold > 1) {
                node->decision->threshold = 1;
            }
            */
        }
        else {
            if ((rand() % 100) < decision_chance) {
                auto& choices = DataSet::discrete_features[node->decision->feature];
                auto iterator = choices.begin();
                std::advance(iterator, (rand() % choices.size()));
                node->decision->threshold = *iterator;
            }
        }
    }

}


void DecisionTree::mutate(std::vector<std::string>& features) {

    float decision_chance = getHeight() * 0.5;
    //float decision_chance = 10;
    std::default_random_engine generator;
    std::normal_distribution<double> threshold_delta(0, 0.1);

    mutateNode(root_, features, decision_chance, generator, threshold_delta);
}

static void crossover(std::shared_ptr<DecisionTree> parent_one, std::shared_ptr<DecisionTree> parent_two, const std::vector<const DataElem*>& training_data, std::vector<std::shared_ptr<DecisionTree>>& offspring) {

    // Copy the parent trees
    auto offspring_one = parent_one->copy();
    auto offspring_two = parent_two->copy();

    // Swap two random non-root nodes from the same height
    const uint32_t HEIGHT = (rand() % (parent_one->getHeight())) + 1;
    auto swap_node_one = offspring_one->getRandomNodeAtHeight(HEIGHT);
    auto swap_node_two = offspring_two->getRandomNodeAtHeight(HEIGHT);

    if (swap_node_one->parent.lock()->l_child->node_number == swap_node_one->node_number) {
        swap_node_one->parent.lock()->l_child = swap_node_two;
    }
    else {
        swap_node_one->parent.lock()->r_child = swap_node_two;
    }

    if (swap_node_two->parent.lock()->l_child->node_number == swap_node_two->node_number) {
        swap_node_two->parent.lock()->l_child = swap_node_one;
    }
    else {
        swap_node_two->parent.lock()->r_child = swap_node_one;
    }

    // Modify the decisions
    std::vector<std::string> features;
    for (auto feature : training_data[0]->features) {
        features.push_back(feature.first);
    }
    offspring_one->mutate(features);
    offspring_two->mutate(features);

    // Relabel the trees and recalculate guesses
    offspring_one->repair(training_data);
    offspring_two->repair(training_data);
    offspring.push_back(offspring_one);
    offspring.push_back(offspring_two);

}

std::vector<std::shared_ptr<DecisionTree>> DecisionTree::geneticProgrammingTrain(const std::vector<const DataElem*>& training_data, uint32_t height, 
                                                                                 uint32_t population_size, uint32_t forest_size, uint32_t num_generations) {

    // Initialize population randomly
    std::vector<std::shared_ptr<DecisionTree>> population;
    std::vector<float> fitness;
    for (uint32_t i = 0; i < population_size; i++) {
        population.push_back(randomTrain(training_data, height, 0));
        fitness.push_back(population.back()->testAccuracy(training_data, false));
    }

    // Evaluate fitness

    // TODO EVOLVE
    const uint32_t TOURNAMENT_SIZE = population_size / 10;
    const uint32_t NUM_OFFSPRING = 10;
    std::cout << "evolution progress: 0%" << std::endl;
    for (uint32_t generation = 0; generation < num_generations; generation++) {

        std::cout << "\x1b[A" << "evolution progress: " << (((float)generation)/num_generations) * 100 << "%" << std::endl; 
        // Create offspring
        uint32_t first_idx = 0;
        uint32_t second_idx = 0;
        tournamentSelection(fitness, TOURNAMENT_SIZE, first_idx, second_idx);
        std::vector<std::shared_ptr<DecisionTree>> offspring;
        for (uint32_t i = 0; i < NUM_OFFSPRING; i += 2) {
            crossover(population[first_idx], population[second_idx], training_data, offspring);
        }

        // Replace
        uint32_t worst_idx = 0;
        float worst_fitness = std::numeric_limits<float>::max();
        for (uint32_t i = 0; i < fitness.size(); i++) {
            if (fitness[i] < worst_fitness) {
                worst_fitness = fitness[i];
                worst_idx = i;
            }
        }

        std::shared_ptr<DecisionTree> best_offspring = nullptr;
        float best_offspring_fitness = std::numeric_limits<float>::min();
        for (uint32_t i = 0; i < offspring.size(); i++) {
            float offspring_fitness = offspring[i]->testAccuracy(training_data, false);
            if (offspring_fitness > best_offspring_fitness) {
                best_offspring = offspring[i];
                best_offspring_fitness = offspring_fitness;
            }
        }
        //std::cout << "parent1 fitness = " << fitness[first_idx] << std::endl;
        //std::cout << "parent2 fitness = " << fitness[second_idx] << std::endl;
        //std::cout << "offspring fitness = " << best_offspring_fitness << std::endl;

        if (worst_fitness < best_offspring_fitness) {
            population[worst_idx] = best_offspring;
            fitness[worst_idx] = best_offspring_fitness;
        }
    }

    std::vector<std::shared_ptr<DecisionTree>> return_vector;
    if (forest_size == 1) {
        
        // Return best if forest size is 1
        uint32_t best_idx;
        float best_fitness = std::numeric_limits<float>::min();
        for (uint32_t i = 0; i < fitness.size(); i++) {
            if (fitness[i] > best_fitness) {
                best_idx = i;
                best_fitness = fitness[i];
            }
        }
        return_vector.push_back(population[best_idx]);
    }
    else {

        // Return random elements otherwise
        while (return_vector.size() < forest_size) {
            uint32_t idx = rand() % population.size();
            return_vector.push_back(population[idx]);
            population.erase(population.begin() + idx);
        }
    }
    return return_vector;
}

float DecisionTree::testAccuracy(const std::vector<const DataElem*>& testing_data, bool verbose, std::vector<float>* predictions) {


    uint32_t counter = 0;
    uint64_t num_correct = 0;
    for (auto elem : testing_data) {
        float predicted_val = predict(elem->features);
        if (nullptr != predictions) {
            (*predictions)[counter] = predicted_val;
        }
        if (elem->output == predicted_val) {
            num_correct++;
        }
        else {
            if (verbose) {
                std::cout << "mispredicted " << elem->output << " as " << predicted_val << std::endl; 
            }
        }
        counter++;
    }
    return (float)num_correct / testing_data.size();
}

float DecisionTree::testEnsembleAccuracy(const std::vector<std::shared_ptr<DecisionTree>>& ensemble, const std::vector<const DataElem*>& testing_data, bool verbose) {

    uint64_t num_correct = 0;
    for (auto elem : testing_data) {

        // Create votes
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

float DecisionTree::predict(const std::unordered_map<std::string, std::pair<float, bool>> features) {

    auto cur_node = getRoot();
    while (cur_node->decision != nullptr) {
        if (cur_node->label == NOT_LABELED) {
            return cur_node->parent.lock()->label;
        }
        if (cur_node->decision->test(features)) {
            cur_node = cur_node->l_child;
        }
        else {
            cur_node = cur_node->r_child;
        }
    }
    return cur_node->label;
}


std::unordered_map<float, float> DecisionTree::predictConfidence(const std::unordered_map<std::string, std::pair<float, bool>> features) {

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

std::shared_ptr<DecisionTreeBehavioralCharacteristic> DecisionTree::getBehavioralCharacteristic(uint32_t num_bins) {
    auto bc = std::make_shared<DecisionTreeBehavioralCharacteristic>();
    addBcContribution(root_, bc, num_bins, 1);
    return bc;
}

void DecisionTree::addBcContribution(const std::shared_ptr<DecisionTreeNode>& node, std::shared_ptr<DecisionTreeBehavioralCharacteristic> bc, uint32_t num_bins, float value) {
    if (node->decision == nullptr) {
        return;
    }

    // Round to nearest tenth. TODO: configure step size?
    Decision representative;
    representative.feature = node->decision->feature;
    representative.threshold = ((int)(node->decision->threshold * num_bins)) / (float)num_bins;

    // Add frequency contribution to representative decision
    float frequency = bc->decision_frequencies[representative];
    frequency = ((frequency + value) >= 1) ? 1 : frequency + value;
    bc->decision_frequencies[representative] = frequency;

    // Recurse
    addBcContribution(node->l_child, bc, num_bins, value/2);
    addBcContribution(node->r_child, bc, num_bins, value/2);
}

std::vector<std::shared_ptr<DecisionTree>> DecisionTree::QDTrain(const std::vector<const DataElem*>& training_data, uint32_t height, 
                                                                 uint32_t population_size, uint32_t forest_size, uint32_t num_generations, 
                                                                 uint32_t num_bc_bins, uint32_t min_distance_percentage, qdt::SelectionStrategy selection) {

    (void)population_size;
    (void)num_bc_bins;
    // Initialize an empty container of trees
    // TODO: other container types
    std::vector<std::shared_ptr<DecisionTree>> container;
    std::vector<float> container_accuracy;
    std::vector<float> selector;
    std::vector<std::vector<float>> container_predictions;
    constexpr uint32_t INIT_TRIALS = 3000;
    constexpr uint32_t MAX_INIT = 350;
    float selector_total = 0;
    float MIN_DISTANCE = min_distance_percentage * (training_data.size() / 100.0);

    // Generate random trees to initialize container with a population
    for (uint32_t trial = 0; trial < INIT_TRIALS; trial++) {

        if (container.size() > MAX_INIT) {
            //std::cout << "random is too easy " << std::endl;
            return std::vector<std::shared_ptr<DecisionTree>>();
        }

        if (container.size() < 2) {
            container.push_back(randomTrain(training_data, height, 0));
            container_predictions.push_back(std::vector<float>(training_data.size(), -1));
            //auto copy = container.back()->copy();
            //copy->prune(training_data, 1);
            container_accuracy.push_back(container.back()->testAccuracy(training_data, false, &(container_predictions[container_predictions.size() - 1])));
            selector.push_back(1);
            selector_total += 1;
        }
        else {
            auto new_predictions = std::vector<float>(training_data.size(), -1);
            auto new_tree = randomTrain(training_data, height, 0);
            float closest_distance = std::numeric_limits<float>::max();
            uint32_t closest_index = std::numeric_limits<uint32_t>::max();
            float second_closest_distance = std::numeric_limits<float>::max();
            //auto copy = new_tree->copy();
            //copy->prune(training_data, 1);
            float new_accuracy = new_tree->testAccuracy(training_data, false, &new_predictions);

            for (uint32_t i = 0; i < container.size(); i++) {
                float distance = 0;
                for (uint32_t j = 0; j < training_data.size(); j++) {
                    
                    // TODO: handle regression
                    bool equal = new_predictions[j] == container_predictions[i][j];
                    if (!equal) {
                       distance += 1;
                    }
                }         

                if (distance < closest_distance) {
                    second_closest_distance = closest_distance;
                    closest_distance = distance;
                    closest_index = i;
                }
                else if(distance < second_closest_distance) {
                    second_closest_distance = distance;
                }
            }

            // Generate a new tree and find its nearest two neighbors
            if (closest_distance > MIN_DISTANCE) {
                container.push_back(randomTrain(training_data, height, 0));
                container_predictions.push_back(new_predictions);
                container_accuracy.push_back(new_accuracy);
                selector.push_back(1);
                selector_total += 1;                
            }
            else if (second_closest_distance > MIN_DISTANCE) {

                // Replace existing
                if (new_accuracy > container_accuracy[closest_index]) {
                    container[closest_index] = new_tree;
                    container_accuracy[closest_index] = new_accuracy;
                    container_predictions[closest_index] = new_predictions;
                }
            }

        }
    }   

    // Repeat for generations
    //std::cout << "evolution progress: 0%" << std::endl << std::endl;
    uint32_t init_size = container.size();
    uint32_t num_replacements = 0;
    for (uint32_t i = 0; i < num_generations; i++) {

        if(selector_total > (container.size() * 2)) {
            //std::cout << "select is to high " << std::endl;
            return std::vector<std::shared_ptr<DecisionTree>>();
        }
        /*
        if (container.size() > 450) {
            break;
        }
        */
        if (i % 10 == 1) {
         //   std::cout << "\x1b[A\x1b[A" << "evolution progress: " << (((float)i)/num_generations) * 100 << "%" << std::endl; 
         //   std::cout << "container size is " << container.size() << " selector total is " << selector_total << std::endl;
        }
        
        float rand_selector = (rand() % (uint32_t)(2 * selector_total)) / 2.0;
        float selector_val = selector[0];
        uint32_t selector_index = 0;
        while (selector_val < rand_selector) {
            selector_index++;
            selector_val += selector[selector_index];
        }
        uint32_t parent1_index = selector_index;
        std::shared_ptr<DecisionTree> parent1 = container[parent1_index];
           
        rand_selector = (rand() % (uint32_t)(2 * selector_total)) / 2.0;
        selector_index = 0;
        selector_val = selector[0];
        while (selector_val < rand_selector) {
            selector_index++;
            selector_val += selector[selector_index];
        }
        uint32_t parent2_index = selector_index;
        std::shared_ptr<DecisionTree> parent2 = container[parent2_index];

        // Generate children
        constexpr uint32_t NUM_OFFSPRING = 2;
        std::vector<std::shared_ptr<DecisionTree>> offspring;

        
        
        for (uint32_t j = 0; j < NUM_OFFSPRING; j += 2) {
            crossover(parent1, parent2, training_data, offspring);
        }
        
 /*
        for (uint32_t j = 0; j < NUM_OFFSPRING; j += 2) {
            offspring.push_back(randomTrain(training_data, height, 1));
            offspring.push_back(randomTrain(training_data, height, 1));
        }
*/
        // Add children to archive if they're more fit than current representative
        for (auto child : offspring) {

            auto new_predictions = std::vector<float>(training_data.size(), -1);
            //auto copy = child->copy();
            //copy->prune(training_data, 1);
            float new_accuracy = child->testAccuracy(training_data, false, &new_predictions);
            float closest_distance = std::numeric_limits<float>::max();
            uint32_t closest_index = std::numeric_limits<uint32_t>::max();
            float second_closest_distance = std::numeric_limits<float>::max();

            for (uint32_t j = 0; j < container.size(); j++) {
                float distance = 0;
                for (uint32_t k = 0; k < training_data.size(); k++) {
                    
                    
                    // TODO: handle regression
                    bool equal = new_predictions[k] == container_predictions[j][k];
                    if (!equal) {
                       distance += 1;
                    }
                }          

                if (distance < closest_distance) {
                    second_closest_distance = closest_distance;
                    closest_distance = distance;
                    closest_index = j;
                }
                else if(distance < second_closest_distance) {
                    second_closest_distance = distance;
                }
            }

            float accuracy_difference = new_accuracy - container_accuracy[closest_index];
            //float new_extra_room = MIN_DISTANCE - second_closest_distance;
            //float old_extra_room = MIN_DISTANCE - closests_closest_distance;
            //float extra_room_difference = new_extra_room - old_extra_room;
            // Generate a new tree and find its nearest two neighbors
            if (closest_distance > MIN_DISTANCE) {
                container.push_back(randomTrain(training_data, height, 0));
                container_accuracy.push_back(new_accuracy);
                container_predictions.push_back(new_predictions);
                selector.push_back(1);
                selector[parent1_index] += 1;
                selector[parent2_index] += 1;
                selector_total += 3;   
                //i = 0;
            }
            else if (((second_closest_distance > MIN_DISTANCE) && (accuracy_difference > 0)) /*|| 
                    ((new_accuracy >= (0.90 * container_accuracy[closest_index])) &&
                    (new_extra_room >= (0.10 * old_extra_room)) &&
                    ((extra_room_difference * container_accuracy[closest_index]) > -(accuracy_difference * old_extra_room)))*/) {
                    
                    // Replace existing
                    container[closest_index] = child;
                    container_accuracy[closest_index] = new_accuracy;
                    container_predictions[closest_index] = new_predictions;
                    float old_selector_val = selector[closest_index];
                    selector[closest_index] = 1;
                    selector[parent1_index] += 1;
                    selector[parent2_index] += 1;         
                    selector_total += (2 + (1 - old_selector_val));

                    if (closest_index < init_size) {
                        num_replacements++;
                    }
                    //std::cout << "hello" << std::endl;

            }
            // Otherwise, decrement parent selector scores
            else {

                if (selector[parent1_index] > 1) {
                    selector[parent1_index] -= 0.5;
                    selector_total -= 0.5;
                }

                if (selector[parent2_index] > 1) {
                    selector[parent2_index] -= 0.5;
                    selector_total -= 0.5;
                }
            }
        }
    }

    // Return an ensemble
    std::vector<std::shared_ptr<DecisionTree>> return_vector;
    if (forest_size > container.size()) {
        throw std::runtime_error("couldn't fill forest");
    }

    uint32_t num_greedy = 0;
    uint32_t num_random = 0;
    switch(selection) {
        case SelectionStrategy::ACCURATE:
            num_greedy = forest_size;
            break;
        case SelectionStrategy::DIVERSE:
            num_random = forest_size;
            break;
        case SelectionStrategy::HYBRID:
            num_greedy = forest_size / 2;
            num_random = forest_size - num_greedy;
            break;
    }

    std::vector<bool> taken(container.size(), false);
    while (return_vector.size() < num_random) {

        
        uint32_t new_idx = rand() % container.size();
        while (taken[new_idx] == true) {
            new_idx = (new_idx + 1) % container.size();
        }
        return_vector.push_back(container[new_idx]);
        taken[new_idx] = true;

    }
    while (return_vector.size() < num_random + num_greedy) {
                
       float best_fitness = std::numeric_limits<float>::min();
       uint32_t best_idx = 0;
       for (uint32_t i = 0; i < container.size(); i++) {
           if (taken[i] == false) {
               float fitness = container[i]->testAccuracy(training_data, false, nullptr);
               if (fitness > best_fitness) {
                   best_fitness = fitness;
                   best_idx = i;
               }
           }
       }
       taken[best_idx] = true;
       return_vector.push_back(container[best_idx]);

    }
    return return_vector;
}

}