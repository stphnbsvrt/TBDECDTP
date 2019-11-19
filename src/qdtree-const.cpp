#include "qdtree-const.hpp"
#include <iostream>
#include <limits>
#include <random>
#include <unordered_set>

namespace qdt {

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
    if ((best_left_group->size() < min_samples) || (best_right_group->size() < min_samples)) {
        return tree_node;
    }

    tree_node->decision = best_split;
    tree_node->l_child = greedyTrain(*best_left_group, tree_node, (2 * tree_node->node_number) + 1, min_samples);
    tree_node->r_child = greedyTrain(*best_right_group, tree_node, (2 * tree_node->node_number) + 2, min_samples);
    return tree_node;
}

std::shared_ptr<DecisionTreeNode> DecisionTree::generateRandomTree(uint32_t height, const std::vector<std::string>& features, std::shared_ptr<DecisionTreeNode> parent, uint64_t node_number) {

    // Make a new decision tree node
    auto tree_node = std::make_shared<DecisionTreeNode>();
    tree_node->node_number = node_number;
    tree_node->parent = parent;

    // Create decision and subtree for non-leaves
    if (1 <= height) {
        tree_node->decision = std::make_shared<Decision>();
        tree_node->decision->feature = features[rand() % features.size()];
        tree_node->decision->threshold = 0.01 * (rand() % 100);

        // Fill child nodes
        tree_node->l_child = generateRandomTree(height - 1, features, tree_node, (2 * tree_node->node_number) + 1);
        tree_node->r_child = generateRandomTree(height - 1, features, tree_node, (2 * tree_node->node_number) + 2);   
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
            if (elem->features.at(tree_node->decision->feature) < tree_node->decision->threshold) {
                left_group.push_back(elem);
            }
            else {
                right_group.push_back(elem);
            }
        }   

        // Prune if one of the children falls below the minimum sample size
        if ((right_group.size() < min_samples) || (left_group.size() < min_samples)) {
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
    tree->root_ = generateRandomTree(height, features, nullptr, 0);

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
    }
    node->decision->threshold += threshold_delta(generator);
}


void DecisionTree::mutate(std::vector<std::string>& features) {

    float decision_chance = getHeight() * 0.5;
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
    for (uint32_t generation = 0; generation < num_generations; generation++) {

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
                                                                 uint32_t population_size, uint32_t forest_size, uint32_t num_generations, uint32_t num_bc_bins) {

    // Initialize an empty container of trees
    // TODO: other container types
    std::unordered_map<DecisionTreeBehavioralCharacteristic, std::shared_ptr<DecisionTree>> container;

    // Generate random trees to initialize container with a population
    while (container.size() < population_size) {
        auto tree = randomTrain(training_data, height, 0);
        auto grid_location = tree->getBehavioralCharacteristic(num_bc_bins);
        if (container.find(*grid_location) == container.end()) {
            container.insert({*grid_location, tree});
        }
    }   

    // Repeat for generations
    for (uint32_t i = 0; i < num_generations; i++) {
        // Select parents randomly from container
        // TODO: other selection types
        size_t bucket_num, bucket_offset;
        do {
            bucket_num = rand() % container.bucket_count();
        } 
        while ((bucket_offset = container.bucket_size(bucket_num)) == 0);
        bucket_offset = rand() % bucket_offset;
        std::shared_ptr<DecisionTree> parent1 = std::next(container.begin(bucket_num), bucket_offset)->second;
        
        do {
            bucket_num = rand() % container.bucket_count();
        } 
        while ((bucket_offset = container.bucket_size(bucket_num)) == 0);
        bucket_offset = rand() % bucket_offset;
        std::shared_ptr<DecisionTree> parent2 = std::next(container.begin(bucket_num), bucket_offset)->second;

        // Generate children
        constexpr uint32_t NUM_OFFSPRING = 10;
        std::vector<std::shared_ptr<DecisionTree>> offspring;
        for (uint32_t i = 0; i < NUM_OFFSPRING; i += 2) {
            crossover(parent1, parent2, training_data, offspring);
        }

        // Add children to archive if they're more fit than current representative
        for (auto child : offspring) {
            auto grid_location = child->getBehavioralCharacteristic(num_bc_bins);

            // Add if there's no representative
            if (container.find(*grid_location) == container.end()) {
                container.insert({*grid_location, child});
            }        

            // Or if better than current
            if (container.at(*grid_location)->testAccuracy(training_data, false) < child->testAccuracy(training_data, false)) {
                container[*grid_location] = child;
            }
        }
    }

    // Return a forest
    // TODO: do smartly?
    std::vector<std::shared_ptr<DecisionTree>> return_vector;
    while (return_vector.size() < forest_size) {

        // Pull random tree from archive
        size_t bucket_num, bucket_offset;
        do {
            bucket_num = rand() % container.bucket_count();
        } 
        while ((bucket_offset = container.bucket_size(bucket_num)) == 0);
        bucket_offset = rand() % bucket_offset;
        auto it = std::next(container.begin(bucket_num), bucket_offset);
        return_vector.push_back(it->second);
        container.erase(container.find(it->first));
    }
    return return_vector;
}

}