#ifndef __QDTREE_CONST_HPP__
#define __QDTREE_CONST_HPP__

#include <unordered_map>
#include <unordered_set>
#include <map>
#include <sstream>
#include <memory>
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

namespace qdt {

/// 
/// \brief Represents a data point
///

struct DataElem {

    /// 
    /// \brief Constructor for a data element with an id
    ///

    DataElem(uint64_t id) : id(id) {}
    uint64_t id;

    ///
    /// \brief Output (i.e. of classification or regression) of the data point
    ///

    float output;

    ///
    /// \brief Collection of string->float features for the data point
    /// Bool indicates whether the feature is real valued.
    ///

    std::unordered_map<std::string, std::pair<float, bool>> features;


    ///
    /// \brief Returns string representation of the element
    ///

    std::string toStr() {
        std::ostringstream oss;
        oss << "Output: " << output << std::endl;
        for (auto feature : features) {
            oss << "  " << feature.first << ":" << feature.second.first << std::endl;
        }
        return oss.str();
    }
};

///
/// \brief A set of training/testing data set pairs to use for testing
///

struct DataSet {

    ///
    /// \brief Training data for the set
    ///

    std::vector<const DataElem*> training_data;

    ///
    /// \brief Testing data for the set
    ///

    std::vector<const DataElem*> testing_data;

    ///
    /// \brief Map of discrete-valued features to their possible values
    /// Note we can only use one data set at a time with this implementation
    ///

    static std::unordered_map<std::string, std::unordered_set<float>> discrete_features;

};

///
/// \brief Represents a decision for a node to consider for determining branch routing
///

struct Decision {

    ///
    /// \brief Feature component of the decision
    ///

    std::string feature;

    ///
    /// \brief Split threshold for a real-valued feature decision
    /// If feature is a classification, represents the exact feature value which indicates a left branch
    ///

    float threshold;

    ///
    /// \brief Tests a feature map on the decision for this tree
    ///
    /// \return True if the feature is less than the threshold, indicating left branch. 
    /// Otherwise false, indicating right branch.
    ///

    bool test(const std::unordered_map<std::string, std::pair<float, bool>>& test_features) {
        if (test_features.at(feature).second == true) {
            return test_features.at(feature).first < threshold;
        }
        else {
            return (int)test_features.at(feature).first == (int)threshold;
        }
    }
};

///
/// \brief Define equality operator for decisions so we can use them as map keys
///
bool operator==(const qdt::Decision& lhs, const qdt::Decision& rhs);

} // namespace qdt

///
/// \brief Define hash function for decisions so we can use them as map keys
///

template <>
struct std::hash<qdt::Decision>
{
std::size_t operator()(const qdt::Decision& k) const
{
    using std::size_t;
    using std::hash;
    using std::string;

    // Compute individual hash values for first,
    // second and third and combine them using XOR
    // and bit shifting:

    return ((hash<string>()(k.feature)
            ^ (hash<float>()(k.threshold) << 1)) >> 1);
}
};


///
/// \brief Describes the structural quality of a decision tree in order to compare similarity
///

namespace qdt {
struct DecisionTreeBehavioralCharacteristic {

    ///
    /// \brief Map of possible decision categories to their frequency of appearance in a given tree
    /// Each appearance of a decision in a tree increases the "frequency" measure by an amount weighted by its height
    /// For instance, the decision appearing at the root of a tree will be valued at 1 while the decision at the 
    /// child of the root will be valued at .5. The frequency value is capped at 1.
    ///

    std::unordered_map<Decision, float> decision_frequencies;

    ///
    /// \brief Dumps the contents of the behavioral characteristic as a string
    ///

    std::string toStr() {
        std::ostringstream oss;
        for (auto entry : decision_frequencies) {
            oss << "(" << entry.first.feature << " < " << entry.first.threshold << ")? - " << entry.second << std::endl;
        }
        return oss.str();
    }

    ///
    /// \brief Calculates euclidean distance between this BC and another BC
    ///

    float compare(std::shared_ptr<DecisionTreeBehavioralCharacteristic> other) {
        float sum = 0;

        // Add all frequencies present in this BC
        for (auto entry : decision_frequencies) {
            auto other_it = other->decision_frequencies.find(entry.first);
            float other_val = (other_it == other->decision_frequencies.end()) ? 0 : other->decision_frequencies.at(entry.first);
            float difference = entry.second - other_val;
            sum += difference * difference;
        }

        // Add all frequencies present in other BC not present in this BC
        for (auto entry : other->decision_frequencies) {
            if (decision_frequencies.find(entry.first) == decision_frequencies.end()) {
                sum += entry.second * entry.second;
            }
        }

        // Return square root of sum of squares
        return std::sqrt(sum);
    }
};

///
/// \brief Define equality operator for decisions so we can use them as map keys
///
bool operator==(const qdt::DecisionTreeBehavioralCharacteristic& lhs, const qdt::DecisionTreeBehavioralCharacteristic& rhs);

} // namespace qdt

///
/// \brief Define hash function for decisions so we can use them as map keys
///

template <>
struct std::hash<qdt::DecisionTreeBehavioralCharacteristic>
{
std::size_t operator()(const qdt::DecisionTreeBehavioralCharacteristic& k) const
{
    using std::size_t;
    using std::hash;
    using std::string;

    // Compute individual hash values for first,
    // second and third and combine them using XOR
    size_t output = 0;
    for (auto frequency : k.decision_frequencies) {
        output ^= hash<qdt::Decision>()(frequency.first);
    }
    return output;
}
};

namespace qdt {
///
/// \brief Represents a node in a decision tree
///

constexpr float NOT_LABELED = std::numeric_limits<float>::min();
struct DecisionTreeNode {

    ///
    /// \brief Represents the node number in a decision tree
    ///

    uint64_t node_number;

    ///
    /// \brief The label guess for this node
    ///

    float label = NOT_LABELED;

    ///
    /// \brief The confidence of each label for this node
    ///

    std::unordered_map<float, float> confidences;

    ///
    /// \brief Pointer to the parent of this node - nullptr if it is a root
    ///

    std::weak_ptr<DecisionTreeNode> parent;

    ///
    /// \brief Pointer to the left child of this node - nullptr if leaf
    ///

    std::shared_ptr<DecisionTreeNode> l_child;

    ///
    /// \brief Pointer to the right child of this node - nullptr if leaf
    ///

    std::shared_ptr<DecisionTreeNode> r_child;

    ///
    /// \brief Decision used by this node to determine routing - nullptr if leaf
    ///

    std::shared_ptr<Decision> decision;
};

///
/// \brief Wrapper for a root decision tree node which can perform generic operations
///

class DecisionTree {

public:

    ///
    /// \brief Return the root node of the tree
    ///

    std::shared_ptr<DecisionTreeNode> getRoot() {
        return root_;
    } 

    ///
    /// \brief Return the height of the tree
    ///

    uint32_t getHeight();

    ///
    /// \brief Get a node at the given height reached by random traversal
    ///

    std::shared_ptr<DecisionTreeNode> getRandomNodeAtHeight(uint32_t height);

    ///
    /// \brief Return a node-for-node copy of the tree
    ///

    std::shared_ptr<DecisionTree> copy();

    ///
    /// \brief Get a behavioral characteristic descriptor for the tree using a specified number of equivalence classes for thresholds
    ///

    std::shared_ptr<DecisionTreeBehavioralCharacteristic> getBehavioralCharacteristic(uint32_t num_bins);

    ///
    /// \brief Generate a tree from data elements using a greedy heuristic strategy
    ///

    static std::shared_ptr<DecisionTree> greedyTrain(const std::vector<const DataElem*>& training_data, float pruning_factor);

    ///
    /// \brief Generate a tree randomly and create predictions based on the input data
    ///
    /// \param[in] height Height of the complete tree to generate
    /// \param[in] pruning_factor Intensity of pruning. Any node which isn't trained on this percentage of the original sample size will be pruned.
    ///

    static std::shared_ptr<DecisionTree> randomTrain(const std::vector<const DataElem*>& training_data, uint32_t height, float pruning_factor);

    ///
    /// \brief Generate a tree using a genetic programming algorithm
    ///
    
    static std::vector<std::shared_ptr<DecisionTree>> geneticProgrammingTrain(const std::vector<const DataElem*>& training_data, uint32_t height, uint32_t population_size, 
                                                                              uint32_t forest_size, uint32_t num_generations);

    ///
    /// \brief Generate an ensemble of trees using a QD algorithm
    ///

    static std::vector<std::shared_ptr<DecisionTree>> QDTrain(const std::vector<const DataElem*>& training_data, uint32_t height, uint32_t population_size, uint32_t forest_size, 
                                                              uint32_t num_generations, uint32_t num_bc_bins, uint32_t min_distance_percentage);

    ///
    /// \brief Test the accuracy of the tree on a set of data
    /// Verbose option for logging mispredictions
    ///
    /// \return Percent accuracy of the predictions over the data set
    ///

    float testAccuracy(const std::vector<const DataElem*>& testing_data, bool verbose, std::vector<float>* predictions=nullptr);

    ///
    /// \brief Test the accuracy of an ensemble of trees on a set of data
    /// Verbose option for logging mispredictions
    ///
    /// \return Percent accuracy of the predictions over the data set
    ///

    static float testEnsembleAccuracy(const std::vector<std::shared_ptr<DecisionTree>>& ensemble, const std::vector<const DataElem*>& testing_data, bool verbose);

    ///
    /// \brief Predict a label for a set of features using the decision tree
    ///

    float predict(const std::unordered_map<std::string, std::pair<float, bool>> features);

    ///
    /// \brief Predict with confidence for a set of features using the decision tree
    ///

    std::unordered_map<float, float> predictConfidence(const std::unordered_map<std::string, std::pair<float, bool>> features);

    ///
    /// \brief Repair node numbers and training labels in the tree
    ///

    void repair(const std::vector<const DataElem*>& training_data);

    ///
    /// \brief Mutate the decisions in the tree
    ///

    void mutate(std::vector<std::string>& features);

    ///
    /// \brief Fills the labels of the tree and prunes any branches reachable by a percentage training samples smaller than the specified amount
    ///

    void prune(const std::vector<const DataElem*>& training_data, float min_sample_representation);

protected:

    ///
    /// \brief Used to recursively copy one decision tree to another
    ///

    static void copy(const std::shared_ptr<DecisionTreeNode>& source, std::shared_ptr<DecisionTreeNode>& dest, std::shared_ptr<DecisionTreeNode> parent);

    ///
    /// \brief Used to recursively fill a behavioral characteristic
    ///

    static void addBcContribution(const std::shared_ptr<DecisionTreeNode>& node, std::shared_ptr<DecisionTreeBehavioralCharacteristic> bc, uint32_t num_bins, float value);

    ///
    /// \brief Used to recursively apply greedy decision tree node generation
    ///

    static std::shared_ptr<DecisionTreeNode> greedyTrain(const std::vector<const DataElem*>& training_data, std::shared_ptr<DecisionTreeNode> parent, uint64_t node_number, uint32_t min_samples);

    ///
    /// \brief Generates a complete binary tree of a given height using random decisions at each node
    ///

    static std::shared_ptr<DecisionTreeNode> generateRandomTree(uint32_t height, const std::vector<std::string>& features, std::shared_ptr<DecisionTreeNode> parent, uint64_t node_number);

    ///
    /// \brief Fill the labels of the nodes in the tree according to the given data
    ///

    void fillLabels(const std::vector<const DataElem*>& training_data);

    ///
    /// \brief Recursively fill labels of decision tree nodes according to the given data
    /// Will prune children nodes which are reached with a number of samples less than the specified minimum. I.e. 0 does no pruning.
    ///

    static void fillLabels(std::shared_ptr<DecisionTreeNode> tree_node, const std::vector<const DataElem*>& training_data, uint32_t min_samples=0);

    ///
    /// \brief Fix node numbers and parent pointers in a subtree
    ///

    static void fixNodeNumbers(std::shared_ptr<DecisionTreeNode> node, std::shared_ptr<DecisionTreeNode> parent, uint32_t node_number);

    ///
    /// \brief Default constructor
    ///

    DecisionTree() {};

    ///
    /// \brief Root of the tree
    ///
    
    std::shared_ptr<DecisionTreeNode> root_;
};

} // namespace qdt

#endif // __QDTREE_CONST_HPP__