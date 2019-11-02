#ifndef __QDTREE_CONST_HPP__
#define __QDTREE_CONST_HPP__

#include <unordered_map>
#include <sstream>
#include <memory>
#include <vector>

namespace qdt {

/// 
/// \brief Represents a data point
///

struct DataElem {

    /// 
    /// \brief Constructor for a data element with an id
    ///

    DataElem(uint64_t id) : id(id) {}
    const uint64_t id;

    ///
    /// \brief Output (i.e. of classification or regression) of the data point
    ///

    float output;

    ///
    /// \brief Collection of string->float features for the data point
    ///

    std::unordered_map<std::string, float> features;

    ///
    /// \brief Returns string representation of the element
    ///

    std::string toStr() {
        std::ostringstream oss;
        oss << "Output: " << output << std::endl;
        for (auto feature : features) {
            oss << "  " << feature.first << ":" << feature.second << std::endl;
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
    /// \brief Split threshold for the decision
    ///

    float threshold;

    ///
    /// \brief Tests a feature map on the decision for this tree
    ///
    /// \return True if the feature is less than the threshold, indicating left branch. 
    /// Otherwise false, indicating right branch.
    ///

    bool test(const std::unordered_map<std::string, float>& test_features) {
        return test_features.at(feature) < threshold;
    }
};

///
/// \brief Represents a node in a decision tree
///

struct DecisionTreeNode {

    ///
    /// \brief Represents the node number in a decision tree
    ///

    uint64_t node_number;

    ///
    /// \brief The label guess for this node
    ///

    float label;

    ///
    /// \brief The confidence of each label for this node
    ///

    std::unordered_map<float, float> confidences;

    ///
    /// \brief Pointer to the parent of this node - nullptr if it is a root
    ///

    std::shared_ptr<DecisionTreeNode> parent;

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
    /// \brief Generate a tree from data elements using a greedy heuristic strategy
    ///

    static std::shared_ptr<DecisionTree> greedyTrain(const std::vector<const DataElem*>& training_data);

    ///
    /// \brief Test the accuracy of the tree on a set of data
    /// Verbose option for logging mispredictions
    ///
    /// \return Percent accuracy of the predictions over the data set
    ///

    float testAccuracy(const std::vector<const DataElem*>& testing_data, bool verbose);

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

    float predict(const std::unordered_map<std::string, float> features);

    ///
    /// \brief Predict with confidence for a set of features using the decision tree
    ///

    std::unordered_map<float, float> predictConfidence(const std::unordered_map<std::string, float> features);

protected:

    ///
    /// \brief Used to recursively apply greedy decision tree node generation
    ///

    static std::shared_ptr<DecisionTreeNode> greedyTrain(const std::vector<const DataElem*>& training_data, std::shared_ptr<DecisionTreeNode> parent, uint64_t node_number);

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