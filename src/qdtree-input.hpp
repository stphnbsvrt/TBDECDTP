#ifndef __QDTREE_INPUT_HPP__
#define __QDTREE_INPUT_HPP__

#include "qdtree-const.hpp"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace qdt {

///
/// \brief Parse commandline arguments 
///
/// \return boost variables map containing commandline arguments
///

po::variables_map parseCommandline(int argc, char** argv);

///
/// \brief Parse a json pandas dump
///
/// \return Vector of data elements
///

std::vector<DataElem> parseJson(std::string json_file);

///
/// \brief Create data sets to use for testing
///
/// \param[in] data The data to use for data set creation
/// \param[in] args Commandline args to use for test creation
///
/// \return A list of training/testing data pairs to compare between algorithms
///

std::vector<DataSet> createDataSets(const std::vector<DataElem>& data, po::variables_map args);

///
/// \brief Default number of trees to use in an ensemble
///

constexpr uint32_t DEFAULT_FOREST_SIZE = 5;

///
/// \brief Default height of trees generated by random methods
///

constexpr uint32_t DEFAULT_TREE_HEIGHT = 5;

///
/// \brief Default population size for population-based algorithms
///

constexpr uint32_t DEFAULT_POPULATION_SIZE = 50;

///
/// \brief Default pruning factor. Specifies a percentage of the training sample size that every subtree must be trained on.
///

constexpr float DEFAULT_PRUNING_FACTOR = 1.0;

///
/// \brief Default number of generations for genetic algorithms
///

constexpr uint32_t DEFAULT_NUM_GENERATIONS = 100;

///
/// \brief Default number of equivalence classes to use for thresholds when creating behavioral characterizations
///

constexpr uint32_t DEFAULT_BC_BINS = 10;

///
/// \brief Default number of folds to use for cross validation
///

constexpr uint32_t DEFAULT_NUM_FOLDS = 10;

///
/// \brief Default percentage of training data with different guesses for archive
///

constexpr uint32_t DEFAULT_MIN_DISTANCE_PERCENTAGE = 1;

///
/// \brief Default selection strategy for QD algorithm
///

constexpr SelectionStrategy DEFAULT_SELECTION_STRATEGY = SelectionStrategy::HYBRID;

} // namespace qdt

#endif // __QDTREE_INPUT_HPP__