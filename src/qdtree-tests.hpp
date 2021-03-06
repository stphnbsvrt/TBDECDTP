#ifndef __QDTREE_TESTS_HPP__
#define __QDTREE_TESTS_HPP__

#include "qdtree-const.hpp"
#include <boost/program_options.hpp>
#include <vector>

namespace po = boost::program_options;

namespace qdt {

///
/// \brief Test the accuracy of the greedy heuristic tree with provided args on the data
///

void testGreedyHeuristic(const std::vector<DataSet>& data, po::variables_map args);

///
/// \brief Test the accuracy of the bagging-constructed forest with provided args on the data
///
///
/// \return Average accuracy, diversity pair
///

std::pair<float, float> testBaggingEnsemble(const std::vector<DataSet>& data, po::variables_map args);

///
/// \brief Test the accuracy of a single complete-random tree with provided args on the data
///

void testCompleteRandomSingle(const std::vector<DataSet>& data, po::variables_map args);

///
/// \brief Test the accuracy of an ensemble of complete-random trees with provided args on the data
///
///
/// \return Average accuracy, diversity pair
///

std::pair<float, float> testCompleteRandomEnsemble(const std::vector<DataSet>& data, po::variables_map args);

///
/// \brief Test the accuracy of a single tree genetic programming tree with provided args on the data
///

void testGeneticSingle(const std::vector<DataSet>& data, po::variables_map args);

///
/// \brief Test the accuracy of an ensemble of genetic programming trees without considering diversity
///

void testGeneticEnsemble(const std::vector<DataSet>& data, po::variables_map args);

///
/// \brief Test the accuracy of a decision tree formulated from a QD algorithm
///
/// \return Average accuracy, diversity pair
///

std::pair<float, float> testQD(const std::vector<DataSet>& data, po::variables_map args);

}

#endif // __QDTREE_TESTS_HPP__