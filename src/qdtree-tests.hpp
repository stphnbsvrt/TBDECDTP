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

void testGreedyHeuristic(const std::vector<DataElem>& data, po::variables_map args);
}

#endif // __QDTREE_TESTS_HPP__