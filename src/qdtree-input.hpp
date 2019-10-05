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

} // namespace qdt

#endif // __QDTREE_INPUT_HPP__