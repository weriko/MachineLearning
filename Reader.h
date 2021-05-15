#include <unordered_map>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>



using std::string;
using std::vector;
using std::unordered_map;
//template <class T>
std::vector<std::string> split(std::string text, char delim);
std::unordered_map <std::string, std::vector<double>> read_csv_dataframe(std::string);
std::unordered_map <std::string, std::vector<vector<double>> > to_x_y(vector <vector<double>>);
std::vector<std::vector<double>> read_csv(std::string);
