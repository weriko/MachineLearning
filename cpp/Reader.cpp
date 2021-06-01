#include "modules.h"
using std::string;
using std::vector;
using std::unordered_map;
//template <class T>


std::vector<std::string> split(std::string text, char delim) {
	std::string line;
	std::vector<std::string> vec;
	std::stringstream ss(text);
	while (std::getline(ss, line, delim)) {
		vec.push_back(line);
	}
	return vec;
}

std::vector<std::vector<double>> read_csv(std::string filename) {
	std::ifstream t(filename);
	string code((std::istreambuf_iterator<char>(t)),
		std::istreambuf_iterator<char>());

	vector <string> rows = split(code, '\n');
	vector <string> columns = split(rows[0], ',');
	vector <vector<double>> data;

	for (int i = 1;i < rows.size();i++) {
		vector <string> temp = split(rows[i], ',');
		vector <double> temp2;
		for (auto& p : temp) {
			temp2.push_back(std::stod(p));
		}

		data.push_back(temp2);
	}

	return data;

}

std::unordered_map <std::string, std::vector<vector<double>> > to_x_y(vector <vector<double>> data) {
	std::unordered_map <std::string, std::vector<vector<double>>> df;
	vector<vector<double>> x;
	vector<vector<double>> y;
	df["x"] = x;
	df["y"] = y;

	for (int i = 0; i < data.size() ;i++) {
		vector<double> temp(data[0].size()-1);
		for (int j = 0;j < data[0].size()-1;j++) {
			//std::cout << data[i][j] << " ";
			temp[j] = data[i][j];

		}
		df["x"].push_back(temp);
		//std::cout << "\n";
	}
	for (int i = 0; i < data.size();i++) {
		vector<double> temp(1);
		
			//std::cout << data[i][j] << " ";
		temp[0] = data[i][data[0].size()-1];

		
		df["y"].push_back(temp);
		//std::cout << "\n";
	}

	return df;




}

std::unordered_map <std::string, std::vector<double>> read_csv_dataframe(std::string filename) {
	unordered_map <std::string, std::vector<double>> dataframe;
	std::ifstream t(filename);
	string code((std::istreambuf_iterator<char>(t)),
	std::istreambuf_iterator<char>());
	
	vector <string> rows= split(code,'\n');
	vector <string> columns = split(rows[0],',');
	vector <vector<double>> data;

	for (int i = 1;i < rows.size();i++) {
		vector <string> temp = split(rows[i], ',');
		vector <double> temp2;
		for (auto& p : temp) {
			temp2.push_back(std::stod(p));
		}
		
		data.push_back(temp2);
	}
	for (auto& p : data) {
		for (auto& pp : p) {
			std::cout << pp<<"\t" ;
		}
		std::cout << "\n";
	}

	for (int i = 0;i < data[0].size();i++) {
		dataframe[columns[i]] = {};
		for (int j = 0;j < data.size();j++) {
			dataframe[columns[i]].push_back(data[j][i]);
		}
	}

	



	return dataframe;


}