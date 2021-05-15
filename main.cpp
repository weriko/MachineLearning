#include "LogisticRegressor.h"
#include "Reader.h"

void print_map(unordered_map<string, vector<double>> const& m)
{
	for (auto const& pair : m) {
		std::cout << " " << pair.first << "\t ";
		for (auto& n : pair.second) {
			std::cout << n<<" ";
		}
		std::cout << "\n";
	}
}
void print_acc(mat ypred, mat y) {
	double sum = 0;
	for (int i = 0; i < ypred.size();i++) {
		sum += std::abs(ypred[i][0] - y[i][0])<0.5;
	}
	std::cout << sum / ypred.size()<<"\n";
}
int main() {

	//mat x = { {340,530},{1,2},{532,234},{1,3},{523,532} };
	//mat y = { {1},{0},{1},{0},{1} };
	Regressor logistic_regression(0.008, 0.0000001);
	
	//print_vector(logistic_regression.predict(x));
	vector <vector<double>> data = read_csv("train_mod.csv");
	//std::cout << "a0";
	std::unordered_map <std::string, std::vector<vector<double>> > data_x_y = to_x_y(data);
	mat x = data_x_y["x"];
	mat y = data_x_y["y"];
	//std::cout << "a1";
	logistic_regression.fit(x, y);
	//std::cout << "a2";
	logistic_regression.regress(x, y, 20000);
	//std::cout << "a3";
	auto preds = logistic_regression.predict(x);
	//print_vector(preds);
	std::cout << "Precision   ";
	print_acc(preds, y);
	/*
	for (auto& p : logistic_regression.losses) {
		std::cout << " " << p << "\n";
	}
	std::cout << "\nweights-----";
	for (auto& p : logistic_regression.weights[0]) {
		std::cout << " " << p << "\n";
	}
	std::cout << logistic_regression.beta;*/
	


}