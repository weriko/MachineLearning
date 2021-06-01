#include "LogisticRegressor.h"
#include "Reader.h"
#include "NeuralNetwork.h"
#include "utils.h"
#include "modules.h"
int main() {
	
	/*
	Regressor logistic_regression(0.008, 0.0000001);
	
	
	vector <vector<double>> data = read_csv("train_mod.csv");
	
	std::unordered_map <std::string, std::vector<vector<double>> > data_x_y = to_x_y(data);
	mat x = data_x_y["x"];
	mat y = data_x_y["y"];
	
	logistic_regression.fit(x, y);
	
	logistic_regression.regress(x, y, 20000);
	
	auto preds = logistic_regression.predict(x);
	
	std::cout << "Precision   ";
	print_acc(preds, y);*/

	/*
	for (auto& p : logistic_regression.losses) {
		std::cout << " " << p << "\n";
	}
	std::cout << "\nweights-----";
	for (auto& p : logistic_regression.weights[0]) {
		std::cout << " " << p << "\n";
	}
	std::cout << logistic_regression.beta;*/

	/*
	NeuralNetwork NN({ 5,3,2,1 }, { "relu","relu","relu","sigmoid" });
	mat x = rand_mat(6, 2, 10);
	print_vector(x);
	std::cout << "sigmoid\n";
	print_vector((NN.*NN.activation_functions["sigmoid"])(x));
	std::cout << "gsigmoid\n";
	print_vector(NN.gsigmoid(x, x));
	std::cout << "relu\n";
	print_vector(NN.relu(x));
	std::cout << "grelu\n";
	print_vector(NN.grelu(x,x));*/
	/*
	NeuralNetwork NN({ 4,6,4,1 }, { "relu","relu","sigmoid" });
	//mat x = rand_mat(5, 2, 10); // put an assert in case we are dumb and use the wrong dimensions
	
	//NN.initialize();
	//std::vector<tensor> p = NN.propagate(x);
	//print_vector(p[3][0] );
	vector <vector<double>> data = read_csv("train_mod2.csv");
	//std::cout << "a0";
	std::unordered_map <std::string, std::vector<vector<double>> > data_x_y = to_x_y(data);
	mat x = data_x_y["x"];
	mat y = data_x_y["y"];

	

	
	NN.fit(x, y,10000 );

	//print_vector(NN.predict(x));
	
	auto preds = NN.predict(x);
    print_vector(preds);
	//std::cout << "Precision   ";
	print_acc(preds, y);
	//print_tensor(NN.weights);
	//print_tensor(NN.beta);*/

}