#include <vector>
#include <iostream>
#include <stdio.h>
#include <random>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <unordered_map>
typedef std::vector<std::vector<double>> mat;
typedef std::vector<std::vector<std::vector<double>>> tensor;
mat rand_mat(int size1, int size2, double epsi);

class NeuralNetwork {
public:
	double epsilon;
	std::vector <int> layer_sizes;
	std::vector <std::string> layer_activations;
	void initialize();
	tensor weights;
	mat matmul(mat, mat);
	tensor beta;
	double lr;
	mat transpose(mat);
	NeuralNetwork(std::vector<int>, std::vector<std::string>);
	std::vector<tensor> backpropagate(mat, mat, std::vector<tensor>);
	typedef mat(NeuralNetwork::* func)(mat k);
	typedef mat(NeuralNetwork::* gfunc)(mat da, mat k);
	void fit(mat , mat , int);
	void update(std::vector<tensor>);
	mat predict(mat);
	mat relu(mat);
	mat sigmoid(mat);
	mat grelu(mat, mat);
	mat gsigmoid(mat, mat);
	std::vector<tensor> propagate(mat);

	std::unordered_map<std::string, func> activation_functions = {
		{"relu",&NeuralNetwork::relu},
		{"sigmoid",&NeuralNetwork::sigmoid},
	};
	std::unordered_map<std::string, gfunc> gactivation_functions = {
		{"grelu",&NeuralNetwork::grelu},
		{"gsigmoid",&NeuralNetwork::gsigmoid},
	};


};



