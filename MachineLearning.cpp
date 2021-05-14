#include <vector>
#include <iostream>
#include <stdio.h>
#include <random>
#include <stdlib.h>
#include <math.h>
#include <algorithm>


typedef std::vector<std::vector<double>> mat;
void print_vector(mat vec);


class Regressor {
public:
	double lr;
	double epsilon;
	mat x;
	std::vector <std::vector<int>> y;
	double beta;
	std::vector <double> losses;
	mat weights;
	Regressor(float, float);

	void fit(mat x, std::vector<std::vector<int>> y);

	mat sigmoid(mat);
	mat forward(mat);
	mat matmul(mat, mat);
	mat transpose(mat);
	void regress(mat x, std::vector<std::vector<int>> y, int epochs);
	mat predict(mat);

};

float rand_iter() {

	return(double)rand() / ((double)RAND_MAX + 1);
}


Regressor::Regressor(float c, float d)
{

	epsilon = d;
	lr = c;
	beta = 0;
	std::vector <double> losses;
}

void Regressor::fit(mat x, std::vector<std::vector<int>> y) {
	srand(time(0));
	std::vector <double> w(x[0].size());


	generate(w.begin(), w.end(), rand_iter);
	weights.push_back(w);
}


mat Regressor::sigmoid(mat x) {
	mat temp = x;
	for (auto& p : temp) {
		std::for_each(p.begin(), p.end(), [](double& n) { n = 1.0 / ((1.0 + std::exp((-n)))); });

	}
	return temp;

}

mat Regressor::matmul(mat mat1, mat mat2) {
	mat temp;
	for (int i = 0; i < mat1.size();i++) {
		std::vector <double> temp2(mat2[0].size());
		temp.push_back(temp2);

	}
	for (int i = 0;i < mat1.size();i++) {
		for (int j = 0; j < mat2[0].size();j++) {
			for (int k = 0; k < mat2.size();k++) {
				temp[i][j] += mat1[i][k] * mat2[k][j];
			}
		}
	}
	return temp;

}

mat Regressor::transpose(mat x) {
	mat temp;
	for (int i = 0;i < x[0].size();i++) {
		std::vector <double> n;

		for (int j = 0; j < x.size();j++) {
			n.push_back(x[j][i]);
		}
		temp.push_back(n);

	}
	return temp;

}

void   Regressor::regress(mat x, std::vector<std::vector<int>> y, int epochs) {

	for (int _ = 0; _ < epochs;_++) {
		mat ypred;
		mat new_w;
		mat ytemp;

		ypred = Regressor::forward(x);

		for (int i = 0; i < ypred.size();i++) {
			ytemp.push_back({ ypred[i][0] - y[i][0] });

		}

		new_w = Regressor::matmul(x, Regressor::transpose(ytemp));
		for (int i = 0;i < weights[0].size();i++) {


			weights[0][i] -= (Regressor::lr * new_w[0][i]) / x.size();

		}
		double t = 0;
		for (auto& p : ytemp) {
			t += p[0];
		}
		beta -= (Regressor::lr * t) / x.size();

	}

}

mat Regressor::predict(mat x) {
	return Regressor::forward(x);

}

mat Regressor::forward(mat x) {
	mat temp = Regressor::matmul(x, Regressor::transpose(Regressor::weights));
	double b = Regressor::beta;

	for (auto& p : temp) {

		std::for_each(p.begin(), p.end(), [&](double& n) { n = n + b; }); // adds beta to all values in std::vector

	}

	return Regressor::sigmoid(temp);

}

void print_vector(mat vec) {
	for (int i = 0; i < vec.size(); i++) {
		for (int j = 0; j < vec[i].size(); j++)
			std::cout << vec[i][j] << " ";
		std::cout << "\n";
	}
}

int main() {

	mat x = { {340,530},{1,2},{532,234},{1,3},{523,532} };
	std::vector <std::vector<int>> y = { {1},{0},{1},{0},{1} };
	Regressor logistic_regression(0.1, 0.1);
	logistic_regression.fit(x, y);
	logistic_regression.regress(x, y, 100);
	print_vector(logistic_regression.predict(x));

}



