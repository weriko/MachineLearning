#include "modules.h"


typedef std::vector<std::vector<double>> mat;
void print_vector(mat);


class Regressor {
public:
	double lr;
	double epsilon;
	mat x;
	std::vector <std::vector<double>> y;
	double beta;
	std::vector <double> losses;
	mat weights;
	Regressor(double, double);

	void fit(mat x, std::vector<std::vector<double>> y);

	mat sigmoid(mat);
	mat forward(mat);
	double loss(mat ypred, mat y);
	mat matmul(mat, mat);
	mat transpose(mat);
	void regress(mat x, std::vector<std::vector<double>> y, int epochs);
	mat predict(mat);

};

double rand_iter() {

	return(double)rand() / ((double)RAND_MAX + 1);
}


Regressor::Regressor(double c, double d)
{

	epsilon = d;
	lr = c;
	beta = 0;
	std::vector <double> losses;
}

void Regressor::fit(mat x, std::vector<std::vector<double>> y) {
	srand(time(NULL)+rand());
	std::vector <double> w(x[0].size());


	generate(w.begin(), w.end(), rand_iter);
	weights.push_back(w);
	//print_vector(weights);
}


mat Regressor::sigmoid(mat x) {
	mat temp = x;
	for (auto& p : temp) {
		std::for_each(p.begin(), p.end(), [](double& n) { n = 1.0 / ((1.0 + std::exp((-n)))); });

	}
	return temp;
}

double Regressor::loss(mat ypred, mat y) {
	double temp=0;
	for (int i = 0;i < ypred.size();i++) {
		temp += y[i][0] * std::log(ypred[i][0]+Regressor::epsilon)+(1-y[i][0])*log(1-ypred[i][0]+Regressor::epsilon);
	}
	
	return -temp / ypred.size();

}

mat Regressor::matmul(mat mat1, mat mat2) {
	mat temp(mat1.size());
	for (size_t i = 0; i < mat1.size();i++) {
		std::vector <double> temp2(mat2[0].size());
		temp[i]=(temp2);

	}
	for (size_t i = 0;i < mat1.size();i++) {
		for (size_t j = 0; j < mat2[0].size();j++) {
			for (size_t k = 0; k < mat2.size();k++) {
				temp[i][j] += mat1[i][k] * mat2[k][j];
			}
		}
	}
	return temp;

}

mat Regressor::transpose(mat x) {
	mat temp(x[0].size());
	for (size_t i = 0;i < x[0].size();i++) {
		std::vector <double> n(x.size());

		for (size_t j = 0; j < x.size();j++) {
			n[j]=(x[j][i]);
		}
		temp[i]=n;

	}
	return temp;

}

void   Regressor::regress(mat x, std::vector<std::vector<double>> y, int epochs) {

	for (int _ = 0; _ < epochs;_++) {
		mat ypred(y.size());
		mat new_w(Regressor::weights.size());
		

		ypred = Regressor::forward(x);
		mat ytemp(ypred.size());

		for (size_t i = 0; i < ypred.size();i++) {
			ytemp[i]= { ypred[i][0] - y[i][0] };

		}

		new_w = Regressor::matmul(Regressor::transpose(x), ytemp);
		//print_vector(new_w);

		for (size_t i = 0;i < weights[0].size();i++) {


			weights[0][i] -= (Regressor::lr * new_w[i][0]) / x.size();

		}
		double t = 0;
		for (auto& p : ytemp) {
			t += p[0];
		}
		beta -= (Regressor::lr * t) / x.size();
		Regressor::losses.push_back(Regressor::loss(ypred, y));
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
	for (size_t i = 0; i < vec.size(); i++) {
		for (size_t j = 0; j < vec[i].size(); j++)
			std::cout << vec[i][j] << " ";
		std::cout << "\n";
	}
	//system("pause");
}




