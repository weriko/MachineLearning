#include "modules.h"
#include "NeuralNetwork.h"
#include "utils.h"

typedef std::vector<std::vector<double>> mat;
typedef std::vector<mat> tensor;



mat rand_mat(int size1, int size2, double epsi) {
	epsi = 0.1;
	srand(time(NULL) + rand());
	mat w(size1);
	for (auto& p : w) {
		std::vector <double> temp(size2);
		for (int i = 0;i < size2;i++) {
			temp[i] = ((double)rand() / ((double)RAND_MAX + 1)) * epsi; //* ((double)rand() < (double)RAND_MAX / 2 ? -1 : 1);

			//std::cout << temp[i]<<" ";
		}
		p = temp;
	}


	return w;
}
NeuralNetwork::NeuralNetwork(std::vector<int> ls, std::vector<std::string> la) {
	layer_sizes = ls;

	lr = 0.01;
	layer_activations = la;
	epsilon = 0.000001;

}


void NeuralNetwork::initialize() {
	for (int i = 0;i < NeuralNetwork::layer_sizes.size() - 1;i++) {
		 
		
		weights.push_back(rand_mat(NeuralNetwork::layer_sizes[i + 1], NeuralNetwork::layer_sizes[i], NeuralNetwork::epsilon*100000));
		beta.push_back(rand_mat(NeuralNetwork::layer_sizes[i + 1], 1, NeuralNetwork::epsilon));
		//std::cout << NeuralNetwork::layer_sizes[i + 1];
		
		
	}
	//std::cout << " AAAAAAAAAAAA " << beta.size() << " " << beta[1].size() << beta[0][0].size() << "\n";

}


mat NeuralNetwork::relu(mat x)
{
	mat temp = x;
	for (auto& p : temp) {
		std::for_each(p.begin(), p.end(), [](double& n) { n = (n > 0) * n; });

	}
	return temp;

}

mat NeuralNetwork::sigmoid(mat x) {
	mat temp = x;
	for (auto& p : temp) {
		std::for_each(p.begin(), p.end(), [](double& n) { n = 1.0 / ((1.0 + std::exp((-n)))); });

	}
	return temp;
}


mat NeuralNetwork::matmul(mat mat1, mat mat2) {
	mat temp(mat1.size());
	for (size_t i = 0; i < mat1.size();i++) {
		std::vector <double> temp2(mat2[0].size());
		temp[i] = (temp2);

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

mat NeuralNetwork::gsigmoid(mat da, mat x)
{
	mat temp = sigmoid(x);
	for (int i = 0; i < x.size();i++) {
		for (int j = 0; j < x[0].size();j++) {
			temp[i][j] = da[i][j] * temp[i][j] * (1 - temp[i][j]);

		}


	}
	return temp;
}

mat NeuralNetwork::grelu(mat da, mat x) {

	

	mat temp = relu(x);
	

	for (int i = 0; i < x.size();i++) {
		for (int j = 0; j < x[0].size();j++) {
			temp[i][j] = (x[i][j] > 0) * da[i][j];

		}


	}
	return temp;
}

std::vector<tensor> NeuralNetwork::propagate(mat x) {
	std::vector<tensor> cacheW;
	mat z = NeuralNetwork::transpose(x);
	mat a = (this->*activation_functions[layer_activations[0]])(z);

	
	for (int i = 0;i < weights.size();i++) {
		cacheW.push_back({ a,z });
		
		
		z = NeuralNetwork::matmul(weights[i], a);
		//print_mat(weights[i]);
		//print_mat(z);
		//std::cout << " AAAAAAAAAAAA " << beta.size() << " " << beta[0].size()<<"\n";
		//std::cout << " zzzzzzzzzzzzz " << z.size() << " " << z[0].size() << "\n";

		for (int ii = 0;ii < z.size();ii++) {
			for (int jj = 0;jj < z[0].size();jj++) {
				
				
				z[ii][jj] += beta[i][ii][0];
			}
			//std::cout << beta[i][ii][0];
		}

		a = (this->*activation_functions[layer_activations[i]])(z);
		//print_mat(a);
		//std::cout << a.size() << " ";



	}
	
	cacheW.push_back({ a,z });
	return cacheW;



}
mat NeuralNetwork::transpose(mat x) {
	mat temp(x[0].size());
	for (size_t i = 0;i < x[0].size();i++) {
		std::vector <double> n(x.size());

		for (size_t j = 0; j < x.size();j++) {
			n[j] = (x[j][i]);
		}
		temp[i] = n;

	}
	return temp;

}



std::vector<tensor> NeuralNetwork::backpropagate(mat y, mat ypred, std::vector<tensor> cache) {
	tensor gradsW;
	tensor gradsb;

	mat da(y.size());
	
	
	for (int i = 0;i < y.size();i++) {
		std::vector <double> temp(y.size());
		
		for (int j = 0;j < y[0].size();j++) {
			
			temp[j] = -((y[i][j] / (ypred[j][i]+epsilon)) - ((1 - y[i][j]) / ((1 - ypred[j][i])   +epsilon))  ); //ypred has the inverse shape of y

		}
		da[i] =temp;
	}
	

	for (int i = weights.size() - 1;i >= 0;i--) {
		
		
		mat a_prev = cache[i][0];
		mat z = cache[i + 1][1];

		mat w = weights[i];
		mat b = beta[i];


		mat dz = (this->*gactivation_functions["g"+layer_activations[i]])(da, z);
		
		mat dw = NeuralNetwork::matmul(dz, NeuralNetwork::transpose(a_prev));
		
		
		for (auto& p : dw) {
			for (auto& pp : p) {

				pp /= a_prev[0].size();
				//std::cout << pp<<" ";

			}
		}

		mat db;

		for (int i = 0;i < dz.size();i++) {
			double t = 0;
			for (int j = 0;j < dz[i].size();j++) {
				t += dz[i][j];
				 
			}
			db.push_back({ t/a_prev[0].size() });

		}
		//std::cout << db.size() << " "<<db[0].size()<<"\n";
		da = NeuralNetwork::matmul(NeuralNetwork::transpose(w), dz);

		gradsW.push_back(dw);
		gradsb.push_back(db);



	}
	return { gradsW,gradsb };


}

void NeuralNetwork::update(std::vector<tensor> grads) {
	for (int i = 0;i < weights.size();i++) {
		for (int ii = 0 ;ii < weights[i].size();ii++) {
			for (int jj = 0;jj < weights[i][ii].size();jj++) {
			    //std::cout << grads[0][grads[0].size() - i - 1][ii][jj]<<" ";
				
				
				weights[i][ii][jj] -= NeuralNetwork::lr * grads[0][grads[0].size() - i - 1][ii][jj] ; //grads is inverted
				
				
			}
			beta[i][ii][0] -= NeuralNetwork::lr * grads[1][grads[1].size() - i - 1][ii][0] ; //grads is inverted
			
		}
	}


}


void NeuralNetwork::fit(mat x, mat y, int epochs) {
	
	//std::cout << weights.size();
	NeuralNetwork::initialize();
	//print_tensor(NeuralNetwork::weights);
	
	//print_tensor(weights);
	std::vector<tensor> grads;
	std::cout << "\n";
	for (int i = 0;i < epochs;i++) {
		
		std::vector<tensor> cachew = NeuralNetwork::propagate(x);
		//print_tensor(cachew[1]);
		/*
		mat yfuckyou = cachew[cachew.size() - 1][0];
		for (auto& p : yyou[0]) {
			std::cout << p << " ";
		}*/
		
		grads = NeuralNetwork::backpropagate(y, cachew[cachew.size()-1][0], cachew);
		NeuralNetwork::update(grads);
	}
	//print_tensor(weights);

	

}

mat NeuralNetwork::predict(mat x) {
	std::vector<tensor> a = NeuralNetwork::propagate(x);
	return NeuralNetwork::transpose(a[a.size()-1][0]);


}



