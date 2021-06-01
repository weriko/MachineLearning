
#include "modules.h"




void print_tensor(tensor a) {
	for (int i = 0;i < a.size();i++) {
		for (size_t ii = 0; ii < a[i].size(); ii++) {

			for (size_t jj = 0; jj < a[i][ii].size(); jj++)
				std::cout << a[i][ii][jj] << " ";
			std::cout << "\n";
		}
	}

}
void print_mat(mat a) {
	//std::cout << "\n-----------------------------\n";
	//std::cout << "\nPrinting mat with size " << a.size() << " " << a[0].size() << "\n";
	for (int i = 0;i < a.size();i++) {
		for (size_t ii = 0; ii < a[i].size(); ii++) {
			std::cout << a[i][ii] << " ";

		}
		std::cout << "\n";
	}

}
void print_vector(std::vector<double> a) {
	for (auto &p: a) {
			std::cout <<p << " ";
		std::cout << "\n";
	}


}

void print_acc(mat ypred, mat y) {
	double sum = 0; 
	for (int i = 0; i < ypred.size();i++) {
		sum += std::abs(ypred[i][0] - y[i][0]) < 0.5;
	}
	std::cout <<  sum / ypred.size() << "\n";
}