
#include "modules.h"



typedef std::vector<std::vector<double>> mat;
void print_vector(mat vec);


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
	mat matmul(mat, mat);
	mat transpose(mat);
	void regress(mat x, std::vector<std::vector<double>> y, int epochs);
	mat predict(mat);

	double loss(mat ypred, mat y);

};
