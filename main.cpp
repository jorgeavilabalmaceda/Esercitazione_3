#include <iostream>
#include "Eigen/Eigen"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;


//relative error
double err_rel(const VectorXd& x_calc, const VectorXd& x_true) {
    return (x_calc - x_true).norm() / x_true.norm();
};

//PALU function (LU decomposizione con pivoting parziale)
Vector2d calcolo_con_PALU(const Matrix2d& A, const Vector2d& b) {
	FullPivLU<Matrix2d> lu(A);  //decomposizione
	Vector2d x;
	x = lu.solve(b);
    return x; 
};
//checking function
void check_solution(const Matrix2d& A, const Vector2d& b, const Vector2d& x, const string& label) {
    Vector2d residual = A * x - b;
    double res_norm = residual.norm();

    cout << "=== " << label << " ===" << endl;
    cout << "x = " << x.transpose() << endl;
    cout << "Residual norm ||Ax - b|| = " << res_norm << endl;
    cout << endl;
}




//QR function
Vector2d calcolo_con_QR(const Matrix2d& A, const Vector2d& b) {
    HouseholderQR<MatrixXd> qr(A);  //decomposizione
    return qr.solve(b); 
};

int main()
{
	Vector2d x_true;
	x_true << -1.0e+0, -1.0e+00;
	cout <<"x vero è: "<< x_true << endl;
	
	Matrix2d A1; 
	A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
	Vector2d b1;
	b1 << -5.169911863249772e-01, 1.672384680188350e-01;
	cout << "A1: "<< A1 << endl;
	cout <<"b1: " << b1 << endl;
	
	Matrix2d A2;
	A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
	Vector2d b2;
	b2 << -6.394645785530173e-04, 4.259549612877223e-04;
	
	cout <<"A2: "<< A2 << endl;
	cout <<"b2: "<< b2 << endl;
	
	Matrix2d A3;
	A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
	Vector2d b3;
	b3 << -6.400391328043042e-10, 4.266924591433963e-10;
	
	cout <<"A3: "<< A3 << endl;
	cout <<"b3: " << b3 << endl;
	
	//PA=LU solution
    VectorXd x1_PALU = calcolo_con_PALU(A1, b1);
	cout << x1_PALU << endl;
	check_solution(A1, b1, x1_PALU, "System 1");
	double err_palu_1=err_rel(x1_PALU,x_true);
	cout << "l'errore relativo: " <<err_palu_1<<endl;
	
	VectorXd x2_PALU = calcolo_con_PALU(A2, b2);
	cout << x2_PALU << endl;
	check_solution(A2, b2, x2_PALU, "System 2");
	double err_palu_2=err_rel(x2_PALU,x_true);
	cout << "l'errore relativo: " <<err_palu_2<<endl;
	
	
	VectorXd x3_PALU = calcolo_con_PALU(A3, b3);
	check_solution(A3, b3, x3_PALU, "System 3");
	cout << x3_PALU << endl;
	double err_palu_3=err_rel(x3_PALU,x_true);
	cout << "l'errore relativo: " <<err_palu_3<<endl;
	//QR solution
	VectorXd x1_QR = calcolo_con_QR(A1, b1);
	cout << x1_QR << endl;
	check_solution(A1, b1, x1_QR, "System 1 con QR");
	double err_qr_1=err_rel(x1_QR,x_true);
	cout << "l'errore relativo: " <<err_qr_1<<endl;
	
	VectorXd x2_QR = calcolo_con_QR(A2, b2);
	cout << x2_QR << endl;
	check_solution(A2, b2, x2_QR, "System 2 con QR");
	double err_qr_2=err_rel(x2_QR,x_true);
	cout << "l'errore relativo: " <<err_qr_2<<endl;
	
	VectorXd x3_QR = calcolo_con_QR(A3, b3);
	cout << x3_QR << endl;
	check_solution(A3, b3, x3_QR, "System 3 con QR");
	double err_qr_3=err_rel(x3_QR,x_true);
	cout << "l'errore relativo: " <<err_qr_3<<endl;
	
    return 0;
}