#include <fstream>
#include <iomanip>
#include <random>
#include <tuple>
#include <vector>
#include <chrono>

using namespace std;

// Numerical simulation of the Cox-Ingersoll-Ross (CIR) model on [0, T]:
// dy_t = a(b-y_t) dt + sigma sqrt(y_t) dW_t

class CIRmethods {
    public:
        // Input parameters
        double a, b, sigma, stepsize_control;

        // Variable step size sample path of the CIR process
        vector< pair<double, double> > sample_path;

        // Used to count the number of steps performed in the simulations
        int step_counter = 0;

        CIRmethods(double, double, double, double);

        pair<tuple<double, double, double>, tuple<double, double, double> >
            refine(double, double, double, double);

        double implicit_euler(double, double, double);

        double linear_ode_step(double, double, double);

        double zigzag_ode_step
            (double, double, double, double, double, double);

        double fine_zigzag_ode_step
            (double, double, double, int, double, double, double);

        pair<double, double> variable_ode_step
            (pair<double, double>, double, double,
             int, int, double, double, double);

        double variable_ode_step_test
            (double, double, double, int, double, double, double);

    private:
        // Precomputed values that depend on the input parameters
        double half_sigma, sigma_squared, tildeb, atildeb;
        double rescaled_control;

        // Pseudorandom number generator for generating the Brownian path
        std::random_device generator;
        std::normal_distribution<double> arch_normal_distribution;
        std::normal_distribution<double> sign_normal_distribution;
        std::bernoulli_distribution rademacher_distribution;

        // Runge-Kutta coefficients
        const double root_three = sqrt(3.0);
        const double three_minus_root_three = 3.0 - root_three;
        const double root_three_minus_two = root_three - 2.0;
        const double six_minus_two_root_three = 6.0 - 2.0*root_three;
        const double fifteen_minus_nine_root_three_over_two
                        = 7.5 - 4.5*root_three;

        // High order piecewise linear path coefficients
        const double pi = acos(-1.0);
        const double piece_lin_const_1 = -3.0/sqrt(6.0*pi);
        const double root_ten = sqrt(10.0);
        const double small_step = (8.0 - root_ten)/18.0;
        const double middle_step = (1.0+root_ten)/9.0;
        const double big_step = 1.0 - small_step;
        const double big_step_squared = pow(big_step, 2);
        const double half_small_step = 0.5*small_step;
        const double one_over_big_step = 2.0 - 0.2*root_ten;
        const double twelve_over_big_step_squared = 12.0/big_step_squared;
        const double piece_lin_const_2 = twelve_over_big_step_squared \
                                            /(8.0*sqrt(6.0*pi));

        // Variable step sizes coefficients
        const double root_half = sqrt(0.5);
        const double variable_control_coeff = (7.0/3600.0) - (1.0/(384.0*pi));
        const double eleven_over_two_five_two = 11.0/25200.0;
        const double increment_squared_coeff = (1.0/720.0) - (1.0/(384.0*pi));
        const double area_squared_coeff = 1.0/700.0;
        const double sign_coeff = -1.0/(320.0*sqrt(6.0*pi));
};

// Constructor will initialize the above private variables
CIRmethods::CIRmethods(double input_a, double input_b, double input_sigma,
                       double input_stepsize_control){

    a = input_a;
    b = input_b;
    sigma = input_sigma;
    stepsize_control = input_stepsize_control;

    std::normal_distribution<double> temp_distribution_1(0.0, 0.25);
    std::normal_distribution<double> temp_distribution_2(0.0, sqrt(1.0/48.0));
    std::bernoulli_distribution temp_distribution_3(0.5);

    arch_normal_distribution = temp_distribution_1;
    sign_normal_distribution = temp_distribution_2;
    rademacher_distribution = temp_distribution_3;

    half_sigma = 0.5*sigma;
    sigma_squared = pow(input_sigma, 2);

    tildeb = b - (0.25/a)*sigma_squared;
    atildeb = a*b - 0.25*sigma_squared;

    rescaled_control = variable_control_coeff*pow(input_stepsize_control, 3);

    sample_path.push_back(make_pair(0.0, 0.0));
};

// Dyadic refinement procedure for the increment,
// space-time Levy area and space-time orientation
// of the Brownian path
pair<tuple<double, double, double>, tuple<double, double, double> >
    CIRmethods::refine(double sqrt_h, double brownian_increment,
                       double brownian_area, double brownian_sign){

    // Here z is the midpoint of the Brownian arch Z
    // and n denotes the random variable N_{s,t}.
    double z = sqrt_h*arch_normal_distribution(generator);
    double n = brownian_sign*abs(sqrt_h*sign_normal_distribution(generator));

    double first_increment = 0.5*brownian_increment + 1.5*brownian_area + z;

    double symmetric_area = 0.25*brownian_area - 0.5*z;

    double sign_1 = 1.0;
    double sign_2 = 1.0;

    if (rademacher_distribution(generator)){
        sign_1 = -1.0;
    }

    if (rademacher_distribution(generator)){
        sign_2 = -1.0;
    }

    return make_pair(make_tuple(first_increment, symmetric_area + n,  sign_1),
                     make_tuple(brownian_increment - first_increment,
                                symmetric_area - n, sign_2));
};

/*
    One step of the drift-implicit Euler method detailed in the paper:

    A. Alfonsi, On the discretization schemes for the CIR
    (and Bessel squared) processes, Monte Carlo Methods
    and Applications, Volume 11, 2005.
*/
double CIRmethods::implicit_euler(double sqrt_y, double h, double brownian_increment){

    double additive_noise = sqrt_y + half_sigma*brownian_increment;

    double constant1 = 2.0 + a*h;
    double constant2 = constant1*atildeb*h;

    return (additive_noise + sqrt((pow(additive_noise, 2)) + constant2)) \
                /constant1;
};

/*
    One step of the third order A-stable diagonally
    implicit Runge-Kutta method presented in

    C. A. Kennedy and M. H. Carpenter,
    Diagonally Implicit Runge-Kutta Methods
    For Ordinary Differential Equations: A Review,
    NASA Scientific and Technical Information, Hampton, 2016.

    For the (square root) CIR process, this method reduces to
    finding the positive roots of two quadratic equations.
*/
double CIRmethods::linear_ode_step(double sqrt_y, double h,
                                   double brownian_increment){

    double ah = a*h;

    double minus_quadratic_c = tildeb*ah;

    // Evaluate the square root CIR vector field
    double k1 = 0.5*((minus_quadratic_c/sqrt_y) - ah*sqrt_y) \
                    + half_sigma*brownian_increment;

    double minus_half_quadratic_b = three_minus_root_three*sqrt_y \
                                     + k1 + half_sigma*brownian_increment;

    double quadratic_a = six_minus_two_root_three + ah;
    double quadratic_ac = quadratic_a*minus_quadratic_c;
    double one_over_quadratic_a = 1.0/quadratic_a;

    // Solve first quadratic equation
    double temp_sqrt_y = one_over_quadratic_a \
                        * (minus_half_quadratic_b \
                              + sqrt(pow(minus_half_quadratic_b, 2) \
                                  + quadratic_ac));

    minus_half_quadratic_b = minus_half_quadratic_b \
                                + root_three_minus_two*k1 \
                                + fifteen_minus_nine_root_three_over_two\
                                    *(temp_sqrt_y - sqrt_y);

    // Solve second quadratic equation
    return one_over_quadratic_a \
            * (minus_half_quadratic_b + sqrt(pow(minus_half_quadratic_b, 2) \
                                             + quadratic_ac));
};

// Method for propagating the numerical solution along each part of \hat{W}
double CIRmethods::zigzag_ode_step(double sqrt_y, double h, double sqrt_h,
                                   double brownian_increment,
                                   double brownian_area,
                                   double brownian_sign){

    double discriminant = big_step_squared*pow(brownian_increment, 2) \
                            + piece_lin_const_1*sqrt_h*brownian_increment \
                                *brownian_sign + 0.8*h;

    double half_of_cplusb = 0.5*brownian_increment \
                               + one_over_big_step*brownian_area;

    double phi = 0.5;

    if  (piece_lin_const_2*sqrt_h*brownian_sign > brownian_increment){
        phi = -0.5;
    }

    // Compute the connecting points for the discretized Brownian path \hat{W}
    double half_of_cminusb = phi*sqrt(discriminant) \
                                - half_small_step*brownian_increment;

    double b = half_of_cplusb - half_of_cminusb;
    double c = half_of_cplusb + half_of_cminusb;

    // Propagate the numerical solution along the piecewise linear path
    sqrt_y = linear_ode_step(sqrt_y, small_step*h, b);
    sqrt_y = linear_ode_step(sqrt_y, middle_step*h, c - b);
    sqrt_y = linear_ode_step(sqrt_y, small_step*h, brownian_increment - c);

    return sqrt_y;
};

// Method for propagating the numerical solution along a
// finely generated piecewise linear approximant \hat{W}.
// The "fine" step size used is h * 2^(-fine_sub_divide_no).
double CIRmethods::fine_zigzag_ode_step(double sqrt_y,
                                        double h, double sqrt_h,
                                        int fine_sub_divide_no,
                                        double brownian_increment,
                                        double brownian_area,
                                        double brownian_sign){

    // This is the base case of the recursion
    if (fine_sub_divide_no == 0){
        sqrt_y = zigzag_ode_step(sqrt_y, h, sqrt_h, brownian_increment,
                                 brownian_area, brownian_sign);
    }
    else {
        double half_h = 0.5*h;
        double half_sqrt_h = root_half*sqrt_h;

        // Generate the Brownian path over the two half intervals
        pair<tuple<double, double, double>, tuple<double, double, double> >
            refined_bm = refine(sqrt_h, brownian_increment,
                                brownian_area, brownian_sign);

        // Compute numerical solution over the first half interval
        sqrt_y = fine_zigzag_ode_step(sqrt_y, half_h, half_sqrt_h,
                                      fine_sub_divide_no - 1,
                                      get<0>(refined_bm.first),
                                      get<1>(refined_bm.first),
                                      get<2>(refined_bm.first));

        // Compute numerical solution over the second half interval
        sqrt_y = fine_zigzag_ode_step(sqrt_y, half_h, half_sqrt_h,
                                      fine_sub_divide_no - 1,
                                      get<0>(refined_bm.second),
                                      get<1>(refined_bm.second),
                                      get<2>(refined_bm.second));
    }

    return sqrt_y;
};

// Method for propagating the numerical solution a given
// time interval using a variable step size methodology
pair<double, double>
    CIRmethods::variable_ode_step(pair<double, double> sqrt_ys,
                                  double h, double sqrt_h,
                                  int crude_sub_divide_no,
                                  int fine_sub_divide_no,
                                  double brownian_increment,
                                  double brownian_area,
                                  double brownian_sign){

    // Variable step size condition for controlling the local L2(P) error.
    // Note that crude_sub_divide_no is to ensure the recursion will stop.
    if ((pow(h,2)*(eleven_over_two_five_two*h \
                    + increment_squared_coeff*pow(brownian_increment, 2) \
                    + area_squared_coeff*pow(brownian_area, 2) \
                    + sign_coeff*brownian_sign*sqrt_h*brownian_increment) \
                        < rescaled_control*pow(sqrt_ys.first, 4))
        || (crude_sub_divide_no == 0)){

        // Propagate the pair of numerical solutions.
        // Note sqrt_ys.first is the "crude" approximation
        // and sqrt_ys.second is the "fine" approximation.
        // The fine step size is h * 2^(-fine_sub_divide_no).
        sqrt_ys = make_pair(zigzag_ode_step(sqrt_ys.first, h, sqrt_h,
                                            brownian_increment,
                                            brownian_area,
                                            brownian_sign),
                           fine_zigzag_ode_step(sqrt_ys.second, h, sqrt_h,
                                                max(fine_sub_divide_no, 6),
                                                brownian_increment,
                                                brownian_area,
                                                brownian_sign));

        // Add the new point to the sample path
        sample_path.push_back(make_pair(sample_path.back().first + h,
                                        pow(sqrt_ys.first, 2)));

        step_counter = step_counter + 1;
    }
    else {
        // If the condition fails, then we will half the step size
        double half_h = 0.5*h;
        double half_sqrt_h = root_half*sqrt_h;

        // Generate the Brownian path over the two half intervals
        pair<tuple<double, double, double>, tuple<double, double, double> >
            refined_bm = refine(sqrt_h, brownian_increment,
                                brownian_area, brownian_sign);

        // Compute numerical solution over the first half interval
        sqrt_ys = variable_ode_step(sqrt_ys, half_h, half_sqrt_h,
                                    crude_sub_divide_no - 1,
                                    fine_sub_divide_no - 1,
                                    get<0>(refined_bm.first),
                                    get<1>(refined_bm.first),
                                    get<2>(refined_bm.first));

        // Compute numerical solution over the second half interval
        sqrt_ys = variable_ode_step(sqrt_ys, half_h, half_sqrt_h,
                                    crude_sub_divide_no - 1,
                                    fine_sub_divide_no - 1,
                                    get<0>(refined_bm.second),
                                    get<1>(refined_bm.second),
                                    get<2>(refined_bm.second));
    }

    return sqrt_ys;
};

// Method for propagating the numerical solution a given
// time interval using a variable step size methodology
// (this is for the speed test and does not compare methods)
double CIRmethods::variable_ode_step_test(double sqrt_y,
                                          double h, double sqrt_h,
                                          int crude_sub_divide_no,
                                          double brownian_increment,
                                          double brownian_area,
                                          double brownian_sign){

    // Variable step size condition for controlling the local L2(P) error.
    // Note that crude_sub_divide_no is to ensure the recursion will stop.
    if ((pow(h,2)*(eleven_over_two_five_two*h \
                    + increment_squared_coeff*pow(brownian_increment, 2) \
                    + area_squared_coeff*pow(brownian_area, 2) \
                    + sign_coeff*brownian_sign*sqrt_h*brownian_increment) \
                        < rescaled_control*pow(sqrt_y, 4))
        || (crude_sub_divide_no == 0)){

        // Propagate the pair of numerical solutions.
        // Note sqrt_ys.first is the "crude" approximation
        // and sqrt_ys.second is the "fine" approximation.
        // The fine step size is h * 2^(-fine_sub_divide_no).
        sqrt_y = zigzag_ode_step(sqrt_y, h, sqrt_h,
                                            brownian_increment,
                                            brownian_area,
                                            brownian_sign);

        step_counter = step_counter + 1;
    }
    else {
        // If the condition fails, then we will half the step size
        double half_h = 0.5*h;
        double half_sqrt_h = root_half*sqrt_h;

        // Generate the Brownian path over the two half intervals
        pair<tuple<double, double, double>, tuple<double, double, double> >
            refined_bm = refine(sqrt_h, brownian_increment,
                                brownian_area, brownian_sign);

        // Compute numerical solution over the first half interval
        sqrt_y = variable_ode_step_test(sqrt_y, half_h, half_sqrt_h,
                                        crude_sub_divide_no - 1,
                                        get<0>(refined_bm.first),
                                        get<1>(refined_bm.first),
                                        get<2>(refined_bm.first));

        // Compute numerical solution over the second half interval
        sqrt_y = variable_ode_step_test(sqrt_y, half_h, half_sqrt_h,
                                        crude_sub_divide_no - 1,
                                        get<0>(refined_bm.second),
                                        get<1>(refined_bm.second),
                                        get<2>(refined_bm.second));
    }

    return sqrt_y;
};
int main()
{
    // Input parameters
    const double a = 1.0;
    const double b = 1.0;
    const double sigma = sqrt(3.0);
    const double y0 = 1.0;
    const double T = 1.0;
    const double stepsize_control = 0.0675;

    const double root_T = sqrt(T);

    CIRmethods CIR_method(a, b, sigma, stepsize_control);

    // Number of paths used for the Monte Carlo estimators
    const int no_of_paths = 100000;

    // The step size of the "crude" approximation is bounded
    // below by T * 2^(-max_no_crude_subdivisions).
    const int max_no_crude_subdivisions = 20;

    // The step size of the "fine" approximation is bounded
    // above by T * 2^(-min_no_fine_subdivisions).
    const int min_no_fine_subdivisions = 12;

    // Normal distributions for generating the various increments and areas
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> big_increment_distribution(0.0, root_T);
    std::normal_distribution<double> big_area_distribution(0.0, sqrt(T/12.0));
    std::bernoulli_distribution rademacher_distribution(0.5);

    // It will be more convenient to discretize the square root process
    const double sqrt_y0 = sqrt(y0);
    const pair<double, double> sqrt_y0s = make_pair(sqrt_y0, sqrt_y0);

    // Numerical solutions computed with course and fine step sizes
    pair<double, double> sqrt_ys = make_pair(sqrt_y0, sqrt_y0);

    // Information about the Brownian motion (increments and areas)
    // These objects are described in cir_presentation.pdf as:
    // brownian_increment is W_{s,t}
    // brownian_area      is H_{s,t}
    // brownian_sign      is n_{s,t}
    double brownian_increment = 0.0;
    double brownian_area = 0.0;
    double brownian_sign = 1.0;

    // Average number of steps used by the variable step size method
    double average_no_of_steps = 0.0;

    // Strong and weak error estimators for sqrt_ys at time T
    double end_point_error = 0.0;
    double crude_call_price = 0.0;
    double fine_call_price = 0.0;

    for (int i=0; i < no_of_paths; i++){
        // Generate information about the Brownian path over the whole interval
        brownian_increment = big_increment_distribution(generator);
        brownian_area = big_area_distribution(generator);

        if (rademacher_distribution(generator)){
            brownian_sign = - brownian_sign;
        }

        // Propagate the numerical solution over the whole interval [0,T]
        sqrt_ys = CIR_method.variable_ode_step(sqrt_y0s, T, root_T,
                                               max_no_crude_subdivisions,
                                               min_no_fine_subdivisions,
                                               brownian_increment,
                                               brownian_area,
                                               brownian_sign);

        // Compute the L2 error between the methods on the fine and course scales
        end_point_error = end_point_error + pow(pow(sqrt_ys.first, 2) \
                                                - pow(sqrt_ys.second, 2), 2);

        // Evaluate the call option payoffs
        crude_call_price = crude_call_price + max(0.0, pow(sqrt_ys.first, 2) - b);
        fine_call_price = fine_call_price + max(0.0, pow(sqrt_ys.second, 2) - b);

        // Record the number of steps used in the simulation
        average_no_of_steps = average_no_of_steps \
                                 + (double)CIR_method.step_counter;

        CIR_method.step_counter = 0;

        // Keep the sample path if we have reached the final iteration
        if (i != no_of_paths - 1){
            CIR_method.sample_path.clear();
            CIR_method.sample_path.push_back(make_pair(0.0, y0));
        }
        else {
            CIR_method.sample_path.front() = make_pair(0.0, y0);
        }
    }

    // Compute the various averages for estimating the strong and weak errors
    end_point_error = sqrt(end_point_error / (double(no_of_paths)));
    crude_call_price = crude_call_price / (double(no_of_paths));
    fine_call_price = fine_call_price / (double(no_of_paths));

    average_no_of_steps = average_no_of_steps / (double(no_of_paths));

    /*   We now estimate errors for the drift-implicit Euler method   */

    // Number of crude steps used by the drift-implicit Euler method
    const int no_of_steps = 100;

    // Number of steps used by the fine approximation
    // during each step of the crude numerical method.
    const int no_of_fine_steps = 50;

    // Step size parameters
    const double step_size =  T/(double)no_of_steps;
    const double fine_step_size = T/(double)(no_of_steps*no_of_fine_steps);

    // The drift-implicit Euler method only using increments of the Brownian path
    double fine_brownian_increment = 0.0;

    // Numerical solutions computed with course and fine step sizes
    double sqrt_y_euler = sqrt_y0;
    double sqrt_y_fine = sqrt_y0;

    // Strong and weak error estimators for sqrt_y_euler at time T
    double end_point_error_2 = 0.0;
    double call_option_error_2 = 0.0;

    // Normal distribution for generating "fine" increments of the Brownian path
    std::normal_distribution<double>
        fine_increment_distribution(0.0, sqrt(fine_step_size));


    for (int i=0; i<no_of_paths; ++i) {
        for (int j=1; j<=no_of_steps; ++j) {

            brownian_increment = 0.0;

            for (int k=1; k<= no_of_fine_steps; ++k){
                // Generate the "fine" increment of the Brownian path
                fine_brownian_increment = fine_increment_distribution(generator);

                // Propagate the numerical solution over the fine increment
                sqrt_y_fine = CIR_method.implicit_euler(sqrt_y_fine, fine_step_size,
                                                 fine_brownian_increment);

                // Update the Brownian path using the recently generated variable.
                brownian_increment = brownian_increment + fine_brownian_increment;
            }

            // Propagate the numerical solution over the course increment
            sqrt_y_euler = CIR_method.implicit_euler(sqrt_y_euler, step_size,
                                               brownian_increment);
        }

        // Compute the L2 error between the methods on the fine and course scales
        end_point_error_2 = end_point_error_2 + pow(pow(sqrt_y_euler,2) \
                                                    - pow(sqrt_y_fine, 2), 2);

        // Evaluate the call option payoffs
        call_option_error_2 = call_option_error_2 \
                                + abs(max(0.0, pow(sqrt_y_euler,2) - b) \
                                      - max(0.0, pow(sqrt_y_fine, 2) - b));

        // Reset the numerical solutions
        sqrt_y_euler = sqrt_y0;
        sqrt_y_fine = sqrt_y0;
    }

    // Compute the averages for estimating the strong and weak errors
    end_point_error_2 = sqrt(end_point_error_2 / (double(no_of_paths)));
    call_option_error_2 = call_option_error_2 / (double(no_of_paths));


    // Parameters used for the speed test
    const int no_of_paths_for_test = 100000;
    const double stepsize_control_test = 0.0675;
    const double sqrt_step_size = sqrt(step_size);
    double sqrt_y = sqrt_y0;
    double average_no_of_steps_test = 0.0;

    CIRmethods CIR_test(a, b, sigma, stepsize_control_test);

    // Start the speed test
    auto start = std::chrono::high_resolution_clock::now();

    // Run numerical method with a variable step size
    for (int i=0; i < no_of_paths_for_test; i++){

        // Generate information about the Brownian path over the whole interval
        brownian_increment = big_increment_distribution(generator);
        brownian_area = big_area_distribution(generator);

        if (rademacher_distribution(generator)){
            brownian_sign = - brownian_sign;
        }

        // Propagate the numerical solution over the whole interval [0,T]
        sqrt_y = CIR_test.variable_ode_step_test(sqrt_y, T, root_T,
                                                 max_no_crude_subdivisions,
                                                 brownian_increment,
                                                 brownian_area,
                                                 brownian_sign);

        // Record the number of steps used in the simulation
        average_no_of_steps_test = average_no_of_steps_test \
                                 + (double)CIR_test.step_counter;

        // Reset the numerical solution
        sqrt_y = sqrt_y0;
        CIR_test.step_counter = 0;
    }

//    // Run numerical method with a fixed step size
//    for (int i=0; i<no_of_paths_for_test; ++i) {
//        for (int j=1; j<=no_of_steps; ++j) {
//
//            // Generate information about Brownian path
//            brownian_increment = big_increment_distribution(generator);
//            //brownian_area = big_area_distribution(generator);
//
//            //if (rademacher_distribution(generator)){
//            //    brownian_sign = -brownian_sign;
//            //}
//
//
//            // Propagate the numerical solution over the course increment
//            sqrt_y = CIR_test.implicit_euler(sqrt_y, step_size, brownian_increment);
//            //sqrt_y = CIR_test.zigzag_ode_step(sqrt_y, step_size, sqrt_step_size, brownian_increment, brownian_area,
//            //                                     brownian_sign);
//
//        }
//         // Reset the numerical solution
//        sqrt_y = sqrt_y0;
//    }

    // End the speed test
    auto finish = std::chrono::high_resolution_clock::now();

    // Obtain the time taken by the speed test
    std::chrono::duration<double> elapsed = finish - start;

    average_no_of_steps_test = average_no_of_steps_test / (double(no_of_paths_for_test));

    // Display the results in a text file
    ofstream myfile;
    myfile.open ("cir_simulation.txt");

    myfile << std::fixed << std::setprecision(15)
           << "Average number of steps used by the ODE method: \t"
           << average_no_of_steps << "\n";

    myfile << std::fixed << std::setprecision(2)
           << "L2 error at time T = " << T << " for the ODE method: \t\t"
           << std::setprecision(15) << end_point_error << "\n";

    myfile << std::fixed << std::setprecision(2)
           << "Call option error at time T = " << T
           << " for the ODE method: \t"  << std::setprecision(15)
           << abs(crude_call_price - fine_call_price) << "\n\n";

    myfile << "Number of steps used by the drift-implicit Euler method: " << "\t\t"
           << no_of_steps << "\n";

    myfile << std::fixed << std::setprecision(2)
           << "L2 error at time T = " << T << " for the drift-implicit Euler method: \t\t"
           << std::setprecision(15) << end_point_error_2 << "\n";

    myfile << std::fixed << std::setprecision(2)
           << "Call option error at time T = " << T
           << " for the drift-implicit Euler method: "
           << std::setprecision(15) << call_option_error_2 << "\n\n";

    myfile << std::fixed << std::setprecision(2) \
           << "Call option value computed at time T = " << T << ": \t" \
           << std::setprecision(15) << fine_call_price << "\n\n";

    myfile << std::fixed << std::setprecision(10) \
           << "Time taken in the speed test: " << "\t\t\t" << elapsed.count() << "\n\n";

    myfile << std::fixed << std::setprecision(15)
           << "Average number of steps used in the speed test: "
           << average_no_of_steps_test << "\n\n";

    myfile << "Example sample path" << "\n\n" ;

    myfile << "t" << "\t\t" << "y_t" << "\n";

    for (auto i = CIR_method.sample_path.begin();
              i != CIR_method.sample_path.end(); ++i){

        myfile << std::fixed << std::setprecision(10)
               << (*i).first << "\t" << (*i).second << "\n";
    }

    myfile.close();

    return 0;
}
