Numerical simulation of the Cox-Ingersoll-Ross model (CIR)

This code estimates the discretization error of a new high order method for the CIR process:

dy_t = a(b-y_t) dt + sigma sqrt(y_t) dW_t.


The numerical method and example are outlined in the document cir_presentation.pdf.

The source file cir.cpp only requires headers from the C++ standard library.

The text file cir_simulation.txt displays the output of the code.


License

The content of cir_presentation.pdf is licensed under the Creative Commons Attribution 4.0 International license, and cir.cpp is licensed under the MIT license.