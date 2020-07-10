# breit_hamil

Code for running Monte-Carlo integrations using the mcint package for matrix elements of the Breit Hamiltonian in the Laguerre Orbital basis.
[A. E. McCoy and M. A. Caprio, J. Math. Phys. 57, (2016).]

Parameters for the matrix element quantum numbers and the tunable monte carlo parameters can be set in main.py

Importance sampling using gamma function PDFs has been implemented, with the gamma pdf given by:
![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/e17d7cc2f7f0e724776f777dc6552b261fea46fe)
