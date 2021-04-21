# Network Generators
This folder contains the scripts used to generate the networks used for
simulations. These functions are not able to be called from the Beowulf 
library, but are contained here for record keeping.

`network-uniform.py` creates a random network with a uniform degree 
distribution. The minimum degree is 1 and a maximum of 6.

`network-random.py` creates a modified clustered power-law graph. First,
several distinct clustered power-law networks (of differing sizes) are
generated. These networks then randomly have edges drawn between each
of the components. This approach results in a singular component with
a clustered power-law and underlying community structure.
