# Network Generators
This folder contains the scripts used to generate the networks used for
simulations. These functions are not able to be called from the Beowulf 
library, but are contained here for record keeping.

`network-uniform.py`, `network-uniform-1k.py`, `network-uniform-2k.py` create random networks with a uniform degree 
distribution for n=500, n=1000, and n=2000, respectively. The minimum degree is 1 and a maximum of 6.

`network-random.py`, `network-random-1k.py`, `network-random-2k.py` create modified clustered power-law graphs for 
n=500, n=1000, n=2000, respectively. First, several distinct clustered power-law networks (of differing sizes) are
generated. These networks then randomly have edges drawn between each of the components. This approach results in a 
singular component with a clustered power-law and underlying community structure.
