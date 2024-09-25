################
Building a model
################

Energy systems in ZEN-garden are represented by a network of nodes and edges. At each node, conversion technologies can be installed. Conversion technologies can take a specific carriers as input and convert it into another carrier. Moreover, storage technologies are available which can store energy at a specific location over time. Nodes are connected by edges. At each edge, transport technologies can be installed to transport carriers between nodes. The nodal carrier import and export availabilities specify how much of one carrier can be imported or exported from the system. Nodal carrier demands can be specified which must be satisfied. Considering all these possibilities allows users to model their system with as much detail as needed to answer their research question. 

All carriers and technologies must be defined in the dataset folder located in ``<Your-ZEN-garden-Repository>/data/``.