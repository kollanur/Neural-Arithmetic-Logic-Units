# Neural-Arithmetic-Logic-Units

## Introduction
  The ability to represent and manipulate numerical quantities is apparent in the behaviour of many
  species, from insects to mammals to humans, suggesting that basic quantitative reasoning is a general component
  of intellegince.\
  While neural networks can successfully represent and manipulate numerical quantities given an appropriate learning signal, the behaviour that they learn does not generally
  exhibit systematic generalisation. Specifically, one frequently observes failures when quantities that lie outside the numerical range used during training are encountered at test time, even when 
  the target function is simple. This failure pattern indicates that the learned behaviour is better characterized by memorization than by systematic abstraction.


failure.ipynb file demonstrates the behaviour of various MLPs trained to learn the scalar identity function, which is the most straight forward systematic relationship possible.

## Neural Accumulator (NAC)

  The neural accumulator (NAC), which is a special case of a linear (affine) layer whose transformation matrix 
  W consists just of 1's, 0's, and 1's, that is, its outputs are additions or subtractions (rather than arbitray rescalings) of rows
  in the input vector. This prevents the layer from changing the scale of the representations of the numbers when mapping the input to the output
  , meaning that they are consistent throughout the model, no matter how many operations are chained togehter.
  \
  
## Neural Arithmetic Logic Unit (NALU)

The NALU consists of two NAC cells (the purple cells) interpolated by a learned sigmoidal gate g (the orange cell), such that if the add/subtract subcell’s output value is applied with a weight of 1 (on), the multiply/divide subcell’s is 0 (off) and vice versa. The first NAC (the smaller purple subcell) computes the accumulation vector a, which stores results of the NALU’s addition/subtraction operations; it is computed identically to the original NAC, (i.e., a = Wx). The second NAC (the larger purple subcell) operates in log space and is therefore capable of learning to multiply and divide, storing its results in m:

