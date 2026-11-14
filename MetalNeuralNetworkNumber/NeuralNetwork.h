//
//  NeuralNetwork.h
//  MetalNeuralNetworkNumber
//
//  Created by machinateur on 14.11.26.
//

#import <Foundation/Foundation.h>
#import "Matrix.h"
#import "NeuralNetworkArchitecture.h"

// TODO: Convert MACRO uses to instance methods.
#define NN_DEPTH(nn)  ((nn).arch.depth)
#define NN_INPUT(nn)  ((nn).activation[0].data)
#define NN_OUTPUT(nn) ((nn).activation[NN_DEPTH(nn)].data)

// TODO: Convert MACRO uses to instance methods.
#define NN_LAYER_SIZE(nn, l)   ((nn).arch.layer[(1+l)])
#define NN_LAYER_DATA(nn, l)   ((nn).data[(l)].data)
#define NN_LAYER_WEIGHT(nn, l) NN_LAYER_DATA(nn, (l*2))
#define NN_LAYER_BIAS(nn, l)   NN_LAYER_DATA(nn, (l*2)+1)
#define NN_LAYER(nn, l)        ((nn).activation[(l)+1].data)

NS_ASSUME_NONNULL_BEGIN

typedef struct {
    Matrix *data;
} MatrixArray;

@interface NeuralNetwork : NSObject

// The structural description of the network.
@property (readonly) NeuralNetworkArchitecture *arch;

// The network layers data; an array of pointers to Matrix.
// Data is always a combination of weights and biases next to each other.
@property (readonly) MatrixArray *data;

// The network layers neuron activation; an array of pointers to Matrix.
@property (readonly) MatrixArray *activation;

- (id) initWithArchitecture: (NeuralNetworkArchitecture *)arch;

- (id) initWithNeuralNetwork: (NeuralNetwork *)network;

// Print the network data.
- (void) echo: (NSString *)label;

- (void) fill: (float)value;

// Fill the matrix with random values.
- (void) rand;
- (void) rand: (float)max;
- (void) rand: (float)max
          min: (float)min;

// Save the input parameter.
- (void) prompt: (Matrix *)param;

// Feed-forward through the layers.
- (void) forward;

// Resolve the output for a given input param.
- (Matrix *) resolve: (Matrix *)param;

// Calculate the MSE (mean square error) score (aka. loss;
// i.e. sum of all squared differences between expected and actual output).
+ (float) mse: (NeuralNetwork *)net
        input: (Matrix *)input
       output: (Matrix *)output;

// Calculate the FD (finite difference) between two neural networks
// (i.e. add epsilon to the network weights and biases, then
// take the difference from new to baseline MSE, divided by epsilon).
+ (void) fd: (NeuralNetwork *)net
       diff: (NeuralNetwork *)diff
    epsilon: (float)epsilon
      input: (Matrix *)input
     output: (Matrix *)output;

// Perform GD (gradient descent) on a neural network to optimize the output
// (i.e. for N interations, adapt the weights and biases towards the GD slope,
// using the derivative approximation function - FD - on the previous result).
+ (void) gd: (NeuralNetwork *)net
       rate: (float)rate
    epsilon: (float)epsilon
      input: (Matrix *)input
     output: (Matrix *)output
 iterations: (size_t)iterations;

@end

NS_ASSUME_NONNULL_END
