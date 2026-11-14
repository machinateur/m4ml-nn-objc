//
//  NeuralNetwork.h
//  MetalNeuralNetworkNumber
//
//  Created by machinateur on 08.11.26.
//

#ifdef __USE_OLD_NN

#import <Foundation/Foundation.h>
#import "Matrix.h"
#import "NeuralNetworkArchitecture.h"

NS_ASSUME_NONNULL_BEGIN

@interface NeuralNetwork : NSObject

// Input layer
@property (readonly) Matrix *input;

// Hidden layers - hidden in the implementation...

// Output layer
@property (readonly) Matrix *output;

// Print the network data.
- (void) echo: (NSString *)label;

// Activation function at network level
// (the advantage: can be redefined as needed
// while independent of matrix implementation).
+ (void) activate: (Matrix *) matrix;

// Save the input parameter.
- (void) prompt: (Matrix *)param;

// Feed-forward through the layers
// (use input and output property).
- (void) forward;

// Resolve the output for a given param.
- (Matrix *) resolve: (Matrix *)param;

// Calculate the MSE (mean square error) score (aka. loss;
// i.e. sum of all squared differences between expected and actual output).
+ (float) mse: (NeuralNetwork *)net
        input: (Matrix *)input
       output: (Matrix *)output;

// Perform the training runs and MSE comparison.
+ (void) train: (NeuralNetwork *)net
         input: (Matrix *)input
        output: (Matrix *)output;

// Training utils, like fd and gd.
+ (void) calculateFiniteDiff: (NeuralNetwork *)net
                        diff: (NeuralNetwork *)diff
                     epsilon: (float)epsilon
                       input: (Matrix *)input
                      output: (Matrix *)output;
+ (void) applyGradientDescent: (NeuralNetwork *)net
                         rate: (float)rate
                      epsilon: (float)epsilon
                        input: (Matrix *)input
                       output: (Matrix *)output
                   iterations: (size_t)iterations;

// Find the overall differences across the network for the given i/o set.
- (void) diff: (NeuralNetwork *)net
      epsilon: (float)epsilon
        input: (Matrix *)input
       output: (Matrix *)output;

@end

NS_ASSUME_NONNULL_END

#endif // __USE_OLD_NN
