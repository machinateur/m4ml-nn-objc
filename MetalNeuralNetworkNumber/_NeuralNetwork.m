//
//  NeuralNetwork.m
//  MetalNeuralNetworkNumber
//
//  Created by machinateur on 08.11.26.
//

#ifdef __USE_OLD_NN

#import "NeuralNetwork.h"
#import "sigmoid.m"

NS_ASSUME_NONNULL_BEGIN

@implementation NeuralNetwork
{
    // Hidden layer 1
    Matrix *w01; // weight
    Matrix *b01; // bias
    Matrix *a01; // activation
    
    // Hidden layer 2
    Matrix *w02; // weight
    Matrix *b02; // bias
    Matrix *a02; // activation
}

- (id) init
{
    id obj = [super init];
    
    // Input layer = 2x1
    _input = [[Matrix alloc] initWithSize:1 width:2];
    //_input = [[Matrix alloc] initWithSize:2 width:1];
    
    // Hidden layer 1 = 2x1, 2 connections
    w01 = [[Matrix alloc] initWithSize:2];
    b01 = [[Matrix alloc] initWithSize:1 width:2];
    a01 = [[Matrix alloc] initWithSize:1 width:2];
    //b01 = [[Matrix alloc] initWithSize:2 width:1];
    //a01 = [[Matrix alloc] initWithSize:2 width:1];
    
    // Hidden layer 2 = 1x1, 2 connections
    w02 = [[Matrix alloc] initWithSize:2 width:1];
    b02 = [[Matrix alloc] initWithSize:1];
    a02 = [[Matrix alloc] initWithSize:1];
    
    [w01 rand];
    [b01 rand];
    [w02 rand];
    [b02 rand];
    
    // Output layer = 1x1
    _output = [[Matrix alloc] initWithSize:1];
    
    return obj;
}

// Debug output.
- (void)echo: (NSString *)label {
    NSLog(@"=== %@ ===", label);
    
    [w01 echo:@"w01"];
    [b01 echo:@"b01"];
    [w02 echo:@"w02"];
    [b02 echo:@"b02"];
}

// Sigmoid function on a matrix - activation function.
+ (void)activate:(nonnull Matrix *)matrix
{
    [matrix sigmoid];
}

// Copy the provided input.
- (void) prompt: (Matrix *)param
{
    [_input copy:param];
}

// Main forward process of the network.
- (void) forward
{
    [Matrix dot:a01 a:_input b:w01];
    //[Matrix dot:a01 a:w01 b:_input];
    [Matrix sum:a01 param:b01];
    [NeuralNetwork activate:a01];
    
    [Matrix dot:a02 a:a01 b:w02];
    //[Matrix dot:a02 a:w02 b:a01];
    [Matrix sum:a02 param:b02];
    [NeuralNetwork activate:a02];
    
    // Matrix copy for the output.
    [_output copy:a02];
}

// Instance method for convenience,
- (Matrix *) resolve: (Matrix *)param
{
    // Matrix copy for the input.
    [self prompt:param];
    
    [self forward];
    
    return [[Matrix alloc] initWithMatrix:_output];
}

// Calculate the mean square error value.
+ (float) mse: (NeuralNetwork *)net
        input: (Matrix *)input
       output: (Matrix *)output
{
    assert(input.rows == output.rows);
    assert(input.cols == net.input.cols);
    assert(output.cols == net.output.cols);

    float result = 0.0f;

    for (size_t i = 0; i < input.rows; i++)
    {
        Matrix *row_in = [input row:i];
        Matrix *row_out = [output row:i];

        [net prompt:row_in];
        [net forward];

        for (size_t j = 0; j < output.cols; j++)
        {
            float diff = MAT_AT(net.output, 0, j) - MAT_AT(row_out, 0, j);
            
            result += diff * diff;
        }
    }

    return result / input.rows;
}

+ (void) train: (NeuralNetwork *)net
         input: (Matrix *)input
        output: (Matrix *)output
{
    NSLog(@"MSE  PRE TRAIN: %f\n", [NeuralNetwork mse:net input:input output:output]);
    
    [NeuralNetwork applyGradientDescent:net rate:5e-1 epsilon:1e-1 input:input output:output iterations:10*1000];
    
    NSLog(@"MSE POST TRAIN: %f\n", [NeuralNetwork mse:net input:input output:output]);
}

+ (void) calculateFiniteDiff: (NeuralNetwork *)net
                        diff: (NeuralNetwork *)diff
                     epsilon: (float)epsilon
                       input: (Matrix *)input
                      output: (Matrix *)output
{
    float loss = [NeuralNetwork mse:net input:input output:output];
    float saved;

    for (size_t i = 0; i < net->w01.rows; i++)
    {
        for (size_t j = 0; j < net->w01.cols; j++)
        {
            saved = MAT_AT(net->w01, i, j);
            MAT_AT(net->w01, i, j) += epsilon;
            MAT_AT(diff->w01, i, j) = ([NeuralNetwork mse:net input:input output:output] - loss) / epsilon;
            MAT_AT(net->w01, i, j)  = saved;
        }
    }

    for (size_t i = 0; i < net->b01.rows; i++)
    {
        for (size_t j = 0; j < net->b01.cols; j++)
        {
            saved = MAT_AT(net->b01, i, j);
            MAT_AT(net->b01, i, j) += epsilon;
            MAT_AT(diff->b01, i, j) = ([NeuralNetwork mse:net input:input output:output] - loss) / epsilon;
            MAT_AT(net->b01, i, j)  = saved;
        }
    }

    for (size_t i = 0; i < net->w02.rows; i++)
    {
        for (size_t j = 0; j < net->w02.cols; j++)
        {
            saved = MAT_AT(net->w02, i, j);
            MAT_AT(net->w02, i, j) += epsilon;
            MAT_AT(diff->w02, i, j) = ([NeuralNetwork mse:net input:input output:output] - loss) / epsilon;
            MAT_AT(net->w02, i, j)  = saved;
        }
    }

    for (size_t i = 0; i < net->b02.rows; i++)
    {
        for (size_t j = 0; j < net->b02.cols; j++)
        {
            saved = MAT_AT(net->b02, i, j);
            MAT_AT(net->b02, i, j) += epsilon;
            MAT_AT(diff->b02, i, j) = ([NeuralNetwork mse:net input:input output:output] - loss) / epsilon;
            MAT_AT(net->b02, i, j)  = saved;
        }
    }
}

+ (void) applyGradientDescent: (NeuralNetwork *)net
                         rate: (float)rate
                      epsilon: (float)epsilon
                        input: (Matrix *)input
                       output: (Matrix *)output
                   iterations: (size_t)iterations
{
    NeuralNetwork *diff = [[NeuralNetwork alloc] init];

    for (size_t i = 0; i < iterations; i++)
    {
        [NeuralNetwork calculateFiniteDiff:net diff:diff epsilon:epsilon input:input output:output];

        for (size_t j = 0; j < net->w01.rows; j++)
        {
            for (size_t k = 0; k < net->w01.cols; k++)
            {
                MAT_AT(net->w01, j, k) -= rate * MAT_AT(diff->w01, j, k);
            }
        }

        for (size_t j = 0; j < net->b01.rows; j++)
        {
            for (size_t k = 0; k < net->b01.cols; k++)
            {
                MAT_AT(net->b01, j, k) -= rate * MAT_AT(diff->b01, j, k);
            }
        }

        for (size_t j = 0; j < net->w02.rows; j++)
        {
            for (size_t k = 0; k < net->w02.cols; k++)
            {
                MAT_AT(net->w02, j, k) -= rate * MAT_AT(diff->w02, j, k);
            }
        }

        for (size_t j = 0; j < net->b02.rows; j++)
        {
            for (size_t k = 0; k < net->b02.cols; k++)
            {
                MAT_AT(net->b02, j, k) -= rate * MAT_AT(diff->b02, j, k);
            }
        }
    }
}

// Instance method for convenience.
- (void) diff: (NeuralNetwork *)net
      epsilon: (float)epsilon
        input: (Matrix *)input
       output: (Matrix *)output
{
    [NeuralNetwork calculateFiniteDiff:self diff:net epsilon:epsilon input:input output:output];
}

@end

NS_ASSUME_NONNULL_END

#endif // __USE_OLD_NN
