//
//  NeuralNetwork.m
//  MetalNeuralNetworkNumber
//
//  Created by machinateur on 14.11.26.
//

#import "NeuralNetwork.h"

NS_ASSUME_NONNULL_BEGIN

@implementation NeuralNetwork

- (id)initWithArchitecture: (NeuralNetworkArchitecture *)arch {
    id obj = [self init];
    
    _arch = arch;
    
    _data       = malloc((2 * NN_DEPTH(self)) * sizeof(_data));
    _activation = malloc((1 + NN_DEPTH(self)) * sizeof(_activation));

    NN_INPUT(self) = [_arch createLayerInput];
    for (size_t l = 0; l < NN_DEPTH(self); l++) {
        NN_LAYER(self, l) = [_arch createLayer:l];

        NN_LAYER_WEIGHT(self, l) = [_arch createLayerWeight:l];
        NN_LAYER_BIAS(self, l) = [_arch createLayerBias:l];

        // Same as with [self rand].
        [NN_LAYER_WEIGHT(self, l) rand];
        [NN_LAYER_BIAS(self, l) rand];
    }

    return obj;
}

- (id)initWithNeuralNetwork: (NeuralNetwork *)network {
    return [self initWithArchitecture:network.arch];
}

- (void)echo: (NSString *)label {
    NSLog(@"=== %@ ===", label);

    for (size_t i = 0; i < NN_DEPTH(self); i++)
    {
        [NN_LAYER_WEIGHT(self, i) echo:[NSString stringWithFormat:@"Layer weight %lu", i]];
        [NN_LAYER_BIAS(self, i) echo:[NSString stringWithFormat:@"Layer bias   %lu", i]];
    }
}

// Fill the entire matrix with a value.
- (void) fill: (float)value
         flag: (int8_t)flag
{
    assert(0 != flag);

    for (size_t l = 0; l < NN_DEPTH(self); l++)
    {
        if (flag & NN_FILL_W) {
            [NN_LAYER_WEIGHT(self, l) fill:value];
        }
        if (flag & NN_FILL_B) {
            [NN_LAYER_BIAS(self, l) fill:value];
        }
        if (flag & NN_FILL_A) {
            [NN_LAYER(self, l - 1) fill:value];
        }
    }
}

- (void) fill: (float) value
{
    [self fill:value flag:NN_FILL_WB];
}

// Fill the matrix with random values.
- (void) rand
{
    [self rand:1.0f];
}

- (void) rand: (float)max
{
    [self rand:max min:0.0f];
}
- (void) rand: (float)max
          min: (float)min
{
    for (size_t l = 0; l < NN_DEPTH(self); l++)
    {
        [NN_LAYER_WEIGHT(self, l) rand:max min:min];
        [NN_LAYER_BIAS(self, l)   rand:max min:min];
    }
}

// Copy the input param into the input activation layer.
- (void) prompt: (Matrix *)param
{
    [NN_INPUT(self) copy:param];
}

// Propagate signals through the network.
- (void) forward
{
    for (size_t l = 0; l < NN_DEPTH(self); l++)
    {
        [Matrix dot:NN_LAYER(self, l) a:NN_LAYER(self, (l - 1)) b:NN_LAYER_WEIGHT(self, l)];
        [Matrix sum:NN_LAYER(self, l) param:NN_LAYER_BIAS(self, l)];
        [Matrix sigmoid:NN_LAYER(self, l)];
    }
}

// Copy the input param into, propagate though
// and return copy of the output activation layer in one go.
- (Matrix *) resolve: (Matrix *)param
{
    [self prompt:param];
    [self forward];
    
    return [[Matrix alloc] initWithMatrix:NN_OUTPUT(self)];
}

+ (float) mse:(NeuralNetwork *)net
        input:(Matrix *)input
       output:(Matrix *)output
{
    assert(input.rows == output.rows);
    assert(input.cols == NN_INPUT(net).cols);
    assert(output.cols == NN_OUTPUT(net).cols);

    float result = 0.0f,
          diff;

    for (size_t i = 0; i < input.rows; i++) {
        Matrix *row_input = [input row:i];
        Matrix *row_output = [output row:i];

        [net prompt:row_input];
        [net forward];

        for (size_t j = 0; j < output.cols; j++) {
            diff = MAT_AT(NN_OUTPUT(net), 0, j) - MAT_AT(row_output, 0, j);

            result += diff * diff;
        }
    }
    
    return result / input.rows;
}

+ (void) fd: (NeuralNetwork *)net
       diff: (NeuralNetwork *)diff
    epsilon: (float)epsilon
      input: (Matrix *)input
     output: (Matrix *)output
{
    assert(NN_DEPTH(net) == NN_DEPTH(diff));
    
    float mse = [NeuralNetwork mse:net input:input output:output],
          saved,
          new_mse;
    
    // Iterate all layers of the raw network data.
    for (size_t l = 0; l < 2 * NN_DEPTH(net); l++)
    {
        // Iterate rows of weight or bias.
        for (size_t i = 0; i < NN_LAYER_DATA(net, l).rows; i++)
        {
            // Iterate cols of weight or bias.
            for (size_t j = 0; j < NN_LAYER_DATA(net, l).cols; j++)
            {
                // Save and restore the previous value.
                saved = MAT_AT(NN_LAYER_DATA(net, l), i, j);
                {
                    // Calculate the approximated derivative: `L(w) = lim(eps -> 0)[ (mse(w + eps) - mse(w)) / eps ]`.
                    MAT_AT(NN_LAYER_DATA(net, l), i, j) += epsilon;
                    new_mse = [NeuralNetwork mse:net input:input output:output];
                    
                    MAT_AT(NN_LAYER_DATA(diff, l), i, j) = (new_mse - mse) / epsilon;
                }
                MAT_AT(NN_LAYER_DATA(net, l), i, j)  = saved;
            }
        }
    }
}

+ (void) bp: (NeuralNetwork *)net
       diff: (NeuralNetwork *)diff
      input: (Matrix *) input
     output: (Matrix *) output
{
    assert(NN_DEPTH(net) == NN_DEPTH(diff));

    size_t sampleSize = input.rows;
    float dC; // derived cost

    [diff fill:0.0f flag:NN_FILL_ALL];

    // Iterate each sample from input and output.
    for (size_t i = 0; i < sampleSize; i++) {
        [net resolve:[input row:i]];

        // Reset activation of diff.
        [diff fill:0.0f flag:NN_FILL_A];

        // Initialize the correct value for dC.
        for (size_t j = 0; j < output.cols; j++)
        {
            MAT_AT(NN_OUTPUT(diff), 0, j) = 2 * (MAT_AT(NN_OUTPUT(net), 0, j) - MAT_AT(output, i, j))
                                            * MAT_AT(NN_OUTPUT(net), 0, j) * (1 - MAT_AT(NN_OUTPUT(net), 0, j));
        }

        // Iterate network layer data (-1, because accessor will handle the offset).
        // TODO: Note: View the loop from the same perspective, but just moving backwards.
        for (size_t l = NN_DEPTH(net); 0 < l; l--)
        {
            // Iterate cols of activation for the current layer.
            for (size_t j = 0; j < NN_LAYER_SIZE(net, l - 1); j++)
            {
                // This represents the derivative of the prior layer activation (i.e. from next layer).
                dC = MAT_AT(NN_LAYER(diff, l - 1), 0, j);

                // Accumulate the value for bias.
                MAT_AT(NN_LAYER_BIAS(diff, l - 1), 0, j) += dC;

                // Iterate cols of activation for the next layer.
                for (size_t k = 0; k < NN_LAYER_SIZE(net, l - 2); k++)
                {
                    // Accumulate the values for weight.
                    MAT_AT(NN_LAYER_WEIGHT(diff, l - 1), k, j) += dC
                        * MAT_AT(NN_LAYER(net, l - 2), 0, k);

                    // Accumulate the values for the activation
                    // (this goes to the next iteration for the layer).
                    MAT_AT(NN_LAYER(diff, l - 2), 0, k) += dC
                        * MAT_AT(NN_LAYER_WEIGHT(net, l - 1), k, j)
                        * MAT_AT(NN_LAYER(net, l - 2), 0, k)
                        * (1 - MAT_AT(NN_LAYER(net, l - 2), 0, k));
                }
            }
        }
    }

    // Iterate all layers of the raw network data.
    for (size_t l = 0; l < 2 * NN_DEPTH(net); l++)
    {
        for (size_t i = 0; i < NN_LAYER_DATA(diff, l).rows; i++)
        {
            for (size_t j = 0; j < NN_LAYER_DATA(diff, l).cols; j++)
            {
                MAT_AT(NN_LAYER_DATA(diff, l), i, j) /= sampleSize;
            }
        }
    }
}

// TODO: Investigate adaptive rate per data point.
+ (void) gd: (NeuralNetwork *)net
       rate: (float)rate
    epsilon: (float)epsilon
      input: (Matrix *)input
     output: (Matrix *)output
 iterations: (size_t)iterations
{
    NeuralNetwork *diff = [[NeuralNetwork alloc] initWithNeuralNetwork:net];

    // Iterate for N times through the whole GD process.
    for (size_t n = 0; n < iterations; n++)
    {
        if (0 != epsilon) {
            [NeuralNetwork fd:net diff:diff epsilon:epsilon input:input output:output];
        } else {
            [NeuralNetwork bp:net diff:diff input:input output:output];
        }

        // Iterate all layers of the raw network data.
        for (size_t l = 0; l < 2 * NN_DEPTH(net); l++)
        {
            // Iterate rows of weight or bias.
            for (size_t i = 0; i < NN_LAYER_DATA(net, l).rows; i++)
            {
                // Iterate cols of weight or bias.
                for (size_t j = 0; j < NN_LAYER_DATA(net, l).cols; j++)
                {
                    MAT_AT(NN_LAYER_DATA(net, l), i, j) -= rate * MAT_AT(NN_LAYER_DATA(diff, l), i, j);
                }
            }
        }
    }
}


@end

NS_ASSUME_NONNULL_END
