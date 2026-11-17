//
//  NeuralNetworkArchitechture.m
//  MetalNeuralNetworkNumber
//
//  Created by machinateur on 14.11.26.
//

#import "Matrix.h"
#import "NeuralNetworkArchitecture.h"

NS_ASSUME_NONNULL_BEGIN

@implementation NeuralNetworkArchitecture

- (id) initWithLayer: (size_t *)layer
               depth: (size_t)depth
{
    id obj = [self init];

    _layer = layer;
    _depth = depth;

    return obj;
}

- (id) initWithSize: (size_t)size
              depth: (size_t)depth
{
    return [self initWithSize:size depth:depth input:size];
}

- (id) initWithSize: (size_t)size
              depth: (size_t)depth
              input: (size_t)input
{
    size_t *layer = malloc((1 + depth) * sizeof(*_layer));

    layer[0] = input;
    for (size_t i = 0; i < depth; i++) {
        layer[1 + i] = size;
    }

    return [self initWithLayer:layer depth:depth];
}

- (Matrix *) createLayerWeight: (size_t)layer
{
    assert(layer <= _depth);

    return [[Matrix alloc] initWithSize:_layer[layer] width:_layer[1 + layer]];
}

- (Matrix *) createLayerBias: (size_t)layer
{
    assert(layer <= _depth);

    return [[Matrix alloc] initWithSize:1 width:_layer[1 + layer]];
}

- (Matrix *) createLayer: (size_t)layer
{
    assert(layer <= _depth);

    return [[Matrix alloc] initWithSize:1 width:_layer[1 + layer]];
}

- (Matrix *) createLayerInput
{
    return [[Matrix alloc] initWithSize:1 width:_layer[0]];
}

@end

NS_ASSUME_NONNULL_END
