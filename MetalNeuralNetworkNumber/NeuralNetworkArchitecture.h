//
//  NeuralNetworkArchitechture.h
//  MetalNeuralNetworkNumber
//
//  Created by machinateur on 14.11.26.
//

#import <Foundation/Foundation.h>

// Macro to caluclate the array length using only the data types.
#define ARRAY_LEN(arr) (sizeof((arr)) / sizeof((arr)[0]))

NS_ASSUME_NONNULL_BEGIN

@interface NeuralNetworkArchitecture : NSObject

@property (readonly) size_t *layer;
@property (readonly) size_t depth;

- (id) initWithLayer: (size_t *)layer
               depth: (size_t)depth;

- (id) initWithSize: (size_t)size
              depth: (size_t)depth;
- (id) initWithSize: (size_t)size
              depth: (size_t)depth
              input: (size_t)input;

- (Matrix *) createLayerWeight: (size_t)layer;
- (Matrix *) createLayerBias: (size_t)layer;
- (Matrix *) createLayer: (size_t)layer;
- (Matrix *) createLayerInput;

@end

NS_ASSUME_NONNULL_END
