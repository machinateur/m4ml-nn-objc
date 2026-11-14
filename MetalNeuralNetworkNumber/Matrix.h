//
//  Matrix.h
//  MetalNeuralNetworkNumber
//
//  Created by machinateur on 08.11.26.
//

#import <Foundation/Foundation.h>

// TODO: Convert MACRO uses to instance methods.
// Macro to access raw data by column count only.
#define COL_AT(m, i, j) (m).data[(i) * (m).cols + (j)]
// Macro to access raw data by stride instead of column count.
#define MAT_AT(m, i, j) (m).data[(i) * (m).strd + (j)]

NS_ASSUME_NONNULL_BEGIN

// Basic matrix data layer
// (can also be used as vector when w=1).
@interface Matrix : NSObject

@property (readonly) size_t rows;
@property (readonly) size_t cols;
@property (readonly) size_t strd;
@property float *data;

// Init by data with size.
- (id) initWithData: (float*)data
             height:(size_t)h;
- (id) initWithData: (float*)data
             height:(size_t)h
              width:(size_t)w;
- (id) initWithData: (float*)data
             height:(size_t)h
              width:(size_t)w
             stride:(size_t)s;

// Init by size, allocate data as needed.
- (id) initWithSize: (size_t)h;
- (id) initWithSize: (size_t)h
              width: (size_t)w;
- (id) initWithSize: (size_t)h
              width: (size_t)w
             stride: (size_t)s;

- (id) initWithMatrix: (Matrix *)param;

// Print the matrix data.
- (void) echo: (NSString *)label;

// Fill the matrix will one value.
- (void) fill: (float)value;

+ (float) rand_float;

// Fill the matrix with random values.
- (void) rand;
- (void) rand: (float)max;
- (void) rand: (float)max
          min: (float)min;

+ (void) sigmoid: (Matrix *)dst;
- (void) sigmoid;

- (Matrix *) row: (size_t)row;

// Dot product.
+ (void) dot: (Matrix *)dst
           a: (Matrix *)a
           b: (Matrix *)b;
+ (void) dot: (Matrix *)dst
       param: (Matrix *)param;
- (void) mul: (Matrix *)param;

// Sum utils.
+ (void) sum: (Matrix *)dst
           a: (Matrix *)a
           b: (Matrix *)b;
+ (void) sum: (Matrix *)dst
       param: (Matrix *)param;
- (void) add: (Matrix *)param;

// Copy utils.
+ (void) copy: (Matrix *)dst
          src: (Matrix *)src;
- (void) copy: (Matrix *)src;

@end

NS_ASSUME_NONNULL_END
