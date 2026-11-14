//
//  Matrix.m
//  MetalNeuralNetworkNumber
//
//  Created by machinateur on 08.11.26.
//

#import "Matrix.h"
#import "sigmoid.h"

NS_ASSUME_NONNULL_BEGIN

@implementation Matrix

- (id) initWithData: (float*)data
             height: (size_t)h
{
    return [self initWithData:data height:h width:1];
}

- (id) initWithData: (float*)data
             height: (size_t)h
              width: (size_t)w
{
    return [self initWithData:data height:h width:w stride:w];
}

- (id) initWithData: (float*)data
             height: (size_t)h
              width: (size_t)w
             stride: (size_t)s
{
    id obj = [super init];
    
    _rows = h;
    _cols = w;
    _strd = s;
    // Just reference the data.
    _data = data;
    
    return obj;
}

- (id) initWithSize: (size_t)h
{
    return [self initWithSize:h width:h];
}

- (id) initWithSize: (size_t)h
              width: (size_t)w
{
    return [self initWithSize:h width:w stride:w];
}

- (id) initWithSize: (size_t)h
              width: (size_t)w
             stride: (size_t)s
{
    id obj = [super init];
    
    _rows = h;
    _cols = w;
    _strd = s;
    _data = malloc(_rows * _cols * sizeof(_data));
    
    assert(_data != NULL);
    
    return obj;
}

- (id) initWithMatrix:(Matrix *)param
{
    id obj = [self initWithSize:param.rows width:param.cols stride:param.strd];
    
    [obj copy:param];
    
    return obj;
}

// Debug output.
- (void) echo: (NSString *)label
{
    NSLog(@"%@: [", label);
    
    for (size_t i = 0; i < _rows; i++)
    {
        for (size_t j = 0; j < _cols; j++)
        {
            printf("%f ", _data[i * _strd + j]);
        }
        
        printf("\n");
    }
    
    NSLog(@"]");
}

// Fill the entire matrix with a value.
- (void) fill: (float)value
{
    for (size_t i = 0; i < _rows; i++)
    {
        for (size_t j = 0; j < _cols; j++)
        {
            MAT_AT(self, i, j) = value;
        }
    }
}

// Generate a random float value between 1 and 0.
+ (float)rand_float
{
    return (float)rand() / RAND_MAX;
}

// Randomize the matrix between 1.0 and 0.0.
- (void) rand
{
    return [self rand:1.0f];
}

// Randomize the matrix between max and 0.0.
- (void) rand: (float)max
{
    return [self rand:max min:0.0f];
}

// Randomize the matrix with values between max and min.
- (void) rand: (float)max
          min: (float)min
{
    for (size_t i = 0; i < _rows; i++)
    {
        for (size_t j = 0; j < _cols; j++)
        {
            MAT_AT(self, i, j) = [Matrix rand_float] * (max - min) + min;
        }
    }
}

+ (void)sigmoid:(Matrix *)dst
{
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = sigmoidf(MAT_AT(dst, i, j));
        }
    }
}

// Instance method for convenience.
- (void)sigmoid
{
    [Matrix sigmoid:self];
}

// Slice one row from the matrix.
- (Matrix *) row: (size_t)row
{
    return [[Matrix alloc] initWithData:&MAT_AT(self, row, 0) height:1 width:_cols stride:_strd];
}

// Calculate the matrix dot product of a and b into dst.
+ (void) dot: (Matrix *)dst
           a: (Matrix *)a
           b: (Matrix *)b
{
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    assert(a.cols == b.rows);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = 0.0f;
            for (size_t k = 0; k < a.cols; k++)
            {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

// Class method for convenience.
+ (void) dot: (Matrix *)dst
       param: (Matrix *)param
{
    return [Matrix dot:dst a:dst b:param];
}

// Instance method for convenience.
- (void) mul: (Matrix *)param
{
    return [Matrix dot:self param:param];
}

// Calculate the matrix sum of a and b into dst.
+ (void) sum: (Matrix *)dst
           a: (Matrix *)a
           b: (Matrix *)b
{
    assert(dst.rows == a.rows);
    assert(dst.cols == a.cols);
    assert(dst.rows == b.rows);
    assert(dst.cols == b.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(a, i, j) + MAT_AT(b, i, j);
        }
    }
}

// Class method for convenience.
+ (void) sum: (Matrix *)dst
       param: (Matrix *)param
{
    return [Matrix sum:dst a:dst b:param];
}

// Instance method for convenience.
- (void) add: (Matrix *)param
{
    return [Matrix sum:self param:param];
}

// Copy the given matrix to self.
+ (void) copy: (Matrix *)dst
          src: (Matrix *)src
{
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

// Class method for convenience.
- (void) copy: (Matrix *)src
{
    [Matrix copy:self src:src];
}

@end

NS_ASSUME_NONNULL_END
