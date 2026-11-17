//
//  main.m
//  MetalNeuralNetworkNumber
//
//  Created by machinateur on 08.11.26.
//

#import <Foundation/Foundation.h>
//#import <Metal/Metal.h>
#import "Matrix.h"
#import "NeuralNetwork.h"
#import "NeuralNetworkArchitecture.h"
#import "sigmoid.h"

const float train_set[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        size_t layer[] = {2,2,1};
        NeuralNetworkArchitecture *arch = [[NeuralNetworkArchitecture alloc] initWithLayer:layer depth:(ARRAY_LEN(layer) - 1)];
        NeuralNetwork             *net  = [[NeuralNetwork alloc] initWithArchitecture:arch]; // xor
        
        Matrix *train_input  = [[Matrix alloc] initWithData:(float*)train_set   height:4 width:2 stride:3];
        Matrix *train_output = [[Matrix alloc] initWithData:(float*)train_set+2 height:4 width:1 stride:3];
        
        [train_input  echo:@"SET IN"];
        [train_output echo:@"SET OUT"];
        
        srand((int)time(NULL));

        [net rand];
        [net echo:@"rand nn"];

        NSLog(@"MSE before training = %f", [NeuralNetwork mse:net input:train_input output:train_output]);

        [NeuralNetwork gd:net rate:5e-1 epsilon:/*1e-1*/0 input:train_input output:train_output iterations:(10 * 1000)];

        NSLog(@"MSE  after training = %f", [NeuralNetwork mse:net input:train_input output:train_output]);

        [net echo:@"xor"];

        for (size_t i = 0; i < 2; i++)
        {
            for (size_t j = 0; j < 2; j++)
            {
                MAT_AT(NN_INPUT(net), 0, 0) = i;
                MAT_AT(NN_INPUT(net), 0, 1) = j;

                [net forward];

                printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NN_OUTPUT(net), 0, 0));
            }
        }
    }
    return EXIT_SUCCESS;
}
