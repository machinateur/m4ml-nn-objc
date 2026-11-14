//
//  figmoid.c
//  MetalNeuralNetworkNumber
//
//  Created by machinateur on 09.11.26.
//
//  Source: https://github.com/nayayayay/sigmoid-function
//

#include "sigmoid.h"

#include <math.h>

double sigmoid(double n) {
    return (1 / (1 + pow(EULER_NUMBER, -n)));
}

float sigmoidf(float n) {
    return (1 / (1 + powf(EULER_NUMBER_F, -n)));
}

long double sigmoidl(long double n) {
    return (1 / (1 + powl(EULER_NUMBER_L, -n)));
}
