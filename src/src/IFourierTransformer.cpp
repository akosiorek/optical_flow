//
// Created by Adam Kosiorek on 23.05.15.
//

#include "IFourierTransformer.h"

ComplexMatrix IFourierTransformer::forward(const RealMatrix &src) const {

     ComplexMatrix dst(src.rows(), src.cols());
     forward(src, dst);
     return dst;
}

RealMatrix IFourierTransformer::backward(const ComplexMatrix &src) const {

     RealMatrix dst(src.rows(), src.cols());
     backward(src, dst);
     return dst;
}