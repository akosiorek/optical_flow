#include "gtest/gtest.h"

#include <memory>

#include <Eigen/SparseCore>
#include <Eigen/Core>

#include "utils.h"

#include "../src/FourierPadder.h"


class FourierPadderTest : public testing::Test {
public:

    // similar for odd/odd, odd/even, even/odd combinations of filter and input size
    const uint32_t inputSize = 20;
    const uint32_t filterSize = 10;

    void SetUp() {
        padder = make_unique<FourierPadder>(inputSize, filterSize);
    }

public:
    std::unique_ptr<FourierPadder> padder;
};

TEST_F(FourierPadderTest, GetPowerOfTwoTest)
{
    ASSERT_EQ(padder->getNextPowerOfTwo(0), 0);
    ASSERT_EQ(padder->getNextPowerOfTwo(1), 1);
    ASSERT_EQ(padder->getNextPowerOfTwo(2), 2);
    ASSERT_EQ(padder->getNextPowerOfTwo(3), 4);   
    ASSERT_EQ(padder->getNextPowerOfTwo(4), 4);  
    ASSERT_EQ(padder->getNextPowerOfTwo(5), 8);        
    ASSERT_EQ(padder->getNextPowerOfTwo(857), 1024);
    ASSERT_EQ(padder->getNextPowerOfTwo(1023), 1024);
    ASSERT_EQ(padder->getNextPowerOfTwo(1024), 1024);
    ASSERT_EQ(padder->getNextPowerOfTwo(1025), 2048);
}

// TEST_F(FourierPadderTest, PadInputTest) {

//     auto input = std::make_shared<Eigen::SparseMatrix>(inputSize, inputSize);
//     input->setZero();
//     input->insert(2, 3) = 1;
//     input->insert(4, 5) = 12;

//     std::shared_ptr<Eigen::Matrix<float, paddedSize, paddedSize>> padded = padder.padInput(input);

//     ASSERT_EQ(*padded(2, 3), 1);
//     ASSERT_EQ(*padded(4, 5), 12);
//     *padded(2, 3) = 0;
//     *padded(4, 5) = 0;
//     ASSERT_TRUE(padded->isZero(0));
// }

// TEST_F(FourierPadderTest, PadFilterTest) {

//     auto filter = std::make_shared<Eigen::Matrix<float, filterSize, filterSize>();
//     filter->setZero();
//     input->insert(2, 3) = 1;
//     input->insert(4, 5) = 12;


//     std::shared_ptr<Eigen::Matrix<float, paddedSize, paddedSize>> padded = padder.padFilter(filter);

//     ASSERT_EQ(*padded(2, 3), 1);
//     ASSERT_EQ(*padded(4, 5), 12);
//     *padded(2, 3) = 0;
//     *padded(4, 5) = 0;
//     ASSERT_TRUE(padded->isZero(0));
// }

// TEST_F(FourierPadderTest, ExtractDenseOutputTest) {

//     auto paddedOutput = std::make_shared<Eigen::Matrix<float, paddedSize, paddedSize>();
//     filter->setZero();
//     *filter(15, 19) = 1;
//     *filter(21, 16) = 12;

//     std::shared_ptr<Eigen::Matrix<float, inputSize, inputSize>> extracted = padder.extract(paddedOutput);

//     ASSERT_EQ(*extracted(10, 14), 1);
//     ASSERT_EQ(*extracted(16, 11), 12);
//     *extracted(10, 14) = 0;
//     *extracted(16, 11) = 0;
//     ASSERT_TRUE(extracted->isZero(0));
// }

// TEST_F(FourierPadderTest, ExtractSparseOutputTest) {
//     FourierPadder padder(3, 3);

//     auto denseOutput = std::make_shared<Eigen::Matrix<float, 3, 3>();

//     *filter(1, 1) = 1;
//     *filter(2, 1) = 2;
//     *filter(3, 1) = 3;
//     *filter(1, 2) = 4;
//     *filter(2, 2) = 5;
//     *filter(3, 2) = 6;
//     *filter(1, 3) = 7;
//     *filter(2, 3) = 8;
//     *filter(3, 3) = 9;

//     auto input = std::make_shared<Eigen::SparseMatrix>(3, 3);
//     input->setZero();
//     input->insert(1, 1) = 1; // 1
//     input->insert(2, 3) = 1; // 8
//     input->insert(3, 2) = 1; // 6
//     input->insert(3, 3) = 1; // 9
//     input->insert(1, 3) = 1; // 7
//     input->insert(3, 1) = 1; // 3

//     std::shared_ptr<Eigen::Matrix<float, inputSize, inputSize>> extracted = padder.extractSparse(padded, input);

//     ASSERT_EQ(*extracted(10, 14), 1);
//     ASSERT_EQ(*extracted(16, 11), 12);
//     *extracted(10, 14) = 0;
//     *extracted(16, 11) = 0;
//     ASSERT_TRUE(extracted->isZero(0));
// }