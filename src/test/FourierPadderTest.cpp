#include "gtest/gtest.h"

#include <memory>

#include <Eigen/SparseCore>
#include <Eigen/Core>

#include "utils.h"

#include "../src/FourierPadder.h"


class FourierPadderTest : public testing::Test {
public:
	static const uint32_t inputSize = 96;
	static const uint32_t filterSize = 15;

	void SetUp() {
		padder = std::make_unique<FourierPadder>(inputSize, filterSize);
	}

	std::unique_ptr<FourierPadder> padder;
};

// Fuck you C++ standard
const uint32_t FourierPadderTest::inputSize;
const uint32_t FourierPadderTest::filterSize;

TEST_F(FourierPadderTest, GetPowerOfTwoTest)
{
	ASSERT_EQ(0,roundUpPow2(0));
	ASSERT_EQ(1,roundUpPow2(1));
	ASSERT_EQ(2,roundUpPow2(2));
	ASSERT_EQ(4,roundUpPow2(3));
	ASSERT_EQ(4,roundUpPow2(4));
	ASSERT_EQ(8,roundUpPow2(5));
	ASSERT_EQ(1024,roundUpPow2(823));
	ASSERT_EQ(1024,roundUpPow2(1023));
	ASSERT_EQ(1024,roundUpPow2(1024));
	ASSERT_EQ(2048,roundUpPow2(1025));
}

TEST_F(FourierPadderTest, PadInputDenseTest)
{
	auto input = FourierPadder::EBOFMatrix(FourierPadderTest::inputSize, FourierPadderTest::inputSize);
	input.setZero();
	input(15, 19) = 22;
	input(55, 12) = 3;
	input(3, 23) = 454;
	input(21, 17) = 34;
	input(66, 4) = 54;
	input(76, 1) = 12;

	std::shared_ptr<FourierPadder::EBOFMatrix> padded = padder->padData(input);

	// Check Size
	int fs = padder->fourierSize_;
	ASSERT_EQ(fs,(*padded).rows());
	ASSERT_EQ(fs,(*padded).cols());

	// Check Values
	ASSERT_EQ(22,	(*padded)(15, 19));
	ASSERT_EQ(3,	(*padded)(55, 12));
	ASSERT_EQ(454,	(*padded)(3, 23));
	ASSERT_EQ(34,	(*padded)(21, 17));
	ASSERT_EQ(54,	(*padded)(66, 4));
	ASSERT_EQ(12,	(*padded)(76, 1));
	(*padded)(15, 19) = 0;
	(*padded)(55, 12) = 0;
	(*padded)(3, 23) = 0;
	(*padded)(21, 17) = 0;
	(*padded)(66, 4) = 0;
	(*padded)(76, 1) = 0;
	ASSERT_TRUE(padded->isZero());
}

TEST_F(FourierPadderTest, PadInputSparseTest)
{
	// This also tests implicility the correct conversion between row/colmajor
	auto input = Eigen::SparseMatrix<float>(inputSize, inputSize);
	input.insert(15, 19) = 22;
	input.insert(55, 12) = 3;
	input.insert(3, 23) = 454;
	input.insert(21, 17) = 34;
	input.insert(66, 4) = 54;
	input.insert(76, 1) = 12;

	std::shared_ptr<FourierPadder::EBOFMatrix> padded = padder->padData(input);

	// Check Size
	int fs = padder->fourierSize_;
	ASSERT_EQ(fs,(*padded).rows());
	ASSERT_EQ(fs,(*padded).cols());

	// Check Values
	ASSERT_EQ(22,	(*padded)(15, 19));
	ASSERT_EQ(3,	(*padded)(55, 12));
	ASSERT_EQ(454,	(*padded)(3, 23));
	ASSERT_EQ(34,	(*padded)(21, 17));
	ASSERT_EQ(54,	(*padded)(66, 4));
	ASSERT_EQ(12,	(*padded)(76, 1));
	(*padded)(15, 19) = 0;
	(*padded)(55, 12) = 0;
	(*padded)(3, 23) = 0;
	(*padded)(21, 17) = 0;
	(*padded)(66, 4) = 0;
	(*padded)(76, 1) = 0;
	ASSERT_TRUE(padded->isZero());
}


TEST_F(FourierPadderTest, ExtractDenseOutputTest)
{
	auto paddedOutput = FourierPadder::EBOFMatrix(padder->fourierSize_,padder->fourierSize_);
	paddedOutput.setZero();
	paddedOutput(15, 19) = 22;
	paddedOutput(55, 12) = 3;
	paddedOutput(3, 23) = 454;
	paddedOutput(21, 17) = 34;
	paddedOutput(66, 4) = 54;
	paddedOutput(76, 1) = 12;

	std::shared_ptr<FourierPadder::EBOFMatrix> extracted = padder->extractDenseOutput(paddedOutput);

	// Check size of returned matrix
	ASSERT_EQ(inputSize,(*extracted).rows());
	ASSERT_EQ(inputSize,(*extracted).cols());

	// Check Values
	ASSERT_EQ(22,	(*extracted)(15, 19));
	ASSERT_EQ(3,	(*extracted)(55, 12));
	ASSERT_EQ(454,	(*extracted)(3, 23));
	ASSERT_EQ(34,	(*extracted)(21, 17));
	ASSERT_EQ(54,	(*extracted)(66, 4));
	ASSERT_EQ(12,	(*extracted)(76, 1));
	(*extracted)(15, 19) = 0;
	(*extracted)(55, 12) = 0;
	(*extracted)(3, 23) = 0;
	(*extracted)(21, 17) = 0;
	(*extracted)(66, 4) = 0;
	(*extracted)(76, 1) = 0;
	ASSERT_TRUE(extracted->isZero());
}

TEST_F(FourierPadderTest, ExtractSparseOutputTest) 
{
	auto paddedOutput = FourierPadder::EBOFMatrix(padder->fourierSize_,padder->fourierSize_);
	paddedOutput.setZero();
	paddedOutput(15, 19) = 22;
	paddedOutput(55, 12) = 3;
	paddedOutput(3, 23) = 454;
	paddedOutput(21, 17) = 34;
	paddedOutput(66, 4) = 54;
	paddedOutput(76, 1) = 12;

	std::shared_ptr<Eigen::SparseMatrix<float> > extracted = padder->extractSparseOutput(paddedOutput);

	// Check size of returned matrix
	ASSERT_EQ(inputSize,extracted->rows());
	ASSERT_EQ(inputSize,extracted->cols());
	// Verfiy Sparsity
	ASSERT_EQ(6,extracted->nonZeros());  

	// Check output for correctness
	ASSERT_EQ(22,	extracted->coeffRef(15, 19));
	ASSERT_EQ(3,	extracted->coeffRef(55, 12));
	ASSERT_EQ(454,	extracted->coeffRef(3, 23));
	ASSERT_EQ(34,	extracted->coeffRef(21, 17));
	ASSERT_EQ(54,	extracted->coeffRef(66, 4));
	ASSERT_EQ(12,	extracted->coeffRef(76, 1));

	// Clean and Check again
	//obviously not the smartest way for a sparse matrix
	//just for verbosity
	extracted->coeffRef(15, 19) = 0;
	extracted->coeffRef(55, 12) = 0;
	extracted->coeffRef(3, 23) = 0;
	extracted->coeffRef(21, 17) = 0;
	extracted->coeffRef(66, 4) = 0;
	extracted->coeffRef(76, 1) = 0;
	extracted->prune(0.1,0.001);
	ASSERT_EQ(extracted->nonZeros(), 0);
}