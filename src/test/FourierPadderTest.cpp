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
		padder = make_unique<FourierPadder<inputSize, filterSize> >();
	}

	std::unique_ptr<FourierPadder<inputSize, filterSize> > padder;

	// similar for odd/odd, odd/even, even/odd combinations of filter and input size
	typedef FourierPadder<inputSize, filterSize> FPType;
};

// Fuck you C++ standard
const uint32_t FourierPadderTest::inputSize;
const uint32_t FourierPadderTest::filterSize;

//Test had to be disabled as function is now  evaluate at compile time
TEST_F(FourierPadderTest, GetPowerOfTwoTest)
{
	ASSERT_EQ(Pow2RoundUp<0>::value, 0);
	ASSERT_EQ(Pow2RoundUp<1>::value, 1);
	ASSERT_EQ(Pow2RoundUp<2>::value, 2);
	ASSERT_EQ(Pow2RoundUp<3>::value, 4);
	ASSERT_EQ(Pow2RoundUp<4>::value, 4);
	ASSERT_EQ(Pow2RoundUp<5>::value, 8);
	ASSERT_EQ(Pow2RoundUp<823>::value, 1024);
	ASSERT_EQ(Pow2RoundUp<1023>::value, 1024);
	ASSERT_EQ(Pow2RoundUp<1024>::value, 1024);
	ASSERT_EQ(Pow2RoundUp<1025>::value, 2048);
}

TEST_F(FourierPadderTest, PadInputSparseTest)
{
	auto input = std::make_shared<FPType::InputMatrixSparse >
					(FPType::FilterSize::value, FPType::FilterSize::value);
	input->setZero();
	input->insert(2, 3) = 1;
	input->insert(4, 5) = 12;

	std::shared_ptr<FPType::FourierMatrix > padded = padder->padData(input);

	ASSERT_EQ((*padded)(2, 3), 1);
	ASSERT_EQ((*padded)(4, 5), 12);
	(*padded)(2, 3) = 0;
	(*padded)(4, 5) = 0;
	ASSERT_TRUE(padded->isZero());
}

TEST_F(FourierPadderTest, PadInputDenseTest)
{
	auto input = std::make_shared<FPType::InputMatrixDense >();
	input->setZero();
	(*input)(15, 19) = 22;
	(*input)(55, 12) = 3;
	(*input)(3, 23) = 454;
	(*input)(21, 17) = 34;
	(*input)(66, 4) = 54;
	(*input)(76, 1) = 12;

	std::shared_ptr<FPType::FourierMatrix > padded = padder->padData(input);

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
	auto paddedOutput = std::make_shared<FPType::FourierMatrix>();
	paddedOutput->setZero();
	(*paddedOutput)(15, 19) = 1;
	(*paddedOutput)(21, 16) = 12;

	std::shared_ptr<FPType::OutputMatrixDense> extracted = padder->extractDenseOutput(paddedOutput);

	// Check size of returned matrix
	ASSERT_EQ(inputSize,(*extracted).rows());
	ASSERT_EQ(inputSize,(*extracted).cols());

	// Check output for correctness
	ASSERT_EQ((*extracted)(15, 19), 1);
	ASSERT_EQ((*extracted)(21, 16), 12);

	// Clean and Check again
	(*extracted)(15, 19) = 0;
	(*extracted)(21, 16) = 0;
	ASSERT_TRUE(extracted->isZero(0));
}

TEST_F(FourierPadderTest, ExtractSparseOutputTest) 
{
	auto paddedOutput = std::make_shared<FPType::FourierMatrix>();
	paddedOutput->setZero();
	(*paddedOutput)(15, 19) = 22;
	(*paddedOutput)(55, 12) = 3;
	(*paddedOutput)(3, 23) = 454;
	(*paddedOutput)(21, 17) = 34;
	(*paddedOutput)(66, 4) = 54;
	(*paddedOutput)(76, 1) = 12;

	std::shared_ptr<FPType::OutputMatrixSparse> extracted = padder->extractSparseOutput(paddedOutput);

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