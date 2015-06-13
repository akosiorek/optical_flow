#include "gtest/gtest.h"
#include "common.h"
#include "FourierPadder.h"

class FourierPadderTest : public testing::Test {
public:
	static const uint32_t inputSizeRows = 96;
	static const uint32_t inputSizeCols = 82;
	static const uint32_t filterSizeRows = 15;
	static const uint32_t filterSizeCols = 41;

	void SetUp() {
		padder = std::make_unique<FourierPadder>(inputSizeRows, inputSizeCols,
												 filterSizeRows, filterSizeCols);
	}

	std::unique_ptr<FourierPadder> padder;
};

// Fuck you C++ standard
const uint32_t FourierPadderTest::inputSizeRows;
const uint32_t FourierPadderTest::inputSizeCols;
const uint32_t FourierPadderTest::filterSizeRows;
const uint32_t FourierPadderTest::filterSizeCols;

TEST_F(FourierPadderTest, GetPowerOfTwoTest)
{
	ASSERT_EQ(static_cast<uint32_t>(0),FourierPadder::roundUpPow2(0));
	ASSERT_EQ(static_cast<uint32_t>(1),FourierPadder::roundUpPow2(1));
	ASSERT_EQ(static_cast<uint32_t>(2),FourierPadder::roundUpPow2(2));
	ASSERT_EQ(static_cast<uint32_t>(4),FourierPadder::roundUpPow2(3));
	ASSERT_EQ(static_cast<uint32_t>(4),FourierPadder::roundUpPow2(4));
	ASSERT_EQ(static_cast<uint32_t>(8),FourierPadder::roundUpPow2(5));
	ASSERT_EQ(static_cast<uint32_t>(1024),FourierPadder::roundUpPow2(823));
	ASSERT_EQ(static_cast<uint32_t>(1024),FourierPadder::roundUpPow2(1023));
	ASSERT_EQ(static_cast<uint32_t>(1024),FourierPadder::roundUpPow2(1024));
	ASSERT_EQ(static_cast<uint32_t>(2048),FourierPadder::roundUpPow2(1025));
}


TEST_F(FourierPadderTest, PaddingDenseMemoryTest)
{
	auto input = RealMatrix(FourierPadderTest::inputSizeRows, FourierPadderTest::inputSizeCols);
	input.setZero();

	RealMatrix padded;
	ASSERT_EQ(0, padded.rows());
	ASSERT_EQ(0, padded.cols());

	// Check Size
	int fsR = padder->fourierSizeRows_;
	int fsC = padder->fourierSizeCols_;
	padder->padData(input, padded);

	ASSERT_EQ(fsR, padded.rows());
	ASSERT_EQ(fsC, padded.cols());
}

TEST_F(FourierPadderTest, PadInputDenseTest)
{
	auto input = RealMatrix(FourierPadderTest::inputSizeRows, FourierPadderTest::inputSizeCols);
	input.setZero();
	input(15, 19) = 22;
	input(55, 12) = 3;
	input(3, 23) = 454;
	input(21, 17) = 34;
	input(66, 4) = 54;
	input(76, 1) = 12;

	RealMatrix padded;
	padder->padData(input, padded);

	int fsR = padder->fourierSizeRows_;
	int fsC = padder->fourierSizeCols_;
	ASSERT_EQ(fsR, padded.rows());
	ASSERT_EQ(fsC, padded.cols());

	// Check Values
	ASSERT_EQ(22,	padded(15, 19));
	ASSERT_EQ(3,	padded(55, 12));
	ASSERT_EQ(454,	padded(3, 23));
	ASSERT_EQ(34,	padded(21, 17));
	ASSERT_EQ(54,	padded(66, 4));
	ASSERT_EQ(12,	padded(76, 1));
	padded(15, 19) = 0;
	padded(55, 12) = 0;
	padded(3, 23) = 0;
	padded(21, 17) = 0;
	padded(66, 4) = 0;
	padded(76, 1) = 0;
	ASSERT_TRUE(padded.isZero());
}

TEST_F(FourierPadderTest, PadInputSparseTest)
{
	// This also tests implicility the correct conversion between row/colmajor
	auto input = SparseMatrix(FourierPadderTest::inputSizeRows, FourierPadderTest::inputSizeCols);
	input.insert(15, 19) = 22;
	input.insert(55, 12) = 3;
	input.insert(3, 23) = 454;
	input.insert(21, 17) = 34;
	input.insert(66, 4) = 54;
	input.insert(76, 1) = 12;

	RealMatrix padded;
	padder->padData(input, padded);

	int fsR = padder->fourierSizeRows_;
	int fsC = padder->fourierSizeCols_;
	ASSERT_EQ(fsR, padded.rows());
	ASSERT_EQ(fsC, padded.cols());

	// Check Values
	ASSERT_EQ(22,	padded(15, 19));
	ASSERT_EQ(3,	padded(55, 12));
	ASSERT_EQ(454,	padded(3, 23));
	ASSERT_EQ(34,	padded(21, 17));
	ASSERT_EQ(54,	padded(66, 4));
	ASSERT_EQ(12,	padded(76, 1));
	padded(15, 19) = 0;
	padded(55, 12) = 0;
	padded(3, 23) = 0;
	padded(21, 17) = 0;
	padded(66, 4) = 0;
	padded(76, 1) = 0;
	ASSERT_TRUE(padded.isZero());
}


TEST_F(FourierPadderTest, ExtractDenseOutputTest)
{
	auto paddedOutput = RealMatrix(padder->fourierSizeRows_, padder->fourierSizeCols_);
	paddedOutput.setZero();
	paddedOutput(padder->borderTop_+15,padder->borderLeft_+ 19) = 22;
	paddedOutput(padder->borderTop_+55,padder->borderLeft_+ 12) = 3;
	paddedOutput(padder->borderTop_+3, padder->borderLeft_+23) = 454;
	paddedOutput(padder->borderTop_+21,padder->borderLeft_+ 17) = 34;
	paddedOutput(padder->borderTop_+66,padder->borderLeft_+ 4) = 54;
	paddedOutput(padder->borderTop_+76,padder->borderLeft_+ 1) = 12;

	RealMatrix extracted;
	padder->extractDenseOutput(paddedOutput, extracted);

	// Check size of returned matrix
	ASSERT_EQ(padder->dataRows_,extracted.rows());
	ASSERT_EQ(padder->dataCols_,extracted.cols());

	// Check Values
	ASSERT_EQ(22,	extracted(15, 19));
	ASSERT_EQ(3,	extracted(55, 12));
	ASSERT_EQ(454,	extracted(3, 23));
	ASSERT_EQ(34,	extracted(21, 17));
	ASSERT_EQ(54,	extracted(66, 4));
	ASSERT_EQ(12,	extracted(76, 1));
	extracted(15, 19) = 0;
	extracted(55, 12) = 0;
	extracted(3, 23) = 0;
	extracted(21, 17) = 0;
	extracted(66, 4) = 0;
	extracted(76, 1) = 0;
	ASSERT_TRUE(extracted.isZero());
}

TEST_F(FourierPadderTest, ExtractSparseOutputTest)
{
	auto paddedOutput = RealMatrix(padder->fourierSizeRows_, padder->fourierSizeCols_);
	paddedOutput.setZero();
	paddedOutput(padder->borderTop_+15,padder->borderLeft_+ 19) = 22;
	paddedOutput(padder->borderTop_+55,padder->borderLeft_+ 12) = 3;
	paddedOutput(padder->borderTop_+3, padder->borderLeft_+23) = 454;
	paddedOutput(padder->borderTop_+21,padder->borderLeft_+ 17) = 34;
	paddedOutput(padder->borderTop_+66,padder->borderLeft_+ 4) = 54;
	paddedOutput(padder->borderTop_+76,padder->borderLeft_+ 1) = 12;

	SparseMatrix extracted;
	padder->extractSparseOutput(paddedOutput, extracted);

	// Check size of returned matrix
	ASSERT_EQ(padder->dataRows_, static_cast<std::size_t>(extracted.rows()));
	ASSERT_EQ(padder->dataCols_, static_cast<std::size_t>(extracted.cols()));
	// Verfiy Sparsity
	ASSERT_EQ(6, extracted.nonZeros());

	// Check output for correctness
	ASSERT_EQ(22,	extracted.coeffRef(15, 19));
	ASSERT_EQ(3,	extracted.coeffRef(55, 12));
	ASSERT_EQ(454,	extracted.coeffRef(3, 23));
	ASSERT_EQ(34,	extracted.coeffRef(21, 17));
	ASSERT_EQ(54,	extracted.coeffRef(66, 4));
	ASSERT_EQ(12,	extracted.coeffRef(76, 1));

	// Clean and Check again
	//obviously not the smartest way for a sparse matrix
	//just for verbosity
	extracted.coeffRef(15, 19) = 0;
	extracted.coeffRef(55, 12) = 0;
	extracted.coeffRef(3, 23) = 0;
	extracted.coeffRef(21, 17) = 0;
	extracted.coeffRef(66, 4) = 0;
	extracted.coeffRef(76, 1) = 0;
	extracted.prune(0.1,0.001);
	ASSERT_EQ(extracted.nonZeros(), 0);
}