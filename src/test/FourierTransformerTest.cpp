//
// Created by Adam Kosiorek on 23.05.15.
//

#include "gtest/gtest.h"

#include "types.h"
#include "utils.h"
#include "IFourierTransformer.cpp"
#include "FourierTransformerFFTW.cpp"
#include "FourierPadder.h"

class FourierTransformerTest : public testing::Test {

    void SetUp() {
        ftfftw = std::make_unique<FourierTransformerFFTW>(8,8);
    }

public:
    std::unique_ptr<FourierTransformerFFTW> ftfftw;
};

TEST_F(FourierTransformerTest, ConstructorTest)
{
	ASSERT_EQ(8,ftfftw->rows_);
	ASSERT_EQ(8,ftfftw->cols_);
	ASSERT_EQ(8/2+1,ftfftw->colsHS_);
}

TEST_F(FourierTransformerTest, ForwardTestMatrixSizes)
{
	RealMatrix 	inT(ftfftw->rows_,ftfftw->rows_),
				inF(7,9);
	ComplexMatrix 	outT(ftfftw->rows_,ftfftw->colsHS_),
				  	outF(3,3);

	ASSERT_THROW(ftfftw->forward(inT,outF), std::invalid_argument);
	ASSERT_THROW(ftfftw->forward(inF,outT), std::invalid_argument);
	ASSERT_THROW(ftfftw->forward(inF,outF), std::invalid_argument);
	ASSERT_NO_THROW(ftfftw->forward(inT,outT));
}

TEST_F(FourierTransformerTest, BackwardTestMatrixSizes)
{
	ComplexMatrix 	inT(ftfftw->rows_,ftfftw->colsHS_),
				  	inF(3,3);
	RealMatrix 	outT(ftfftw->rows_,ftfftw->rows_),
				outF(7,9);

	ASSERT_THROW(ftfftw->backward(inT,outF), std::invalid_argument);
	ASSERT_THROW(ftfftw->backward(inF,outT), std::invalid_argument);
	ASSERT_THROW(ftfftw->backward(inF,outF), std::invalid_argument);
	ASSERT_NO_THROW(ftfftw->backward(inT,outT));
}

TEST_F(FourierTransformerTest, FowardInputConsistency)
{
	RealMatrix 		rm(ftfftw->rows_,ftfftw->rows_);

	rm << 	17	,24	,1	,8	,15	,0,	0,	0,
			23	,5	,7	,14	,16	,0,	0,	0,
			4	,6	,13	,20	,22	,0,	0,	0,
			10	,12	,19	,21	,3	,0,	0,	0,
			11	,18	,25	,2	,9	,0,	0,	0,
			0	,0	,0	,0	,0	,0,	0,	0,
			0	,0	,0	,0	,0	,0,	0,	0,
			0	,0	,0	,0	,0	,0,	0,	0;

	RealMatrix		rm_copy = rm;

	ComplexMatrix	cm(ftfftw->rows_,ftfftw->colsHS_);

	ftfftw->forward(rm,cm);
	ASSERT_EQ(true,rm.isApprox(rm_copy,0.000000001));
}

TEST_F(FourierTransformerTest, ForwardBackwardScaling)
{
	RealMatrix 		rm(ftfftw->rows_,ftfftw->rows_);

	rm << 	17	,24	,1	,8	,15	,0,	0,	0,
			23	,5	,7	,14	,16	,0,	0,	0,
			4	,6	,13	,20	,22	,0,	0,	0,
			10	,12	,19	,21	,3	,0,	0,	0,
			11	,18	,25	,2	,9	,0,	0,	0,
			0	,0	,0	,0	,0	,0,	0,	0,
			0	,0	,0	,0	,0	,0,	0,	0,
			0	,0	,0	,0	,0	,0,	0,	0;
	RealMatrix		rm_copy = rm;

	ComplexMatrix	cm(ftfftw->rows_,ftfftw->cols_/2+1);
	cm.setZero();

	ftfftw->forward(rm,cm);
	ftfftw->backward(cm,rm);
	rm = rm/(ftfftw->rows_*ftfftw->cols_);

	ASSERT_EQ(true,rm.isApprox(rm_copy,0.000001));
}

TEST_F(FourierTransformerTest, FwdBwdFilterTest)
{
	RealMatrix	rmData(5,5), rmFilter(3,3);

	rmData << 	17, 24, 1, 8, 15,
			23, 5, 7, 14, 16,
			4, 6, 13, 20, 22,
			10, 12, 19, 21, 3,
			11, 18, 25, 2, 9;

	rmFilter << 8,	1,	6,
				3,	5,	7,
				4,	9,	2;

	RealMatrix		rmDataPadded, rmFilterPadded;
	FourierPadder padder(5, 3);
	padder.padData(rmData,rmDataPadded);
	padder.padFilter(rmFilter,rmFilterPadded);

	ComplexMatrix	cmData(ftfftw->rows_,ftfftw->cols_/2+1);
	ComplexMatrix	cmFilter(ftfftw->rows_,ftfftw->cols_/2+1);
	cmData.setZero(); 	cmFilter.setZero();

	ftfftw->forward(rmDataPadded,cmData);
	ftfftw->forward(rmFilterPadded,cmFilter);

	ComplexMatrix cmResult = cmData.cwiseProduct(cmFilter);

	ftfftw->backward(cmResult,rmDataPadded);
	rmDataPadded = rmDataPadded/(ftfftw->rows_*ftfftw->cols_);
	padder.extractDenseOutput(rmDataPadded,rmData);

	RealMatrix expectedResult(5,5);
	expectedResult << 	220, 441, 346, 276, 231,
						431, 595, 410, 575, 471,
						371, 440, 555, 620, 551,
						301, 585, 600, 765, 421,
						247, 446, 536, 451, 128;

	std::cout << rmData << std::endl;
	ASSERT_EQ(true,rmData.isApprox(expectedResult,0.000001));
}