//
// Created by Adam Kosiorek on 6/13/15.
//

#include "gtest/gtest.h"
#include "TestUtils.h"
#include "DeviceBlob.h"


struct DeviceBlobTest : public testing::Test {
    using Blob = DeviceBlob<float>;

    float tolerance = 1e-8;
};


TEST_F(DeviceBlobTest, DefaultCtorTest) {
    Blob b;
    ASSERT_EQ(b.rows(), 0);
    ASSERT_EQ(b.cols(), 0);
    ASSERT_EQ(b.count(), 0);
    ASSERT_EQ(b.data(), nullptr);
}

TEST_F(DeviceBlobTest, AllocatingCtorTest) {

    int rows = 2;
    int cols = 3;
    Blob b(rows, cols);
    ASSERT_EQ(b.rows(), rows);
    ASSERT_EQ(b.cols(), cols);
    ASSERT_EQ(b.count(), rows * cols);
    ASSERT_NE(b.data(), nullptr);
}

TEST_F(DeviceBlobTest, CopyFromAndToTest) {
    int rows = 2;
    int cols = 3;
    Blob b(rows, cols);
    float from[] = {1, 2, 3, 4, 5, 6};
    float to[6];

    b.copyFrom(from);
    b.copyTo(to);

    ASSERT_NEAR_VEC(from, to, 6);
}

TEST_F(DeviceBlobTest, CopyFromCtorTest) {
    int rows = 2;
    int cols = 3;
    float from[] = {1, 2, 3, 4, 5, 6};
    float to[6];
    Blob b(rows, cols, from);
    b.copyTo(to);

    ASSERT_NEAR_VEC(from, to, 6);
}

TEST_F(DeviceBlobTest, CopyingCtorTest) {

    int rows = 2;
    int cols = 3;
    float from[] = {1, 2, 3, 4, 5, 6};
    float to[6];
    Blob* b = new Blob(rows, cols, from);
    Blob copiedBlob(*b);
    delete b;

    ASSERT_EQ(copiedBlob.rows(), rows);
    ASSERT_EQ(copiedBlob.cols(), cols);
    ASSERT_EQ(copiedBlob.count(), rows * cols);
    ASSERT_NE(copiedBlob.data(), nullptr);

    copiedBlob.copyTo(to);
    ASSERT_NEAR_VEC(from, to, 6);
}