#include "./include/utils.hpp"
#include "gtest/gtest.h"
#include <bitset>
#include <mbsm.hpp>
#include <sycl/sycl.hpp>


mbsm::candidates::Signature signature;
uint64_t expected_signature = 0b01000000000000000000100000110001;


TEST(SignatureTest, CheckSignatureMethods) {
  signature.setLabelCount(0, 1);
  signature.setLabelCount(2, 3);
  signature.setLabelCount(5, 2);
  signature.setLabelCount(15, 1);

  ASSERT_EQ(signature.signature, expected_signature);

  ASSERT_EQ(signature.getLabelCount(15), static_cast<uint8_t>(1));

  signature.incrementLabelCount(15);
  expected_signature = 0b10000000000000000000100000110001;

  ASSERT_EQ(signature.signature, expected_signature);

  signature.incrementLabelCount(14);
  expected_signature = 0b10010000000000000000100000110001;

  ASSERT_EQ(signature.signature, expected_signature);

  signature.incrementLabelCount(2);
  ASSERT_EQ(signature.signature, expected_signature);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}