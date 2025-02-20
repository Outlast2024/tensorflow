// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/runtime/compiled_model.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/runtime/open_cl_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"

namespace litert {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::Pointwise;

// Creates input buffers for the given LiteRtCompiledModelT by leveraging
// TensorBufferRequirements.
Expected<std::vector<LiteRtTensorBuffer>> CreateInputBuffers(
    LiteRtModel& model, absl::string_view signature_key,
    LiteRtCompiledModelT& compiled_model) {
  LITERT_ASSIGN_OR_RETURN(LiteRtSubgraph subgraph,
                          LookupSubgraph(*model, signature_key));

  std::vector<LiteRtTensorBuffer> input_buffers;
  input_buffers.reserve(subgraph->NumInputs());
  for (int i = 0; i < subgraph->NumInputs(); ++i) {
    // Get buffer requirements and tensor at index i.
    LITERT_ASSIGN_OR_RETURN(
        const LiteRtTensorBufferRequirements& requirements_at_index,
        compiled_model.GetInputBufferRequirements(signature_key, i));
    const LiteRtTensor& tensor = subgraph->Inputs().at(i);

    // Get buffer type, tensor type and buffer size.
    const LiteRtTensorBufferType& buffer_type =
        requirements_at_index->SupportedBufferTypes().at(0);
    const LiteRtRankedTensorType& ranked_tensor_type =
        tensor->Type().second.ranked_tensor_type;
    size_t bytes = requirements_at_index->BufferSize();

    LiteRtTensorBuffer input_buffer;
    LITERT_RETURN_IF_ERROR(LiteRtCreateManagedTensorBuffer(
        buffer_type, &ranked_tensor_type, bytes, &input_buffer));
    input_buffers.push_back(std::move(input_buffer));
  }
  return input_buffers;
}

// Creates input buffers for the given LiteRtTensorBufferType and size.
Expected<std::vector<LiteRtTensorBuffer>> CreateInputBuffers(
    LiteRtModel& model, absl::string_view signature_key,
    LiteRtTensorBufferType buffer_type, size_t bytes) {
  LITERT_ASSIGN_OR_RETURN(LiteRtSubgraph subgraph,
                          LookupSubgraph(*model, signature_key));

  std::vector<LiteRtTensorBuffer> input_buffers;
  input_buffers.reserve(subgraph->NumInputs());
  for (int i = 0; i < subgraph->NumInputs(); ++i) {
    const LiteRtTensor& tensor = subgraph->Inputs().at(i);

    // Instead of using TensorBufferRequirements, we directly create
    // buffers using buffer_type and bytes.
    const LiteRtRankedTensorType& ranked_tensor_type =
        tensor->Type().second.ranked_tensor_type;
    LiteRtTensorBuffer input_buffer;
    LITERT_RETURN_IF_ERROR(LiteRtCreateManagedTensorBuffer(
        buffer_type, &ranked_tensor_type, bytes, &input_buffer));
    input_buffers.push_back(std::move(input_buffer));
  }
  return input_buffers;
}

// Creates output buffers for the given LiteRtCompiledModelT by leveraging
// TensorBufferRequirements.
Expected<std::vector<LiteRtTensorBuffer>> CreateOutputBuffers(
    LiteRtModel& model, absl::string_view signature_key,
    LiteRtCompiledModelT& compiled_model) {
  LITERT_ASSIGN_OR_RETURN(LiteRtSubgraph subgraph,
                          LookupSubgraph(*model, signature_key));

  std::vector<LiteRtTensorBuffer> output_buffers;
  output_buffers.reserve(subgraph->NumOutputs());
  for (int i = 0; i < subgraph->NumOutputs(); ++i) {
    // Get tensor and buffer requirements at index i.
    LITERT_ASSIGN_OR_RETURN(
        const LiteRtTensorBufferRequirements& requirements_at_index,
        compiled_model.GetOutputBufferRequirements(signature_key, i));
    const LiteRtTensor& tensor = subgraph->Outputs().at(i);

    // Get buffer type, tensor type and buffer size.
    const LiteRtTensorBufferType& buffer_type =
        requirements_at_index->SupportedBufferTypes().at(0);
    const LiteRtRankedTensorType& ranked_tensor_type =
        tensor->Type().second.ranked_tensor_type;
    size_t bytes = requirements_at_index->BufferSize();

    LiteRtTensorBuffer output_buffer;
    LITERT_RETURN_IF_ERROR(LiteRtCreateManagedTensorBuffer(
        buffer_type, &ranked_tensor_type, bytes, &output_buffer));
    output_buffers.push_back(std::move(output_buffer));
  }
  return output_buffers;
}

// Creates output buffers for the given LiteRtTensorBufferType and size.
Expected<std::vector<LiteRtTensorBuffer>> CreateOutputBuffers(
    LiteRtModel& model, absl::string_view signature_key,
    LiteRtTensorBufferType buffer_type, size_t bytes) {
  LITERT_ASSIGN_OR_RETURN(LiteRtSubgraph subgraph,
                          LookupSubgraph(*model, signature_key));

  std::vector<LiteRtTensorBuffer> output_buffers;
  output_buffers.reserve(subgraph->NumOutputs());
  for (int i = 0; i < subgraph->NumOutputs(); ++i) {
    const LiteRtTensor& tensor = subgraph->Outputs().at(i);

    // Instead of using TensorBufferRequirements, we directly create
    // buffers using buffer_type and bytes.
    const LiteRtRankedTensorType& ranked_tensor_type =
        tensor->Type().second.ranked_tensor_type;
    LiteRtTensorBuffer output_buffer;
    LITERT_RETURN_IF_ERROR(LiteRtCreateManagedTensorBuffer(
        buffer_type, &ranked_tensor_type, bytes, &output_buffer));
    output_buffers.push_back(std::move(output_buffer));
  }
  return output_buffers;
}

TEST(CompiledModelTest, Basic) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, LiteRtSignatureT::kDefaultSignatureKey);

  const std::vector<std::string>& input_names = signatures[0]->InputNames();
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  const std::vector<std::string>& output_names = signatures[0]->OutputNames();
  EXPECT_THAT(output_names, ElementsAre("tfl.add"));

  // Create CompiledModel with options.
  LiteRtCompilationOptions compilation_options;
  ASSERT_EQ(LiteRtCreateCompilationOptions(&compilation_options),
            kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                compilation_options, kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(
          env_ptr, model,
          LiteRtCompiledModelT::OptionsPtr(compilation_options)));

  // Check CompiledModel buffer requirements.
  // input and output expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg0,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/0));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg0 =
      input_buffer_requirements_arg0->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg1,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/1));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg1 =
      input_buffer_requirements_arg1->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * output_buffer_requirements,
      compiled_model->GetOutputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*output_index=*/0));
  const std::vector<LiteRtTensorBufferType>& output_buffer_types =
      output_buffer_requirements->SupportedBufferTypes();
  EXPECT_THAT(output_buffer_types,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  // Create and fill input and output LiteRtTensorBuffers. Buffers are
  // created to match CompiledModel's TensorBufferRequirements.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> input_buffers,
      CreateInputBuffers(model, signature_key, *compiled_model));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> output_buffers,
      CreateOutputBuffers(model, signature_key, *compiled_model));

  LiteRtTensorBuffer& input_0_buffer = input_buffers[0];
  {
    TensorBuffer cpu_buffer(input_0_buffer, /*owned=*/false);
    cpu_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  LiteRtTensorBuffer& input_1_buffer = input_buffers[1];
  {
    TensorBuffer cpu_buffer(input_1_buffer, /*owned=*/false);
    cpu_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size));
  }

  // Execute model.
  bool async = false;
  compiled_model->Run(signature_key, input_buffers, output_buffers, async);

  // Check model output.
  {
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_buffers[0], &host_mem_addr),
              kLiteRtStatusOk);
    absl::Span<const float> output = absl::MakeSpan(
        static_cast<const float*>(host_mem_addr), kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_buffers[0]), kLiteRtStatusOk);
  }

  // Since Buffers in LiteRtTensorBuffer, we need to destroy them explicitly.
  for (auto& input_buffer : input_buffers) {
    LiteRtDestroyTensorBuffer(input_buffer);
  }
  for (auto& output_buffer : output_buffers) {
    LiteRtDestroyTensorBuffer(output_buffer);
  }

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, UseAhwbBuffer) {
#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices";
#endif
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, LiteRtSignatureT::kDefaultSignatureKey);

  const std::vector<std::string>& input_names = signatures[0]->InputNames();
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  const std::vector<std::string>& output_names = signatures[0]->OutputNames();
  EXPECT_THAT(output_names, ElementsAre("tfl.add"));

  // Create CompiledModel with options.
  LiteRtCompilationOptions compilation_options;
  ASSERT_EQ(LiteRtCreateCompilationOptions(&compilation_options),
            kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                compilation_options, kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(
          env_ptr, model,
          LiteRtCompiledModelT::OptionsPtr(compilation_options)));

  // Check input and output buffer requirements expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg0,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/0));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg0 =
      input_buffer_requirements_arg0->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg1,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/1));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg1 =
      input_buffer_requirements_arg1->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * output_buffer_requirements,
      compiled_model->GetOutputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*output_index=*/0));
  const std::vector<LiteRtTensorBufferType>& output_buffer_types =
      output_buffer_requirements->SupportedBufferTypes();
  EXPECT_THAT(output_buffer_types,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  // Create and fill input and output buffers. CompiledModel's
  // TensorBufferRequirements expect host memory,but we create AHWB buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> input_buffers,
      CreateInputBuffers(model, signature_key, kLiteRtTensorBufferTypeAhwb,
                         sizeof(float) * kTestInput0Size));

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> output_buffers,
      CreateOutputBuffers(model, signature_key, kLiteRtTensorBufferTypeAhwb,
                          sizeof(float) * kTestOutputSize));

  LiteRtTensorBuffer& input_0_buffer = input_buffers[0];
  EXPECT_EQ(input_0_buffer->buffer_type(), kLiteRtTensorBufferTypeAhwb);
  {
    TensorBuffer ahwb_buffer(input_0_buffer, /*owned=*/false);
    ahwb_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  LiteRtTensorBuffer& input_1_buffer = input_buffers[1];
  {
    TensorBuffer ahwb_buffer(input_1_buffer, /*owned=*/false);
    ahwb_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size));
  }

  // Execute model.
  bool async = false;
  compiled_model->Run(signature_key, input_buffers, output_buffers, async);

  // Check model output.
  {
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_buffers[0], &host_mem_addr),
              kLiteRtStatusOk);
    absl::Span<const float> output = absl::MakeSpan(
        static_cast<const float*>(host_mem_addr), kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_buffers[0]), kLiteRtStatusOk);
  }

  // Since Buffers in LiteRtTensorBuffer, we need to destroy them explicitly.
  for (auto& input_buffer : input_buffers) {
    LiteRtDestroyTensorBuffer(input_buffer);
  }
  for (auto& output_buffer : output_buffers) {
    LiteRtDestroyTensorBuffer(output_buffer);
  }

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, UseOpenCLBuffer) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported In msan";
#endif

  if (!litert::internal::OpenClBuffer::IsSupported()) {
    GTEST_SKIP() << "OpenCL buffers are not supported on this platform; "
                    "skipping the test";
  }
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, LiteRtSignatureT::kDefaultSignatureKey);

  const std::vector<std::string>& input_names = signatures[0]->InputNames();
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  const std::vector<std::string>& output_names = signatures[0]->OutputNames();
  EXPECT_THAT(output_names, ElementsAre("tfl.add"));

  // Create CompiledModel with options.
  LiteRtCompilationOptions compilation_options;
  ASSERT_EQ(LiteRtCreateCompilationOptions(&compilation_options),
            kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                compilation_options, kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(
          env_ptr, model,
          LiteRtCompiledModelT::OptionsPtr(compilation_options)));

  // Check ComiledModel buffer requirements.
  // input and output expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg0,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/0));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg0 =
      input_buffer_requirements_arg0->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg1,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/1));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg1 =
      input_buffer_requirements_arg1->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * output_buffer_requirements,
      compiled_model->GetOutputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*output_index=*/0));
  const std::vector<LiteRtTensorBufferType>& output_buffer_types =
      output_buffer_requirements->SupportedBufferTypes();
  EXPECT_THAT(output_buffer_types,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  // Create and fill input and output buffers. CompiledModel's
  // TensorBufferRequirements expect host memory,but we create OpenCL buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> input_buffers,
      CreateInputBuffers(model, signature_key, kLiteRtTensorBufferTypeOpenCl,
                         sizeof(float) * kTestInput0Size));

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> output_buffers,
      CreateOutputBuffers(model, signature_key, kLiteRtTensorBufferTypeOpenCl,
                          sizeof(float) * kTestOutputSize));

  // Fill model inputs.
  LiteRtTensorBuffer& input_0_buffer = input_buffers[0];
  EXPECT_EQ(input_0_buffer->buffer_type(), kLiteRtTensorBufferTypeOpenCl);
  {
    TensorBuffer opencl_buffer(input_0_buffer, /*owned=*/false);
    opencl_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  LiteRtTensorBuffer& input_1_buffer = input_buffers[1];
  {
    TensorBuffer opencl_buffer(input_1_buffer, /*owned=*/false);
    opencl_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size));
  }

  // Execute model.
  bool async = false;
  compiled_model->Run(signature_key, input_buffers, output_buffers, async);

  // Check model output.
  {
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_buffers[0], &host_mem_addr),
              kLiteRtStatusOk);
    absl::Span<const float> output = absl::MakeSpan(
        static_cast<const float*>(host_mem_addr), kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));

    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_buffers[0]), kLiteRtStatusOk);
  }

  // Since Buffers in LiteRtTensorBuffer, we need to destroy them explicitly.
  for (auto& input_buffer : input_buffers) {
    LiteRtDestroyTensorBuffer(input_buffer);
  }
  for (auto& output_buffer : output_buffers) {
    LiteRtDestroyTensorBuffer(output_buffer);
  }

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}
}  // namespace
}  // namespace litert
