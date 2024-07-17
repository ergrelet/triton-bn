#pragma once

#include <binaryninjaapi.h>

namespace triton_bn {

void SimplifyBasicBlockPreviewCommand(BinaryNinja::BinaryView* p_view);
void SimplifyBasicBlockPatchCommand(BinaryNinja::BinaryView* p_view);
bool ValidateSimplifyBasicBlockCommand(BinaryNinja::BinaryView* p_view);

void SimplifyFunctionPreviewCommand(BinaryNinja::BinaryView* p_view);
void SimplifyFunctionPatchCommand(BinaryNinja::BinaryView* p_view);
bool ValidateSimplifyFunctionCommand(BinaryNinja::BinaryView* p_view);

}  // namespace triton_bn
