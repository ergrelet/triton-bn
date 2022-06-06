#pragma once

#include <binaryninjaapi.h>

namespace triton_bn {

void SimplifyBasicBlockCommand(BinaryNinja::BinaryView* p_view);
bool ValidateSimplifyBasicBlockCommand(BinaryNinja::BinaryView* p_view);

void SimplifyFunctionCommand(BinaryNinja::BinaryView* p_view);
bool ValidateSimplifyFunctionCommand(BinaryNinja::BinaryView* p_view);

}  // namespace triton_bn
