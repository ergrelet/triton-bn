#include <binaryninjaapi.h>
#include <uicontext.h>

#include <triton/api.hpp>
#include <triton/basicBlock.hpp>
#include <triton/x86Specifications.hpp>

using namespace BinaryNinja;

namespace triton_bn {

void simplify_function_command(BinaryView* p_view) { LogInfo("Hello World!"); }

}  // namespace triton_bn

extern "C" {

BN_DECLARE_UI_ABI_VERSION

BINARYNINJAPLUGIN bool UIPluginInit() {
  PluginCommand::Register("triton-bn\\Simplify function",
                          "Simply function using Triton",
                          triton_bn::simplify_function_command);

  return true;
}
}
