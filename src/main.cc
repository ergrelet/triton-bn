#include <binaryninjaapi.h>

#include "commands.h"

using namespace BinaryNinja;

extern "C" {

BN_DECLARE_CORE_ABI_VERSION

BINARYNINJAPLUGIN bool CorePluginInit() {
  auto settings = Settings::Instance();
  settings->RegisterGroup("triton-bn", "triton-bn");
  settings->RegisterSetting("triton-bn.mergeBasicBlocks", R"({
		"title" : "Merge basic blocks before simplification",
		"type" : "boolean",
		"default" : true,
		"description" : "Automatically merge basic blocks linked with a single unconditional branch before running the simplification passes on functions."
	})");

  // Preview commands
  PluginCommand::Register("triton-bn\\Preview\\Simplify basic block (DSE)",
                          "Simplify basic block using Triton's DSE pass",
                          triton_bn::SimplifyBasicBlockPreviewCommand,
                          triton_bn::ValidateSimplifyBasicBlockCommand);
  PluginCommand::Register("triton-bn\\Preview\\Simplify function (DSE)",
                          "Simplify function using Triton's DSE pass",
                          triton_bn::SimplifyFunctionPreviewCommand,
                          triton_bn::ValidateSimplifyFunctionCommand);
  // Patch commands
  PluginCommand::Register("triton-bn\\Patch\\Simplify basic block (DSE)",
                          "Simplify basic block using Triton's DSE pass",
                          triton_bn::SimplifyBasicBlockPatchCommand,
                          triton_bn::ValidateSimplifyBasicBlockCommand);
  PluginCommand::Register("triton-bn\\Patch\\Simplify function (DSE)",
                          "Simplify function using Triton's DSE pass",
                          triton_bn::SimplifyFunctionPatchCommand,
                          triton_bn::ValidateSimplifyFunctionCommand);

  return true;
}
}
