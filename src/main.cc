#include <binaryninjaapi.h>
#include <uicontext.h>

#include "commands.h"

using namespace BinaryNinja;

extern "C" {

BN_DECLARE_UI_ABI_VERSION

BINARYNINJAPLUGIN bool UIPluginInit() {
  auto settings = Settings::Instance();
  settings->RegisterGroup("triton-bn", "triton-bn");
  settings->RegisterSetting("triton-bn.mergeBasicBlocks", R"({
		"title" : "Merge basic blocks before simplification",
		"type" : "boolean",
		"default" : true,
		"description" : "Automatically merge basic blocks linked with a single unconditional branch before running the simplification passes on functions."
	})");

  PluginCommand::Register(
      "triton-bn\\Simplify basic block (dead store elimination)",
      "Simplify basic block using Triton's DSE pass",
      triton_bn::SimplifyBasicBlockCommand,
      triton_bn::ValidateSimplifyBasicBlockCommand);

  PluginCommand::Register(
      "triton-bn\\Simplify function (dead store elimination)",
      "Simplify function using Triton's DSE pass",
      triton_bn::SimplifyFunctionCommand,
      triton_bn::ValidateSimplifyFunctionCommand);

  return true;
}
}
