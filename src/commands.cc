#include "commands.h"

#include <fmt/format.h>

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "meta_basic_block.h"

namespace triton_bn {

using namespace BinaryNinja;

static FlowGraph* GenerateFlowGraphFromMetaBasicBlocks(
    std::vector<MetaBasicBlock> basic_blocks);

void SimplifyBasicBlockCommand(BinaryNinja::BinaryView* p_view) {
  BinaryView& view = *p_view;

  // Get currently selected address in the view
  const auto current_offset = view.GetCurrentOffset();
  LogDebug("Current offset=0x%p", (void*)current_offset);

  // Find the function in which this address resides
  const auto candidate_basic_blocks =
      view.GetBasicBlocksForAddress(current_offset);
  if (candidate_basic_blocks.empty()) {
    LogError("Failed to find the currently selected basic block");
    return;
  }
  // TODO: Alert the users if multiple candidate basic blocks exist?
  const auto basic_block = candidate_basic_blocks[0];
  LogDebug("Current basic block=0x%p", (void*)basic_block->GetStart());

  // Determine the current platform/architecture
  const std::string architecture_name =
      view.GetDefaultArchitecture()->GetName();
  LogDebug("Architecture is '%s'", architecture_name.c_str());

  triton::API triton{};
  if (architecture_name == "x86_64") {
    triton.setArchitecture(triton::arch::ARCH_X86_64);
  } else if (architecture_name == "x86") {
    triton.setArchitecture(triton::arch::ARCH_X86);
  }

  auto meta_basic_blocks =
      ExtractMetaBasicBlocksFromBasicBlock(view, basic_block, triton);

  // Simplify basic block
  auto simplified_basic_blocks =
      SimplifyMetaBasicBlocks(triton, std::move(meta_basic_blocks));

  // Construct result flow graph and display it
  FlowGraph* flow_graph =
      GenerateFlowGraphFromMetaBasicBlocks(std::move(simplified_basic_blocks));

  const std::string report_title =
      fmt::format("Simplified basic block (0x{:x})", basic_block->GetStart());
  view.ShowGraphReport(report_title, flow_graph);
}

bool ValidateSimplifyBasicBlockCommand(BinaryView* p_view) {
  return ValidateSimplifyFunctionCommand(p_view);
}

void SimplifyFunctionCommand(BinaryView* p_view) {
  BinaryView& view = *p_view;

  // Get currently selected address in the view
  const auto current_offset = view.GetCurrentOffset();
  LogDebug("Current offset=0x%p", (void*)current_offset);

  // Find the function in which this address resides
  const auto candidate_functions =
      view.GetAnalysisFunctionsContainingAddress(current_offset);
  if (candidate_functions.empty()) {
    LogError("Failed to find the currently selected function");
    return;
  }
  // TODO: Alert the users if multiple candidate functions exist
  const auto current_function = candidate_functions[0];
  LogDebug("Current function=0x%p", (void*)current_function->GetStart());

  // Determine the current architecture
  const std::string architecture_name =
      view.GetDefaultArchitecture()->GetName();
  LogDebug("Architecture is '%s'", architecture_name.c_str());

  // Intialize Triton's context
  triton::API triton{};
  if (architecture_name == "x86_64") {
    triton.setArchitecture(triton::arch::ARCH_X86_64);
  } else if (architecture_name == "x86") {
    triton.setArchitecture(triton::arch::ARCH_X86);
  }

  // Create `MetaBasicBlock`s from the current Binja function's basic blocks
  auto meta_basic_blocks =
      ExtractMetaBasicBlocksFromFunction(view, current_function, triton);
  LogDebug("%zu meta basic block(s) extracted", meta_basic_blocks.size());

  if (Settings::Instance()->Get<bool>("triton-bn.mergeBasicBlocks")) {
    // Merge basic blocks
    meta_basic_blocks = MergeMetaBasicBlocks(std::move(meta_basic_blocks));
  }

  // Simplify basic blocks
  auto simplified_basic_blocks =
      SimplifyMetaBasicBlocks(triton, std::move(meta_basic_blocks));

  // Construct result flow graph and display it
  const std::string current_function_name =
      current_function->GetSymbol()->GetFullName();
  const std::string report_title =
      fmt::format("Simplified function ({})", current_function_name);
  FlowGraph* flow_graph =
      GenerateFlowGraphFromMetaBasicBlocks(std::move(simplified_basic_blocks));
  view.ShowGraphReport(report_title, flow_graph);
}

bool ValidateSimplifyFunctionCommand(BinaryView* p_view) {
  if (p_view == nullptr) {
    return false;
  }
  BinaryView& view = *p_view;

  // Check platform compatibility
  const std::string architecture_name =
      view.GetDefaultArchitecture()->GetName();
  if (architecture_name != "x86_64" && architecture_name != "x86") {
    LogError("Unsupported architecture");
    return false;
  }

  return true;
}

static FlowGraph* GenerateFlowGraphFromMetaBasicBlocks(
    std::vector<MetaBasicBlock> basic_blocks) {
  auto* flow_graph = new FlowGraph();

  std::unordered_map<uint64_t, FlowGraphNode*> graph_nodes_addr_map{};
  std::vector<FlowGraphNode*> graph_nodes{};
  for (auto& meta_bb : basic_blocks) {
    // Generate disassembly
    std::vector<DisassemblyTextLine> disassembly_lines{};
    const auto& instructions = meta_bb.triton_bb().getInstructions();
    for (auto& instr : instructions) {
      InstructionTextToken instr_token(
          BNInstructionTextTokenType::InstructionToken, instr.getDisassembly(),
          0, static_cast<size_t>(instr.getSize()));
      instr_token.address = instr.getAddress();

      {
        DisassemblyTextLine line;
        line.addr = instr.getAddress();
        line.tokens = {instr_token};
        disassembly_lines.emplace_back(std::move(line));
      }
    }

    // Construct new node
    {
      auto* node = new FlowGraphNode(flow_graph);
      node->SetLines(disassembly_lines);
      graph_nodes.emplace_back(node);
      graph_nodes_addr_map[meta_bb.GetStart()] = node;
    }
  }

  if (basic_blocks.size() != graph_nodes.size()) {
    LogError("Invalid graph node count");
    return nullptr;
  }

  // Construct graph
  for (size_t i = 0; i < graph_nodes.size(); i++) {
    FlowGraphNode* graph_node = graph_nodes[i];
    MetaBasicBlock& meta_bb = basic_blocks[i];

    // Resolve outgoing edges
    for (const BasicBlockEdge& outgoing_edge : meta_bb.outgoing_edges()) {
      const BasicBlock* target = outgoing_edge.target;
      if (target == nullptr) {
        continue;
      }

      const uint64_t target_addr = target->GetStart();
      const auto node_it = graph_nodes_addr_map.find(target_addr);
      if (node_it == std::cend(graph_nodes_addr_map)) {
        continue;
      }

      graph_node->AddOutgoingEdge(outgoing_edge.type, node_it->second);
    }

    flow_graph->AddNode(graph_node);
  }

  return flow_graph;
}

}  // namespace triton_bn
