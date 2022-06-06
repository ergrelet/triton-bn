#include "meta_basic_block.h"

#include <iterator>

namespace triton_bn {

using namespace BinaryNinja;

static void MergeLinkedBasicBlocks(const BasicBlockEdge& edge,
                                   MetaBasicBlock& root_bb,
                                   MetaBasicBlock& target_bb);

// Transform a given "Binary Ninja" basic block into one or several
// `MetaBasicBlock`s that can be simplified with Triton
std::vector<MetaBasicBlock> ExtractMetaBasicBlocksFromBasicBlock(
    BinaryView& view, Ref<BasicBlock> basic_block, triton::API& triton) {
  // TODO: Merge fallthrough automatically?
  std::vector<MetaBasicBlock> result{};
  triton::arch::BasicBlock triton_bb{};

  const auto default_arch = view.GetDefaultArchitecture();
  const size_t max_instr_len = default_arch->GetMaxInstructionLength();

  // Iterate through the disassembly
  uint64_t last_instr_addr = 0;
  DisassemblySettings settings{};
  auto disassembly_lines = basic_block->GetDisassemblyText(&settings);
  for (size_t i = 0; i < disassembly_lines.size(); i++) {
    const uint64_t cur_instr_addr = disassembly_lines[i].addr;
    if (cur_instr_addr == last_instr_addr) {
      // Note: Might happen when the basic block has a 'fallthrough' edge,
      // ignore.
      continue;
    }
    last_instr_addr = cur_instr_addr;

    const DataBuffer cur_instr_data =
        view.ReadBuffer(cur_instr_addr, max_instr_len);
    BinaryNinja::InstructionInfo binja_instruction{};
    if (!default_arch->GetInstructionInfo(&cur_instr_data[0], cur_instr_addr,
                                          max_instr_len, binja_instruction)) {
      continue;
    }
    LogDebug("0x%p - %zu", (void*)cur_instr_addr, binja_instruction.length);

    // Add disassembled instruction to the basic block
    triton::arch::Instruction new_instr(
        cur_instr_addr, static_cast<const uint8_t*>(cur_instr_data.GetData()),
        binja_instruction.length);
    try {
      triton.disassembly(new_instr);
    } catch (triton::exceptions::Disassembly& ex) {
      LogError("Failed to disassemble instruction at address 0x%p",
               (void*)cur_instr_addr);
      return {};
    }
    triton_bb.add(new_instr);

    // Split basic blocks on `call` instructions to make them simplifiable
    // TODO: Fix for ARM64
    if (new_instr.getDisassembly().find("call") == 0) {
      LogDebug("call detected: %s", new_instr.getDisassembly().c_str());
      // Add basic block to the result
      result.emplace_back(MetaBasicBlock(triton_bb, basic_block));
      triton_bb = {};
    }
  }
  // Add basic block to the result
  result.emplace_back(MetaBasicBlock(triton_bb, basic_block));

  return result;
}

// Same as `ExtractMetaBasicBlocksFromBasicBlock`, except we extract
// `MetaBasicBlock`s from the basic blocks of a given "Binary Ninja" function.
std::vector<MetaBasicBlock> ExtractMetaBasicBlocksFromFunction(
    BinaryView& view, Ref<Function> function, triton::API& triton) {
  std::vector<MetaBasicBlock> func_meta_basic_blocks{};

  // Iterate through the basic blocks
  for (auto binja_bb : function->GetBasicBlocks()) {
    auto meta_basic_blocks =
        ExtractMetaBasicBlocksFromBasicBlock(view, std::move(binja_bb), triton);
    std::move(std::begin(meta_basic_blocks), std::end(meta_basic_blocks),
              back_inserter(func_meta_basic_blocks));
  }

  return func_meta_basic_blocks;
}

// Merge `MetaBasicBlock`s which are linked with single unconditional branches
std::vector<MetaBasicBlock> MergeMetaBasicBlocks(
    std::vector<MetaBasicBlock> basic_blocks) {
  using BasicBlockMergeMap =
      std::unordered_map<size_t, std::pair<MetaBasicBlock*, bool>>;

  std::vector<MetaBasicBlock> merged_meta_basic_blocks{};

  // Populate the `"Binary Ninja" index -> MetaBasicBlock` map
  BasicBlockMergeMap binja_bb_map{};
  for (auto& meta_bb : basic_blocks) {
    binja_bb_map[meta_bb.binja_bb()->GetIndex()] =
        std::make_pair(&meta_bb, false);
  }

  // Iterate over the `MetaBasicBlock`s
  for (auto cur_meta_bb : basic_blocks) {
    auto cur_bb_it = binja_bb_map.find(cur_meta_bb.binja_bb()->GetIndex());
    if (cur_bb_it == std::end(binja_bb_map)) {
      LogError("The basic block index map isn't valid");
      return {};
    }
    auto& cur_bb_pair = cur_bb_it->second;
    if (cur_bb_pair.second) {
      // Basic block has been merged, ignore.
      continue;
    }

    // Iterate through unconditionally linked blocks and merge them until it's
    // not possible
    for (;;) {
      const std::vector<BasicBlockEdge>& outgoing_egdes =
          cur_meta_bb.outgoing_edges();

      const auto outgoing_edge_it = std::find_if(
          std::cbegin(outgoing_egdes), std::cend(outgoing_egdes),
          [](const BasicBlockEdge& edge) {
            return edge.type == BNBranchType::UnconditionalBranch &&
                   edge.target.GetPtr() != nullptr &&
                   edge.target->GetIncomingEdges().size() == 1;
          });
      if (outgoing_edge_it == std::cend(outgoing_egdes)) {
        // No mergeable outgoing edge found, stop the merging process
        break;
      }

      auto target_bb_it =
          binja_bb_map.find(outgoing_edge_it->target->GetIndex());
      if (target_bb_it == std::end(binja_bb_map)) {
        // Invalid target basic block, stop the merging process
        break;
      }

      auto& target_bb_pair = target_bb_it->second;
      if (target_bb_pair.second) {
        // Target has already been merged, stop the merging process
        LogDebug("Target already merged? Aborting");
        break;
      }

      // Proceed with the merge
      MetaBasicBlock* target_meta_bb = target_bb_pair.first;
      MergeLinkedBasicBlocks(*outgoing_edge_it, cur_meta_bb, *target_meta_bb);
      // Mark basic block as merged
      target_bb_pair.second = true;
    }

    merged_meta_basic_blocks.emplace_back(std::move(cur_meta_bb));
  }

  return merged_meta_basic_blocks;
}

// Merge two `MetaBasicBlock`s linked by a given edge
static void MergeLinkedBasicBlocks(const BasicBlockEdge& edge,
                                   MetaBasicBlock& root_bb,
                                   MetaBasicBlock& target_bb) {
  triton::arch::BasicBlock& cur_triton_bb = root_bb.triton_bb();
  triton::arch::BasicBlock& target_triton_bb = target_bb.triton_bb();

  // Merge Triton's basic block
  const uint64_t instruction_count = cur_triton_bb.getSize();
  if (instruction_count > 0) {
    const uint64_t last_instr_index = instruction_count - 1;
    const triton::arch::Instruction last_instr =
        cur_triton_bb.getInstructions()[last_instr_index];
    // Remove last instruction if it's a `jmp`
    // TODO: Fix for ARM64
    if (last_instr.getDisassembly().find("jmp") == 0) {
      cur_triton_bb.remove(last_instr_index);
    }
  }
  // Merge Triton instructions into the current basic block
  for (triton::arch::Instruction& instr : target_triton_bb.getInstructions()) {
    cur_triton_bb.add(instr);
  }
  // Remove the merged edge
  root_bb.RemoveOutgoingEdge(edge);
  // Merge Binja's outgoing edges
  root_bb.AddOutgoingEdges(target_bb.outgoing_edges());
}

// Simplify the given `MetaBasicBlock`s with Triton's dead store elimination
// pass
std::vector<MetaBasicBlock> SimplifyMetaBasicBlocks(
    const triton::API& triton, std::vector<MetaBasicBlock> basic_blocks) {
  // Simplify basic blocks
  std::vector<MetaBasicBlock> simplified_basic_blocks(basic_blocks.size());
  {
    const auto triton_arch = triton.getArchitecture();
    bool transform_failed = false;
    std::transform(
        std::begin(basic_blocks), std::end(basic_blocks),
        std::begin(simplified_basic_blocks),
        [&](MetaBasicBlock meta_bb) -> MetaBasicBlock {
          // Intialize Triton's context
          triton::API triton{};
          triton.setArchitecture(triton_arch);

          // Simplify basic blocks and disassemble the result
          try {
            auto simplified_triton_bb = triton.simplify(meta_bb.triton_bb());
            triton.disassembly(simplified_triton_bb, meta_bb.GetStart());
            meta_bb.set_triton_bb(simplified_triton_bb);
            return std::move(meta_bb);
          } catch (triton::exceptions::Exception& ex) {
            LogError("Failed to simplify basic block: %s", ex.what());
            transform_failed = true;
            return {};
          }
        });
    if (transform_failed) {
      LogError("Failed to simplify function");
      return {};
    }
  }

  // Regroup split simplified basic blocks
  std::unordered_map<uint64_t, MetaBasicBlock*> final_bbs_addr_map{};
  std::vector<MetaBasicBlock> final_basic_blocks{};
  final_basic_blocks.reserve(simplified_basic_blocks.size());
  for (MetaBasicBlock meta_bb : simplified_basic_blocks) {
    const uint64_t bb_addr = meta_bb.GetStart();
    if (final_bbs_addr_map[bb_addr] == nullptr) {
      // Add to the list
      final_basic_blocks.emplace_back(std::move(meta_bb));
      final_bbs_addr_map[bb_addr] =
          &final_basic_blocks[final_basic_blocks.size() - 1];
    } else {
      // Regroup with the previous basic block
      MetaBasicBlock* previous_bb = final_bbs_addr_map[bb_addr];
      for (auto& instr : meta_bb.triton_bb().getInstructions()) {
        previous_bb->triton_bb().add(instr);
      }
    }
  }

  return final_basic_blocks;
}

}  // namespace triton_bn