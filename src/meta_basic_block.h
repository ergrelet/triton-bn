#pragma once

#include <binaryninjaapi.h>

#include <cassert>
#include <triton/basicBlock.hpp>
#include <triton/context.hpp>
#include <vector>

namespace triton_bn {

struct MetaBasicBlock {
  MetaBasicBlock() = default;
  explicit MetaBasicBlock(triton::arch::BasicBlock triton_bb,
                          BinaryNinja::Ref<BinaryNinja::BasicBlock> binja_bb)
      : triton_bb_(triton_bb),
        binja_bb_(binja_bb),
        outgoing_edges_(binja_bb_->GetOutgoingEdges()) {
    assert(binja_bb_.GetPtr() != nullptr);
  }

  triton::arch::BasicBlock& triton_bb() { return triton_bb_; }
  void set_triton_bb(triton::arch::BasicBlock triton_bb) {
    triton_bb_ = std::move(triton_bb);
  }

  BinaryNinja::BasicBlock* binja_bb() { return binja_bb_; }

  const std::vector<BinaryNinja::BasicBlockEdge>& outgoing_edges() const {
    return outgoing_edges_;
  }

  uint64_t GetStart() const { return binja_bb_->GetStart(); }
  std::vector<BinaryNinja::BasicBlockEdge> GetIncomingEdges() const {
    return binja_bb_->GetIncomingEdges();
  }

  void AddOutgoingEdges(
      std::vector<BinaryNinja::BasicBlockEdge> outgoing_edges) {
    std::move(std::begin(outgoing_edges), std::end(outgoing_edges),
              back_inserter(outgoing_edges_));
  }

  void RemoveOutgoingEdge(const BinaryNinja::BasicBlockEdge& edge_to_remove) {
    auto it = std::remove_if(
        std::begin(outgoing_edges_), std::end(outgoing_edges_),
        [&](BinaryNinja::BasicBlockEdge& edge) {
          if (edge.target.GetPtr() != nullptr) {
            return edge.target->GetIndex() == edge_to_remove.target->GetIndex();
          }
          return false;
        });
    outgoing_edges_.erase(it, std::end(outgoing_edges_));
  }

 private:
  triton::arch::BasicBlock triton_bb_{};
  BinaryNinja::Ref<BinaryNinja::BasicBlock> binja_bb_{};
  std::vector<BinaryNinja::BasicBlockEdge> outgoing_edges_{};
};

std::vector<MetaBasicBlock> ExtractMetaBasicBlocksFromBasicBlock(
    BinaryNinja::BinaryView& view,
    BinaryNinja::Ref<BinaryNinja::BasicBlock> basic_block,
    triton::Context& triton);

std::vector<MetaBasicBlock> ExtractMetaBasicBlocksFromFunction(
    BinaryNinja::BinaryView& view,
    BinaryNinja::Ref<BinaryNinja::Function> function, triton::Context& triton);

std::vector<MetaBasicBlock> MergeMetaBasicBlocks(
    std::vector<MetaBasicBlock> basic_blocks);

std::vector<MetaBasicBlock> SimplifyMetaBasicBlocks(
    const triton::Context& triton, std::vector<MetaBasicBlock> basic_blocks);

}  // namespace triton_bn
