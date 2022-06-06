// Code taken and adapted from Triton's `dead_store_elimination.py` unit test.

#include <cstdio>
#include <iostream>
#include <triton/api.hpp>
#include <triton/basicBlock.hpp>
#include <triton/x86Specifications.hpp>

#define TRITON_INSTR_STR(STR) \
  triton::arch::Instruction { (uint8_t*)STR, (uint32_t)(sizeof(STR) - 1) }

// Code from VMProtect
constexpr uint64_t kBB1Address = 0x140004149;
static triton::arch::BasicBlock get_bb1() {
  return {{
      TRITON_INSTR_STR("\x66\xd3\xd7"),              // rcl     di, cl
      TRITON_INSTR_STR("\x58"),                      // pop     rax
      TRITON_INSTR_STR("\x66\x41\x0f\xa4\xdb\x01"),  // shld    r11w, bx, 1
      TRITON_INSTR_STR("\x41\x5b"),                  // pop     r11
      TRITON_INSTR_STR("\x80\xe6\xca"),              // and     dh, 0CAh
      TRITON_INSTR_STR("\x66\xf7\xd7"),              // not     di
      TRITON_INSTR_STR("\x5f"),                      // pop     rdi
      TRITON_INSTR_STR("\x66\x41\xc1\xc1\x0c"),      // rol     r9w, 0Ch
      TRITON_INSTR_STR("\xf9"),                      // stc
      TRITON_INSTR_STR("\x41\x58"),                  // pop     r8
      TRITON_INSTR_STR("\xf5"),                      // cmc
      TRITON_INSTR_STR("\xf8"),                      // clc
      TRITON_INSTR_STR("\x66\x41\xc1\xe1\x0b"),      // shl     r9w, 0Bh
      TRITON_INSTR_STR("\x5a"),                      // pop     rdx
      TRITON_INSTR_STR("\x66\x81\xf9\xeb\xd2"),      // cmp     cx, 0D2EBh
      TRITON_INSTR_STR("\x48\x0f\xa3\xf1"),          // bt      rcx, rsi
      TRITON_INSTR_STR("\x41\x59"),                  // pop     r9
      TRITON_INSTR_STR("\x66\x41\x21\xe2"),          // and r10w, sp
      TRITON_INSTR_STR("\x41\xc1\xd2\x10"),          // rcl     r10d, 10h
      TRITON_INSTR_STR("\x41\x5a"),                  // pop     r10
      TRITON_INSTR_STR("\x66\x0f\xba\xf9\x0c"),      // btc     cx, 0Ch
      TRITON_INSTR_STR("\x49\x0f\xcc"),              // bswap   r12
      TRITON_INSTR_STR(
          "\x48\x3d\x97\x74\x7d\xc7"),   // cmp     rax, 0FFFFFFFFC77D7497h
      TRITON_INSTR_STR("\x41\x5c"),      // pop r12
      TRITON_INSTR_STR("\x66\xd3\xc1"),  // rol     cx, cl
      TRITON_INSTR_STR("\xf5"),          // cmc
      TRITON_INSTR_STR("\x66\x0f\xba\xf5\x01"),  // btr     bp, 1
      TRITON_INSTR_STR("\x66\x41\xd3\xfe"),      // sar r14w, cl
      TRITON_INSTR_STR("\x5d"),                  // pop     rbp
      TRITON_INSTR_STR("\x66\x41\x29\xf6"),      // sub r14w, si
      TRITON_INSTR_STR("\x66\x09\xf6"),          // or      si, si
      TRITON_INSTR_STR("\x01\xc6"),              // add     esi, eax
      TRITON_INSTR_STR("\x66\x0f\xc1\xce"),      // xadd    si, cx
      TRITON_INSTR_STR("\x9d"),                  // popfq
      TRITON_INSTR_STR("\x0f\x9f\xc1"),          // setnle  cl
      TRITON_INSTR_STR("\x0f\x9e\xc1"),          // setle   cl
      TRITON_INSTR_STR("\x4c\x0f\xbe\xf0"),      // movsx   r14, al
      TRITON_INSTR_STR("\x59"),                  // pop     rcx
      TRITON_INSTR_STR("\xf7\xd1"),              // not     ecx
      TRITON_INSTR_STR("\x59"),                  // pop     rcx
      TRITON_INSTR_STR(
          "\x4c\x8d\xa8\xed\x19\x28\xc9"),       // lea r13, [rax - 36D7E613h]
      TRITON_INSTR_STR("\x66\xf7\xd6"),          // not     si
      TRITON_INSTR_STR("\x41\x5e"),              // pop     r14
      TRITON_INSTR_STR("\x66\xf7\xd6"),          // not     si
      TRITON_INSTR_STR("\x66\x44\x0f\xbe\xea"),  // movsx   r13w, dl
      TRITON_INSTR_STR("\x41\xbd\xb2\x6b\x48\xb7"),  // mov     r13d, 0B7486BB2h
      TRITON_INSTR_STR("\x5e"),                      // pop     rsi
      TRITON_INSTR_STR("\x66\x41\xbd\xca\x44"),      // mov     r13w, 44CAh
      TRITON_INSTR_STR(
          "\x4c\x8d\xab\x31\x11\x63\x14"),  // lea r13, [rbx + 14631131h]
      TRITON_INSTR_STR("\x41\x0f\xcd"),     // bswap   r13d
      TRITON_INSTR_STR("\x41\x5d"),         // pop     r13
      TRITON_INSTR_STR("\xc3"),             // ret
  }};
}

static void bb1_test() {
  /* Init the triton context */
  triton::API triton{};
  triton.setArchitecture(triton::arch::ARCH_X86_64);

  // Process first test case
  // Copy `s_bb1`
  auto bb1 = get_bb1();
  // Disassemble instructions
  triton.disassembly(bb1, kBB1Address);
  {
    std::ostringstream ostr;
    ostr << bb1;
    printf("Original:\n%s\n", ostr.str().c_str());
  }

  // Simply basic block
  auto simplified_bb1 = triton.simplify(bb1);

  // Check result
  triton.disassembly(simplified_bb1, kBB1Address);
  {
    std::ostringstream ostr;
    ostr << simplified_bb1;
    printf("Simplified:\n%s\n", ostr.str().c_str());
  }
}

int main(int argc, char* argv[]) {
  std::printf("Hello World!\n");

  bb1_test();

  return 0;
}
