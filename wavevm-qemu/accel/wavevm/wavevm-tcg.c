
#include "qemu/osdep.h"
#include "cpu.h"
#include "../../../common_include/wavevm_protocol.h"

// Export QEMU TCG state to network packet
void wvm_tcg_get_state(CPUState *cpu, wvm_tcg_context_t *ctx) {
    X86CPU *x86_cpu = X86_CPU(cpu);
    CPUX86State *env = &x86_cpu->env;

    // 1. General Registers
    memcpy(ctx->regs, env->regs, sizeof(ctx->regs));
    ctx->eip = env->eip;
    ctx->eflags = env->eflags;

    // 2. Control Registers
    ctx->cr[0] = env->cr[0];
    ctx->cr[2] = env->cr[2];
    ctx->cr[3] = env->cr[3];
    ctx->cr[4] = env->cr[4];
    
    // 3. SSE/AVX Registers
    // Synchronize XMM0-XMM15 to prevent guest OS crash
    for (int i = 0; i < 16; i++) {
        // Accessing ZMMReg union safely
        // ZMM_Q(n) accesses the nth 64-bit part of the register
        ctx->xmm_regs[i*2]     = env->xmm_regs[i].ZMM_Q(0);
        ctx->xmm_regs[i*2 + 1] = env->xmm_regs[i].ZMM_Q(1);
    }
    ctx->mxcsr = env->mxcsr;
    
    ctx->exit_reason = 0; 

    ctx->fs_base = env->segs[R_FS].base;
    ctx->gs_base = env->segs[R_GS].base;
    ctx->gdt_base = env->gdt.base;
    ctx->gdt_limit = env->gdt.limit;
    ctx->idt_base = env->idt.base;
    ctx->idt_limit = env->idt.limit;
}

// Import state from network packet to QEMU TCG
void wvm_tcg_set_state(CPUState *cpu, wvm_tcg_context_t *ctx) {
    X86CPU *x86_cpu = X86_CPU(cpu);
    CPUX86State *env = &x86_cpu->env;

    // 1. General Registers
    memcpy(env->regs, ctx->regs, sizeof(env->regs));
    env->eip = ctx->eip;
    env->eflags = ctx->eflags;

    // 2. Control Registers
    env->cr[0] = ctx->cr[0];
    env->cr[2] = ctx->cr[2];
    env->cr[3] = ctx->cr[3];
    env->cr[4] = ctx->cr[4];
    
    // 3. SSE/AVX Registers
    for (int i = 0; i < 16; i++) {
        env->xmm_regs[i].ZMM_Q(0) = ctx->xmm_regs[i*2];
        env->xmm_regs[i].ZMM_Q(1) = ctx->xmm_regs[i*2 + 1];
    }
    env->mxcsr = ctx->mxcsr;
    
    // Critical: Flush TB cache to force recompilation with new state
    tb_flush(cpu);

    env->segs[R_FS].base = ctx->fs_base;
    env->segs[R_GS].base = ctx->gs_base;
    env->gdt.base = ctx->gdt_base;
    env->gdt.limit = ctx->gdt_limit;
    env->idt.base = ctx->idt_base;
    env->idt.limit = ctx->idt_limit;
}

