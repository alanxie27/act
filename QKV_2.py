"""QKV Accelerator ISA Definition for Tutorial 2"""

from taidl import Accelerator

qkv_2 = Accelerator("QKV_2")

# Define Data Models
qkv_2.add_data_model("d1", [128], [64], "bf16")
qkv_2.add_data_model("d2", [64], [64], "bf16")
qkv_2.add_data_model("d3", [128], [64], "bf16")

# Load instructions
instr = qkv_2.add_instruction("load_rm_d1", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])
instr.set_outputs([["d1", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics("""
ENTRY load_rm_d1 {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")

instr = qkv_2.add_instruction("load_rm_d2", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])
instr.set_outputs([["d2", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics("""
ENTRY load_rm_d2 {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")

instr = qkv_2.add_instruction("load_rm_d3", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])
instr.set_outputs([["d3", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics("""
ENTRY load_rm_d3 {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")


instr = qkv_2.add_instruction("store_rm_d2", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d2", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["d0", ["@a.addr_out"], ["@c.n * 128"]]])
instr.add_semantics("""
ENTRY store_rm_d2 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%a);
}
""")


# Compute instructions
instr = qkv_2.add_instruction("gemm_d1_d3", [], ["addr_1", "addr_2", "addr_out"])
instr.set_inputs([["d1", ["@a.addr_1"], ["64"]], ["d3", ["@a.addr_2"], ["64"]]])
instr.set_outputs([["d2", ["@a.addr_out"], ["64"]]])
instr.add_semantics("""
ENTRY gemm_d1_d3 {
    %In1 = bf16[64,64] parameter(0);
    %In2 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[64,64] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")

instr = qkv_2.add_instruction("gemm_d3_d3", [], ["addr_1", "addr_2", "addr_out"])
instr.set_inputs([["d3", ["@a.addr_1"], ["64"]], ["d3", ["@a.addr_2"], ["64"]]])
instr.set_outputs([["d2", ["@a.addr_out"], ["64"]]])
instr.add_semantics("""
ENTRY gemm_d3_d3 {
    %In1 = bf16[64,64] parameter(0);
    %In2 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[64,64] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")


instr = qkv_2.add_instruction("softmax", ["n"], ["addr"])
instr.set_inputs([["d2", ["@a.addr"], ["@c.n"]]])
instr.set_outputs([["d2", ["@a.addr"], ["@c.n"]]])
instr.add_semantics("""
ENTRY softmax {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = bf16[`@c.n`,64] exponential(%In1);
    %reduced = bf16[`@c.n`] reduce_add(%a), dimensions={1};
    %b = bf16[`@c.n`,64] broadcast(%reduced), dimensions={0};
    ROOT %Out0 = bf16[`@c.n`,64] divide(%a, %b);
}
""")

instr = qkv_2.add_instruction("copy_d2_d1", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d2", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["d1", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics("""
ENTRY copy_d2_d1 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In1);
}
""")

instr = qkv_2.add_instruction("copy_d2_d3", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d2", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["d3", ["@a.addr_out"], ["@c.n"]]])
instr.add_semantics("""
ENTRY copy_d2_d3 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In1);
}
""")

instr = qkv_2.add_instruction("transpose_d1_d3", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d1", ["@a.addr_in"], ["@c.n"]]])
instr.set_outputs([["d3", ["@a.addr_out"], ["64"]]])
instr.add_semantics("""
ENTRY transpose_d1_d3 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[64,`@c.n`] transpose(%In1), dimensions={1,0};
}
""")

 # Generate programming APIs and test oracle (functional simulator)
qkv_2.generate_oracle()

  # Generate compiler backend
qkv_2.generate_backend()