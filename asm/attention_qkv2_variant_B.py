import jax.numpy as jnp

def qkv(kernel, api):
    @kernel(
        hbm=32768,  # 32 KB: enough for 3 inputs + 1 output
        input=[
            {'addr': 0, 'shape': (64, 64), 'dtype': jnp.bfloat16},      # Q
            {'addr': 8192, 'shape': (64, 64), 'dtype': jnp.bfloat16},   # K
            {'addr': 16384, 'shape': (64, 64), 'dtype': jnp.bfloat16},  # V
        ],
        constant=[],  # No constants needed
        output=[
            {'addr': 24576, 'shape': (64, 64), 'dtype': jnp.bfloat16},
        ]
    )
    def qkv_():
        #kernel implementation goes here

        #load k from hbm to d1
        api.load_rm_d1(n=64, addr_in=8192, addr_out=0)

        #tranpose k to k^t
        api.transpose_d1_d3(n=64, addr_in=0, addr_out=0)

        #load q from hbm to d1
        api.load_rm_d1(n=64, addr_in=0, addr_out=0)

        # Compute S = Q × K^T
        api.gemm_d1_d3(addr_1=0, addr_2=0, addr_out=0)

        # Apply softmax to S, converting it to P (in-place in d2)
        api.softmax(n=64, addr=0)

        # move p to d3
        api.copy_d2_d3(n=64, addr_in=0, addr_out=0)

        # load v from hbm to d3
        api.load_rm_d3(n=64, addr_in=16384, addr_out=64)

        # compute O from P x V
        api.gemm_d3_d3(addr_1=0, addr_2=64, addr_out=0)

        #store O back to hbm
        api.store_rm_d2(n=64, addr_in=0, addr_out=24576)

    return qkv_