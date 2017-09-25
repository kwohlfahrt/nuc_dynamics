import numpy as np
import pyopencl as cl
import pytest
from pathlib import Path

@pytest.fixture
def ctx():
    return cl.create_some_context()

@pytest.fixture
def cq(ctx):
    return cl.CommandQueue(ctx)

@pytest.fixture
def kernels(ctx):
    from nuc_dynamics.nuc_dynamics import compile_kernels
    return compile_kernels(ctx)

def test_get_temp():
    from nuc_dynamics.run import getTemp

    np.random.seed(4)
    masses = np.random.uniform(0.1, 1.0, size=20)
    masses[2:4] = float('inf')
    veloc = np.random.uniform(-1.0, 1.0, size=(20, 3))
    values = getTemp(masses, veloc)
    expected = 65.77467716280236 # Using Tim's as reference
    assert values == expected


def test_get_stats():
    from nuc_dynamics.run import getStats

    np.random.seed(4)
    coords = np.random.uniform(-1.0, 1.0, size=(20, 3))
    indices = np.random.choice(len(coords), (40, 2)).astype('int32')
    limits = np.stack([np.random.uniform(0.1, 0.5, size=len(indices)),
                       np.random.uniform(0.6, 1.2, size=len(indices))], axis=-1)
    weights = np.ones(len(indices), dtype='float64')

    values = getStats(indices, limits, weights, coords)
    assert values == (31, 0.721443286736486)


def test_restraint_force(ctx, cq, kernels):
    from nuc_dynamics import fp_type

    # Forces are repulsive 0-1, zero 1-2, attractive 2-3, asymptotic 3+
    coords = np.array([
        [0.0, 0.0, 0.0, float('nan')],
        [0.0, 0.0, 0.5, float('nan')],
        [0.0, 0.0, 1.5, float('nan')],
        [0.0, 0.0, 2.5, float('nan')],
        [0.0, 0.0, 3.5, float('nan')],
        [0.0, 0.0, 5.0, float('nan')],
    ], dtype=fp_type)
    indices = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
    ], dtype='int32')
    weights = np.ones(len(indices), dtype=fp_type)
    forces = np.zeros(coords.shape, dtype=fp_type)
    limits = np.ascontiguousarray(
        np.broadcast_to([[1.0, 2.0]], (len(indices), 2)).astype(fp_type)
    )
    ambig = np.arange(len(indices) + 1, dtype='int32')
    forces = np.zeros(coords.shape, dtype=fp_type)

    coords_buf = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR, hostbuf=coords)
    indices_buf = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR, hostbuf=indices)
    weights_buf = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR, hostbuf=weights)
    forces_buf = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR, hostbuf=forces)
    limits_buf = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR, hostbuf=limits)
    ambig_buf = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR, hostbuf=ambig)

    e = kernels['getRestraintForce'](
        cq, (len(ambig) - 1,), None,
        indices_buf, limits_buf, weights_buf, ambig_buf, coords_buf, forces_buf,
        1.0, 0.5, 4, len(ambig) - 1,
    )

    (forces, _) = cl.enqueue_map_buffer(
        cq, forces_buf, cl.map_flags.READ,
        0, forces.shape, fp_type,
        wait_for=[e], is_blocking=True,
    )
    expected = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
        [0.0, 0.0,-0.5],
        [0.0, 0.0,-1.0],
        [0.0, 0.0,-1.0],
    ], dtype=fp_type)
    expected[0] = -expected[1:].sum(axis=0)

    np.testing.assert_allclose(forces[:, :3], expected)

@pytest.mark.parametrize("func, dtype", [
    ("atomic_fadd", 'float32'), ('atom_fadd', 'float64')
])
def test_atom_fadd(ctx, cq, func, dtype):
    cl_dtype = {'float32': 'float', 'float64': 'double'}[dtype]
    kernel = ("""
    kernel void test(global {dtype} * const sum, global const {dtype} * const inputs) {{
        {func}(sum, inputs[get_global_id(0)]);
    }}""".format(func=func, dtype=cl_dtype))
    with (Path(__file__).parent.parent / "nuc_dynamics" / "kernels.cl").open() as f:
        program = cl.Program(ctx, f.read() + kernel).build()

    values = np.ones(4000, dtype=dtype)

    values_buf = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR, hostbuf=values)
    sum_buf = cl.Buffer(
        ctx, cl.mem_flags.COPY_HOST_PTR, hostbuf=np.zeros(1, dtype=dtype)
    )
    e = program.test(cq, (len(values),), None, sum_buf, values_buf)

    (result, _) = cl.enqueue_map_buffer(
        cq, sum_buf, cl.map_flags.READ, 0, 1, dtype,
        wait_for=[e], is_blocking=True,
    )
    assert result[0] == values.sum()
