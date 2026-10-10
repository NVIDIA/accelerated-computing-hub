# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Actual complete BoxTask runs plus corrupted-evidence rejection.

Subprocesses keep the two articles' identically named tutorial modules isolated.
Native process-pool tests require normal OS shared-memory permission.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

PART2 = Path(__file__).resolve().parents[1] / 'notebooks' / 'mujoco' / 'part2'


def run_script(source, timeout=180):
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / 'check_box.py'
        script.write_text('import sys\nsys.path.insert(0, ' + repr(str(PART2)) + ')\n' + textwrap.dedent(source))
        env = dict(os.environ, PXR_WORK_THREAD_LIMIT='1')
        result = subprocess.run([sys.executable, str(script)], text=True, capture_output=True,
                                env=env, timeout=timeout)
        if result.returncode:
            raise AssertionError(result.stdout + '\n' + result.stderr)
        return result.stdout


class BoxBenchmarkTests(unittest.TestCase):
    def test_complete_native_two_cube_task_and_corrupted_evidence(self):
        run_script('''
            import numpy as np
            import benchmark_box as engine
            import benchmark_box_protocol as protocol
            if __name__ == '__main__':
                for robot in ('so101', 'rebot'):
                    config = engine._configuration(dict(robot=robot, repeats=1, warmups=1))
                    model, data, tape, phases, spec, workload = engine._prepare(config)
                    first = engine._NativeWorld(config, tape)
                    history = np.empty((1, 2000, 61), dtype=np.float32)
                    diagnostics = first.episode(history[0])
                    assert diagnostics['capacity_passed']
                    assert protocol.validate_observations(history, spec, phases, clock_precision='float64',
                        step_counts=[diagnostics['recorded_physics_steps']])['passed']
                    assert workload['initial_robot_ctrl'] == tape[0].tolist()
                    assert workload['physics_steps'] == 40000
                    # Parallel validation must reproduce the serial oracle exactly,
                    # including failures belonging to different world identities.
                    import os
                    from multiprocessing import shared_memory
                    histories = np.repeat(history, 4, axis=0)
                    before_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
                    os.environ['CUDA_VISIBLE_DEVICES'] = 'parent-test-value'
                    validator = protocol.ObservationValidator(spec, phases, 4, 2)
                    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'parent-test-value'
                    assert validator.info['workers'] == 2
                    assert validator.info['child_history_readonly']
                    assert validator.info['child_environment']['CUDA_VISIBLE_DEVICES'] == ''
                    allocation = validator.memory.name
                    workers = list(validator.pool._pool)
                    try:
                        for corrupted in (False, True):
                            counts = [40000]*4
                            if corrupted:
                                counts[1] = 39999
                                histories[2, 1000:, 0] = histories[2, 999, 0]
                                histories[3, -60:, 27] = .1
                            expected = protocol.validate_observations(histories, spec, phases,
                                clock_precision='float64', step_counts=counts)
                            measured = validator.validate(histories, clock_precision='float64', step_counts=counts)
                            assert measured == expected
                            assert measured['task_success_count'] == (1 if corrupted else 4)
                        # A worker-side exception must propagate; close still reaps
                        # every process and unlinks the shared memory allocation.
                        try:
                            validator.validate(histories, clock_precision='unsupported', step_counts=counts)
                        except ValueError:
                            pass
                        else:
                            raise AssertionError('worker error was hidden')
                    finally:
                        validator.close()
                        validator.close()  # idempotent finally blocks are safe
                        if before_cuda is None:
                            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                        else:
                            os.environ['CUDA_VISIBLE_DEVICES'] = before_cuda
                    assert all(not worker.is_alive() for worker in workers)
                    try:
                        leaked = shared_memory.SharedMemory(name=allocation)
                    except FileNotFoundError:
                        pass
                    else:
                        leaked.close()
                        raise AssertionError('validation shared memory leaked')
                    serial = protocol.ObservationValidator(spec, phases, 1, 32)
                    try:
                        assert serial.info['mode'] == 'serial_parent' and serial.info['workers'] == 1
                        assert serial.memory is None and serial.pool is None
                        assert serial.validate(history, clock_precision='float64', step_counts=[40000]) == \
                            protocol.validate_observations(history, spec, phases, clock_precision='float64',step_counts=[40000])
                    finally:
                        serial.close()

                    # Mutate just world 1: no successful world can hide a failed one.
                    for defect in ('missing_blue', 'no_right_contact', 'no_right_force', 'drag',
                                   'drop_in_transport', 'no_opening', 'corner_outside', 'moving',
                                   'stalled_time', 'nan', 'tool_not_clear'):
                        damaged = np.repeat(history, 2, axis=0)
                        row = damaged[1]
                        if defect == 'missing_blue':
                            row[:, 32:56] = row[0, 32:56]
                        elif defect == 'no_right_contact':
                            row[:, 29] = 0
                        elif defect == 'no_right_force':
                            row[:, 31] = 0
                        elif defect == 'drag':
                            selected = np.array([str(p).startswith('red_cube:') and
                                str(p).split(':')[1] in ('lift', 'carry', 'hold') for p in phases])
                            row[selected, 5:27:3] = spec.table_top_z + .005
                        elif defect == 'drop_in_transport':
                            index = next(i for i,p in enumerate(phases) if p == 'red_cube:carry')
                            row[index, 28:32] = 0
                        elif defect == 'no_opening':
                            row[:, 1] = 0
                        elif defect == 'corner_outside':
                            row[-60:, 3] += 1
                        elif defect == 'moving':
                            row[-60:, 27] = .1
                        elif defect == 'stalled_time':
                            row[40, 0] = row[39, 0]
                        elif defect == 'nan':
                            row[90, 32] = np.nan
                        else:
                            row[-1, 2] = 0
                        result = protocol.validate_observations(damaged, spec, phases, clock_precision='float64',
                            step_counts=[diagnostics['recorded_physics_steps']]*2)
                        assert not result['passed'], (robot, defect)
                        assert result['task_success_count'] == 1, (robot, defect, result)
                print('both robots: complete true-contact episodes and all negative evidence gates passed')
        ''')

    def test_persistent_pool_resets_every_world_and_every_episode(self):
        run_script('''
            import json
            from benchmark_workload import run_case
            if __name__ == '__main__':
                for robot in ('so101', 'rebot'):
                    row = run_case(dict(robot=robot, backend='mujoco', worlds=3,
                                        cpu_threads=2, repeats=2, warmups=1))
                    assert row['status'] == 'passed', row['diagnostics']
                    assert row['config']['task'] == 'box'
                    assert row['device_info']['cpu_workers'] == 2
                    assert row['host_validation']['workers'] == 2
                    assert row['host_validation']['mode'] == 'persistent_spawn_pool'
                    assert row['timings']['validation_setup_seconds'] > 0
                    assert row['timings']['validation_teardown_seconds'] > 0
                    assert row['workload']['host_validation_policy']['requested_workers'] == 2
                    assert len(row['warmup_samples']) == 1 and len(row['samples']) == 2
                    samples = row['warmup_samples'] + row['samples']
                    reference = samples[0]['validation']['worlds'][0]['box_task']
                    for sample in samples:
                        assert sample['task_success_count'] == sample['task_total_count'] == 3
                        assert sample['diagnostics']['completed_worlds'] == 3
                        assert sample['diagnostics']['integration_steps_per_world'] == [40000]*3
                        assert all(w['clock']['passed'] for w in sample['validation']['worlds'])
                        assert sample['simulation_seconds'] > 0
                        assert sample['output_transfer_seconds'] >= 0 and sample['validation_seconds'] > 0
                        for world in sample['validation']['worlds']:
                            assert world['box_task'] == reference
                    json.dumps(row, allow_nan=False)
                for change in ({'frames':600}, {'substeps':9}, {'warmups':0},
                               {'worlds':0}, {'validation_workers':0}, {'max_trajectory_mib':.01}, {'task':'unknown'},
                               {'backend':'mjwarp', 'device':'cpu'}):
                    assert run_case(change)['status'] == 'failed', change
        ''')

    def test_clock_contract_rejects_missing_duplicate_wrong_dt_and_frozen_steps(self):
        run_script('''
            import numpy as np
            import benchmark_box_protocol as protocol
            for precision in ('float32', 'float64'):
                dtype = np.dtype(precision)
                # Independent accumulation oracle, sampled only after actual steps.
                times = np.cumsum(np.full(40000, .001, dtype=dtype), dtype=dtype)[19::20].astype('float32')
                assert protocol.validate_clock(times, precision, 40000)['passed']
                if precision == 'float32':
                    assert abs(float(times[-1])-40) > .01  # legitimate accumulator drift
                for count in (39999, 40001, 20000, None, True, 40000.0):
                    assert not protocol.validate_clock(times, precision, count)['passed'], count
                for defect in ('missing_step', 'duplicate_step', 'frozen', 'wrong_dt', 'nominal_replacement', 'nan'):
                    bad = times.copy()
                    if defect == 'missing_step':
                        bad[1000:] -= .001
                    elif defect == 'duplicate_step':
                        bad[1000:] += .001
                    elif defect == 'frozen':
                        bad[1000:] = bad[999]
                    elif defect == 'wrong_dt':
                        bad *= 2
                    elif defect == 'nominal_replacement':
                        if precision == 'float64':
                            continue
                        bad = np.arange(1, 2001, dtype=np.float64).astype('float32') * np.float32(.02)
                    else:
                        bad[1000] = np.nan
                    assert not protocol.validate_clock(bad, precision, 40000)['passed'], (precision, defect)
                # Exactly one stored-observation ULP is allowed; two are rejected.
                one = times.copy()
                one[1200] = np.nextafter(one[1200], np.float32(np.inf))
                assert protocol.validate_clock(one, precision, 40000)['passed']
                one[1200] = np.nextafter(one[1200], np.float32(np.inf))
                assert not protocol.validate_clock(one, precision, 40000)['passed']
            print('all raw-clock and independent-count gates passed')
        ''')

    def test_device_counter_excludes_observation_forward_and_options_are_actual(self):
        run_script('''
            import mujoco
            import mujoco_warp as mjw
            import numpy as np
            import warp as wp
            import benchmark_box as engine
            import benchmark_box_protocol as protocol
            wp.init()
            with wp.ScopedDevice('cpu'):
                qpos = wp.zeros((2, 3), dtype=float)
                qvel = wp.zeros((2, 3), dtype=float)
                bad = wp.zeros(2, dtype=int)
                counts = wp.zeros(2, dtype=int)
                for integration_step in (0, 1, 1, 0, 1, 0):
                    wp.launch(engine._finite_state, 2, inputs=[qpos, qvel, bad, integration_step, counts])
                assert counts.numpy().tolist() == [3, 3]
                assert bad.numpy().tolist() == [0, 0]
                native = mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.001" tolerance="1e-8"/><worldbody><body><freejoint/><geom type="sphere" size=".02"/></body></worldbody></mujoco>')
                uploaded = mjw.put_model(native)
                metadata = protocol.backend_option_metadata(native, uploaded)
                assert metadata['native_requested']['tolerance'] == 1e-8
                tolerance = metadata['uploaded_gpu_effective']['tolerance']
                if isinstance(tolerance, dict):
                    tolerance = tolerance['uniform_value']
                assert np.isclose(tolerance, 1e-6, rtol=1e-6, atol=0)
                assert metadata['native_requested']['disableflags'] == 0
                assert metadata['uploaded_gpu_effective']['disableflags'] == 0
                assert native.opt.tolerance == 1e-8
                import json
                json.dumps(metadata, allow_nan=False)
        ''')


if __name__ == '__main__':
    unittest.main()
